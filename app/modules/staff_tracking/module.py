"""
Staff Tracking Module — AnalyticsModule implementation
=======================================================
Integrates staff/employee tracking into the orchestrator pipeline.

Receives frames and shared person_tracks from FrameContext (no own
detection), runs face recognition to identify employees, monitors
zone activity and idle behaviour, tracks queue length, and returns
AnalyticsEvent objects for the event publisher.

Events emitted:
    employee_idle          — employee stationary beyond idle_window frames
    employee_on_break      — employee entered a designated break zone
    employee_offline       — previously seen employee absent from all frames
    long_queue             — queue zone headcount exceeds threshold
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity


# ── Inline analytics helpers ───────────────────────────────────────────
# These replicate the analytics.* submodules that the original pipeline
# imported but were not included in the zip. Kept here to avoid adding
# new external dependencies on the orchestrator side.

class _ZoneManager:
    """Polygon-based zone classifier loaded from a zones.json config."""

    def __init__(self, zones_dict: dict, break_zones: list[str]):
        # Only keep zones that have at least 3 points (valid polygon)
        self.zones = {
            name: [tuple(p) for p in coords]
            for name, coords in zones_dict.items()
            if len(coords) >= 3
        }
        self.break_zones = set(break_zones)

    def get_zone(self, cx: int, cy: int) -> str:
        for name, pts in self.zones.items():
            if Polygon(pts).contains(Point(cx, cy)):
                return name
        return "outside"

    def is_break_zone(self, zone: str) -> bool:
        return zone in self.break_zones


class _IdleDetector:
    """
    Flags a track as idle when its movement over the last `window` frames
    stays below `threshold` pixels (Euclidean range across all positions).
    """

    def __init__(self, window: int = 150, threshold: float = 15.0):
        self._history: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.window = window
        self.threshold = threshold

    def update(self, track_id: int, center: tuple[int, int]) -> bool:
        hist = self._history[track_id]
        hist.append(center)
        if len(hist) > self.window:
            hist.pop(0)
        if len(hist) < self.window:
            return False
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        movement = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return movement < self.threshold

    def is_idle(self, track_id: int) -> bool:
        """Non-mutating idle check for annotation use."""
        hist = self._history.get(track_id, [])
        if len(hist) < self.window:
            return False
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        movement = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return movement < self.threshold

    def cleanup(self, active_ids: list[int]) -> None:
        for tid in list(self._history):
            if tid not in active_ids:
                del self._history[tid]


class _QueueAnalyzer:
    """Counts people in the queue zone and maintains a running history."""

    def __init__(self, zone_manager: _ZoneManager, queue_zone: str = "queue"):
        self.zone_manager = zone_manager
        self.queue_zone = queue_zone
        self._history: list[int] = []

    def count(self, person_tracks: dict) -> int:
        total = 0
        for tdata in person_tracks.values():
            b = tdata["bbox"]
            cx = (int(b[0]) + int(b[2])) // 2
            cy = (int(b[1]) + int(b[3])) // 2
            if self.zone_manager.get_zone(cx, cy) == self.queue_zone:
                total += 1
        return total

    def update(self, count: int) -> None:
        self._history.append(count)

    def report(self) -> dict:
        if not self._history:
            return {"max_queue": 0, "avg_queue": 0.0, "samples": 0}
        return {
            "max_queue": max(self._history),
            "avg_queue": round(sum(self._history) / len(self._history), 2),
            "samples": len(self._history),
        }


class _FaceRecognizer:
    """
    Matches a live InsightFace embedding against the pickled employee
    embeddings database.  Mirrors Identity/face_recognizer.py but
    accepts paths from config rather than hardcoded project root.
    """

    def __init__(
        self,
        embeddings_path: str,
        employees_config: str,
        threshold: float = 0.9,
    ):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings not found: {embeddings_path}. "
                "Run build_employee_db.py first."
            )
        with open(embeddings_path, "rb") as f:
            self.database: dict[str, np.ndarray] = pickle.load(f)
        with open(employees_config) as f:
            self.employee_map: dict[str, str] = json.load(f)
        self.threshold = threshold

    def recognize(self, embedding: np.ndarray) -> str:
        best_name, best_dist = None, float("inf")
        for name, db_emb in self.database.items():
            dist = float(np.linalg.norm(embedding - db_emb))
            if dist < best_dist:
                best_dist, best_name = dist, name
        if best_name and best_dist < self.threshold:
            return self.employee_map.get(best_name, best_name)
        return "unknown"


# ── Module ────────────────────────────────────────────────────────────

class StaffTrackingModule(AnalyticsModule):
    """
    Staff activity supervision module for the NexaSight orchestrator.

    Uses shared person_tracks from FrameContext — no duplicate detection.
    Runs InsightFace every `face_detect_interval` frames to identify
    employees, then monitors zone activity, idle behaviour, and queue
    length per camera.
    """

    def __init__(self):
        self._config: dict = {}
        self._cameras: list[str] = ["*"]

        # Analytics helpers — built during initialize()
        self._zone_manager: _ZoneManager | None = None
        self._idle_detector: _IdleDetector = _IdleDetector()
        self._queue_analyzer: _QueueAnalyzer | None = None
        self._face_recognizer: _FaceRecognizer | None = None
        self._face_app: Any = None          # InsightFace FaceAnalysis

        # Per-camera state (reset on each camera in a new session)
        # track_id -> employee_id string
        self._identity_map: dict[int, str] = {}
        # employee_id -> last known status string
        self._last_status: dict[str, str] = {}
        # All employee IDs seen so far this session
        self._seen_employees: set[str] = set()

        # Frame-level counters
        self._frame_count: int = 0
        self._face_interval: int = 15
        self._queue_threshold: int = 5

        # Alert cooldowns (frame-based)
        self._idle_cooldown: dict[str, int] = {}     # employee_id -> last alert frame
        self._queue_cooldown_frame: int = -9999
        self._idle_cooldown_frames: int = 75
        self._queue_cooldown_frames: int = 75

    # ── Contract: required properties ────────────────────────────────

    @property
    def name(self) -> str:
        return "staff_tracking"

    # ── Contract: required methods ────────────────────────────────────

    def initialize(self, config: dict) -> None:
        """
        Load zone config, InsightFace model, and employee embeddings.

        Expected config keys (all optional with defaults):
            cameras               : list[str]  — camera IDs to run on, default ["*"]
            zones_config          : str        — path to zones.json
            face_detect_interval  : int        — run face ID every N frames (default 15)
            queue_threshold       : int        — persons in queue to trigger alert (default 5)
            idle_window_frames    : int        — frames to assess idle state (default 150)
            idle_movement_threshold: float     — pixel movement range to call idle (default 15)
            idle_cooldown_frames  : int        — min frames between idle alerts (default 75)
            queue_cooldown_frames : int        — min frames between queue alerts (default 75)
            embeddings_path       : str        — path to employee_embeddings.pkl
            employees_config      : str        — path to employees.json name→ID map
            recognition_threshold : float      — L2 distance threshold (default 0.9)
        """
        self._config = config
        self._cameras = config.get("cameras", ["*"])
        self._face_interval = config.get("face_detect_interval", 15)
        self._queue_threshold = config.get("queue_threshold", 5)
        self._idle_cooldown_frames = config.get("idle_cooldown_frames", 75)
        self._queue_cooldown_frames = config.get("queue_cooldown_frames", 75)

        # ── Zone manager ──────────────────────────────────────────────
        zone_cfg_path = config.get("zones_config", "configs/zones.json")
        try:
            with open(zone_cfg_path) as f:
                zone_data = json.load(f)
            self._zone_manager = _ZoneManager(
                zone_data.get("zones", {}),
                zone_data.get("break_zones", []),
            )
            print(f"[StaffTracking] Zones loaded: {list(self._zone_manager.zones)}")
        except Exception as e:
            print(f"[StaffTracking] WARNING: Could not load zones config: {e}")

        # ── Idle detector + queue analyzer ────────────────────────────
        self._idle_detector = _IdleDetector(
            window=config.get("idle_window_frames", 150),
            threshold=config.get("idle_movement_threshold", 15.0),
        )
        if self._zone_manager:
            self._queue_analyzer = _QueueAnalyzer(self._zone_manager)

        # ── InsightFace ───────────────────────────────────────────────
        try:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(
                allowed_modules=["detection", "recognition"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_app.prepare(ctx_id=0)
            print("[StaffTracking] InsightFace loaded")
        except Exception as e:
            print(f"[StaffTracking] WARNING: InsightFace unavailable: {e}")

        # ── Face recognizer ───────────────────────────────────────────
        embeddings_path = config.get(
            "embeddings_path", "data/employee_embeddings.pkl"
        )
        employees_config = config.get(
            "employees_config", "configs/employees.json"
        )
        try:
            self._face_recognizer = _FaceRecognizer(
                embeddings_path,
                employees_config,
                threshold=config.get("recognition_threshold", 0.9),
            )
            print(
                f"[StaffTracking] Employee DB loaded — "
                f"{len(self._face_recognizer.database)} employees registered"
            )
        except FileNotFoundError as e:
            print(f"[StaffTracking] WARNING: {e} — face recognition disabled")

    def applicable_cameras(self) -> list[str]:
        return self._cameras

    def process_frame(
        self, frame: np.ndarray, context: FrameContext
    ) -> list[AnalyticsEvent]:
        """
        Analyse one frame and return staff-related events.

        Uses context.person_tracks (already populated by the orchestrator's
        shared person detector) — no own YOLO call needed.
        """
        self._frame_count += 1
        events: list[AnalyticsEvent] = []
        person_tracks = context.person_tracks

        # ── Face recognition pass (every N frames) ────────────────────
        if (
            self._frame_count % self._face_interval == 0
            and self._face_app is not None
            and self._face_recognizer is not None
            and len(person_tracks) > 0
        ):
            self._run_face_recognition(frame, person_tracks)

        active_ids = list(person_tracks.keys())

        # ── Per-person analysis ───────────────────────────────────────
        for tid, tdata in person_tracks.items():
            b = tdata["bbox"]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            identity = self._identity_map.get(tid, "unknown")
            zone = (
                self._zone_manager.get_zone(cx, cy)
                if self._zone_manager else "unknown"
            )
            is_idle = self._idle_detector.update(tid, (cx, cy))

            if identity == "unknown":
                continue

            self._seen_employees.add(identity)

            # Resolve current status
            if self._zone_manager and self._zone_manager.is_break_zone(zone):
                status = "ON_BREAK"
            elif is_idle:
                status = "IDLE"
            else:
                status = "ON_DUTY"

            prev_status = self._last_status.get(identity)

            # Emit idle alert (with cooldown to prevent spam)
            if status == "IDLE" and (
                prev_status != "IDLE"
                or self._frame_count - self._idle_cooldown.get(identity, -9999)
                >= self._idle_cooldown_frames
            ):
                events.append(AnalyticsEvent(
                    module=self.name,
                    camera_id=context.camera_id,
                    timestamp=context.timestamp,
                    event_type="employee_idle",
                    confidence=1.0,
                    bbox=[x1, y1, x2, y2],
                    severity=Severity.MEDIUM,
                    frame_idx=context.frame_idx,
                    person_id=tid,
                    metadata={"employee_id": identity, "zone": zone},
                ))
                self._idle_cooldown[identity] = self._frame_count

            # Emit break alert on zone entry (once per transition)
            elif status == "ON_BREAK" and prev_status != "ON_BREAK":
                events.append(AnalyticsEvent(
                    module=self.name,
                    camera_id=context.camera_id,
                    timestamp=context.timestamp,
                    event_type="employee_on_break",
                    confidence=1.0,
                    bbox=[x1, y1, x2, y2],
                    severity=Severity.LOW,
                    frame_idx=context.frame_idx,
                    person_id=tid,
                    metadata={"employee_id": identity, "zone": zone},
                ))

            self._last_status[identity] = status

        # ── Cleanup stale idle tracking entries ───────────────────────
        self._idle_detector.cleanup(active_ids)

        # ── Queue alert ───────────────────────────────────────────────
        if self._queue_analyzer:
            q_count = self._queue_analyzer.count(person_tracks)
            self._queue_analyzer.update(q_count)

            if (
                q_count > self._queue_threshold
                and self._frame_count - self._queue_cooldown_frame
                >= self._queue_cooldown_frames
            ):
                events.append(AnalyticsEvent(
                    module=self.name,
                    camera_id=context.camera_id,
                    timestamp=context.timestamp,
                    event_type="long_queue",
                    confidence=1.0,
                    bbox=[0, 0, context.frame_width, context.frame_height],
                    severity=Severity.HIGH,
                    frame_idx=context.frame_idx,
                    metadata={
                        "queue_count": q_count,
                        "threshold": self._queue_threshold,
                    },
                ))
                self._queue_cooldown_frame = self._frame_count

        # ── Offline check ─────────────────────────────────────────────
        # Employees seen earlier in this session but absent this frame
        if self._seen_employees:
            current_identities = {
                self._identity_map.get(tid, "unknown")
                for tid in active_ids
            } - {"unknown"}

            for emp_id in self._seen_employees - current_identities:
                if self._last_status.get(emp_id) != "OFFLINE":
                    events.append(AnalyticsEvent(
                        module=self.name,
                        camera_id=context.camera_id,
                        timestamp=context.timestamp,
                        event_type="employee_offline",
                        confidence=1.0,
                        bbox=[0, 0, 1, 1],
                        severity=Severity.LOW,
                        frame_idx=context.frame_idx,
                        metadata={"employee_id": emp_id},
                    ))
                    self._last_status[emp_id] = "OFFLINE"

        return events

    def shutdown(self) -> None:
        self._identity_map.clear()
        self._seen_employees.clear()
        self._last_status.clear()
        print("[StaffTracking] Module shut down.")

    # ── Optional: custom frame annotation ────────────────────────────

    def annotate_frame(
        self, frame: np.ndarray, context: FrameContext
    ) -> np.ndarray:
        """
        Draw employee identity labels and idle state on the frame.
        Called by the engine after all modules have processed the frame.
        """
        annotated = frame.copy()

        for tid, tdata in context.person_tracks.items():
            b = tdata["bbox"]
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            identity = self._identity_map.get(tid, "?")
            zone = (
                self._zone_manager.get_zone(cx, cy)
                if self._zone_manager else "?"
            )
            idle = self._idle_detector.is_idle(tid)

            color = (0, 0, 255) if idle else (0, 255, 0)
            label = f"{identity} | {zone}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
            if idle:
                cv2.putText(
                    annotated, "IDLE", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )

        return annotated

    # ── Optional: export session artifacts ───────────────────────────

    def export_artifacts(self, session_dir: Path) -> None:
        """Save the queue analysis report for this session."""
        if self._queue_analyzer:
            report = self._queue_analyzer.report()
            path = Path(session_dir) / "staff_tracking_report.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[StaffTracking] Report saved: {path}")

    # ── Internal helpers ──────────────────────────────────────────────

    def _run_face_recognition(
        self, frame: np.ndarray, person_tracks: dict
    ) -> None:
        """
        Run InsightFace on the frame and assign identities to matching tracks.
        Only updates tracks that are still 'unknown'.
        """
        try:
            faces = self._face_app.get(frame)
        except Exception as e:
            print(f"[StaffTracking] Face detection error: {e}")
            return

        for face in faces:
            identity = self._face_recognizer.recognize(face.embedding)
            if identity == "unknown":
                continue

            # Match face bbox to the person track that contains it
            fx1, fy1, fx2, fy2 = face.bbox.astype(int).tolist()
            best_tid, best_area = None, 0

            for tid, tdata in person_tracks.items():
                # Skip already-identified tracks
                if self._identity_map.get(tid, "unknown") != "unknown":
                    continue
                px1, py1, px2, py2 = [int(v) for v in tdata["bbox"]]
                if fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2:
                    area = (fx2 - fx1) * (fy2 - fy1)
                    if area > best_area:
                        best_area, best_tid = area, tid

            if best_tid is not None:
                self._identity_map[best_tid] = identity
                print(f"[StaffTracking] Identified track {best_tid} as {identity}")

"""
Parking & Traffic Analytics Module — AnalyticsModule implementation
====================================================================
Integrates parking lot vehicle analytics into the orchestrator pipeline.

Unlike staff_tracking, this module runs its OWN vehicle detector because:
  - The shared PersonDetector targets class 0 (people) only
  - Parking needs classes 2,3,5,7 (car, motorcycle, bus, truck)
  - Tiled inference is required for overhead parking footage where
    closely-parked vehicles merge into one detection at full resolution

Events emitted:
    parking_status_update  — periodic occupancy snapshot (every N frames)
    parking_limited        — occupancy crossed 70% threshold
    parking_full           — occupancy crossed 90% threshold
    long_dwell             — vehicle parked beyond dwell_alert_minutes
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity


# ── Inlined analytics helpers (from original analytics/ folder) ────────

class _IouTracker:
    def __init__(self, iou_threshold: float = 0.15, max_age: int = 20):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict = {}
        self.next_id = 0

    def _iou(self, a, b) -> float:
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        aA = (a[2] - a[0]) * (a[3] - a[1])
        aB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(aA + aB - inter + 1e-6)

    def update(self, detections: list) -> list:
        matched = set()
        for det in detections:
            bbox = det["bbox"]
            best_tid, best_iou = None, 0
            for tid, track in self.tracks.items():
                iou = self._iou(bbox, track["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_tid = iou, tid
            if best_tid is None:
                best_tid = self.next_id
                self.next_id += 1
            self.tracks[best_tid] = {"bbox": bbox, "missed": 0}
            matched.add(best_tid)
            det["track_id"] = best_tid
        stale = []
        for tid, t in self.tracks.items():
            if tid not in matched:
                t["missed"] += 1
                if t["missed"] > self.max_age:
                    stale.append(tid)
        for tid in stale:
            del self.tracks[tid]
        return detections

    def reset(self):
        self.tracks.clear()
        self.next_id = 0


class _OccupancyCalculator:
    CLASS_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, total_spaces: int):
        self.total_spaces = max(1, total_spaces)
        self.current_ids: set = set()
        self.total_seen: set = set()
        self._buffer: deque = deque(maxlen=10)

    def update(self, detections: list) -> dict:
        self.current_ids = {d["track_id"] for d in detections}
        self.total_seen.update(self.current_ids)
        self._buffer.append(len(self.current_ids))
        s = sorted(self._buffer)
        mid = len(s) // 2
        occupied = round((s[mid - 1] + s[mid]) / 2) if len(s) % 2 == 0 else s[mid]
        available = max(0, self.total_spaces - occupied)
        pct = round(occupied / self.total_spaces * 100, 1)
        status = "full" if pct >= 90 else ("limited" if pct >= 70 else "available")
        vtypes = {v: 0 for v in self.CLASS_MAP.values()}
        for det in detections:
            label = self.CLASS_MAP.get(det.get("class_id", -1))
            if label:
                vtypes[label] += 1
        return {
            "occupied": occupied,
            "available": available,
            "capacity": self.total_spaces,
            "occupancy_pct": pct,
            "status": status,
            "total_seen": len(self.total_seen),
            "vehicle_types": vtypes,
        }


class _DwellTracker:
    def __init__(self):
        self.entry_times: dict = {}

    def update(self, detections: list) -> dict:
        now = time.time()
        current_ids = {d["track_id"] for d in detections}
        for tid in current_ids:
            if tid not in self.entry_times:
                self.entry_times[tid] = now
        for tid in set(self.entry_times) - current_ids:
            del self.entry_times[tid]
        dwell = {tid: round(now - t, 1) for tid, t in self.entry_times.items()}
        avg = round(sum(dwell.values()) / len(dwell), 1) if dwell else 0.0
        return {"per_vehicle": dwell, "average_seconds": avg}

    def reset(self):
        self.entry_times.clear()


class _VehicleDetector:
    """
    Tiled YOLOv8 detector with CLAHE preprocessing.
    Identical to the original models/detector.py but accepts config dict.
    """
    VEHICLE_CLASSES = [2, 3, 5, 7]

    def __init__(self, model_path: str = "yolov8n.pt",
                 conf_day: float = 0.25, conf_night: float = 0.20,
                 low_light_threshold: int = 60,
                 tile_rows: int = 2, tile_cols: int = 2,
                 tile_overlap: float = 0.2,
                 nms_iou: float = 0.3):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_day = conf_day
        self.conf_night = conf_night
        self.low_light = low_light_threshold
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tile_overlap = tile_overlap
        self.nms_iou = nms_iou
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([self.clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def _conf(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.conf_night if gray.mean() < self.low_light else self.conf_day

    def _tiles(self, frame):
        H, W = frame.shape[:2]
        sy = H // self.tile_rows
        sx = W // self.tile_cols
        oy = int(sy * self.tile_overlap)
        ox = int(sx * self.tile_overlap)
        tiles = []
        for r in range(self.tile_rows):
            for c in range(self.tile_cols):
                y1 = max(0, r * sy - oy)
                y2 = min(H, y1 + sy + oy * 2)
                x1 = max(0, c * sx - ox)
                x2 = min(W, x1 + sx + ox * 2)
                tiles.append((frame[y1:y2, x1:x2], x1, y1))
        return tiles

    def _nms(self, dets):
        if not dets:
            return []
        boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
        scores = np.array([d["confidence"] for d in dets], dtype=np.float32)
        bxywh = boxes.copy()
        bxywh[:, 2] -= boxes[:, 0]
        bxywh[:, 3] -= boxes[:, 1]
        idxs = cv2.dnn.NMSBoxes(bxywh.tolist(), scores.tolist(),
                                 self.conf_day, self.nms_iou)
        return [dets[i] for i in idxs.flatten()] if len(idxs) else []

    def detect(self, frame) -> list:
        enhanced = self._preprocess(frame)
        conf = self._conf(frame)
        all_dets = []
        for tile, xo, yo in self._tiles(enhanced):
            results = self.model(tile, conf=conf, verbose=False)[0]
            if results.boxes is None:
                continue
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls not in self.VEHICLE_CLASSES:
                    continue
                tx1, ty1, tx2, ty2 = map(int, box.xyxy[0])
                all_dets.append({
                    "bbox": [tx1 + xo, ty1 + yo, tx2 + xo, ty2 + yo],
                    "confidence": float(box.conf[0]),
                    "class_id": cls,
                })
        return self._nms(all_dets)

    def shutdown(self):
        del self.model


# ── Module ─────────────────────────────────────────────────────────────

class ParkingModule(AnalyticsModule):
    """
    Parking lot vehicle analytics for the NexaSight orchestrator.

    Runs its own tiled vehicle detector per frame (does NOT use
    context.person_tracks — those are people, not vehicles).
    """

    def __init__(self):
        self._config: dict = {}
        self._cameras: list[str] = ["*"]
        self._detector: _VehicleDetector | None = None
        self._tracker: _IouTracker = _IouTracker()
        self._occupancy: _OccupancyCalculator | None = None
        self._dwell: _DwellTracker = _DwellTracker()

        # Alert state
        self._last_status: str = "available"
        self._status_update_interval: int = 30   # frames between periodic updates
        self._frame_count: int = 0
        self._dwell_alert_seconds: float = 3600  # 1 hour default
        self._alerted_dwell_ids: set = set()

        # Session summary
        self._snapshot_history: list = []

    @property
    def name(self) -> str:
        return "parking"

    def initialize(self, config: dict) -> None:
        """
        Expected config keys:
            cameras                  : list[str]
            model_path               : str    — yolov8n.pt or custom
            total_spaces             : int    — total defined parking spaces
            conf_day                 : float  — detection confidence (day)
            conf_night               : float  — detection confidence (night)
            tile_rows                : int
            tile_cols                : int
            tile_overlap             : float
            status_update_interval   : int    — frames between periodic events
            dwell_alert_minutes      : float  — alert when vehicle parked this long
        """
        self._config = config
        self._cameras = config.get("cameras", ["*"])

        self._detector = _VehicleDetector(
            model_path=config.get("model_path", "yolov8n.pt"),
            conf_day=config.get("conf_day", 0.25),
            conf_night=config.get("conf_night", 0.20),
            tile_rows=config.get("tile_rows", 2),
            tile_cols=config.get("tile_cols", 2),
            tile_overlap=config.get("tile_overlap", 0.2),
        )
        self._tracker = _IouTracker(
            iou_threshold=config.get("iou_threshold", 0.15),
            max_age=config.get("tracker_max_age", 20),
        )
        self._occupancy = _OccupancyCalculator(
            total_spaces=config.get("total_spaces", 20)
        )
        self._dwell = _DwellTracker()
        self._status_update_interval = config.get("status_update_interval", 30)
        self._dwell_alert_seconds = config.get("dwell_alert_minutes", 60) * 60

        print(
            f"[ParkingModule] Ready — "
            f"total_spaces={config.get('total_spaces', 20)}, "
            f"model={config.get('model_path', 'yolov8n.pt')}"
        )

    def applicable_cameras(self) -> list[str]:
        return self._cameras

    def process_frame(
        self, frame: np.ndarray, context: FrameContext
    ) -> list[AnalyticsEvent]:

        self._frame_count += 1
        events: list[AnalyticsEvent] = []

        # ── Detect + track vehicles ────────────────────────────────
        raw = self._detector.detect(frame)
        detections = self._tracker.update(raw)

        # ── Occupancy ──────────────────────────────────────────────
        occ = self._occupancy.update(detections)
        current_status = occ["status"]

        # ── Status-change alerts ───────────────────────────────────
        if current_status != self._last_status:
            if current_status == "full":
                events.append(AnalyticsEvent(
                    module=self.name,
                    camera_id=context.camera_id,
                    timestamp=context.timestamp,
                    event_type="parking_full",
                    confidence=1.0,
                    bbox=[0, 0, context.frame_width, context.frame_height],
                    severity=Severity.HIGH,
                    frame_idx=context.frame_idx,
                    metadata={
                        "occupied": occ["occupied"],
                        "capacity": occ["capacity"],
                        "occupancy_pct": occ["occupancy_pct"],
                        "vehicle_types": occ["vehicle_types"],
                    },
                ))
            elif current_status == "limited":
                events.append(AnalyticsEvent(
                    module=self.name,
                    camera_id=context.camera_id,
                    timestamp=context.timestamp,
                    event_type="parking_limited",
                    confidence=1.0,
                    bbox=[0, 0, context.frame_width, context.frame_height],
                    severity=Severity.MEDIUM,
                    frame_idx=context.frame_idx,
                    metadata={
                        "occupied": occ["occupied"],
                        "capacity": occ["capacity"],
                        "occupancy_pct": occ["occupancy_pct"],
                        "vehicle_types": occ["vehicle_types"],
                    },
                ))
            self._last_status = current_status

        # ── Periodic status snapshot ───────────────────────────────
        if self._frame_count % self._status_update_interval == 0:
            dwell = self._dwell.update(detections)
            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type="parking_status_update",
                confidence=1.0,
                bbox=[0, 0, context.frame_width, context.frame_height],
                severity=Severity.LOW,
                frame_idx=context.frame_idx,
                metadata={
                    "occupied": occ["occupied"],
                    "available": occ["available"],
                    "capacity": occ["capacity"],
                    "occupancy_pct": occ["occupancy_pct"],
                    "status": current_status,
                    "avg_dwell_seconds": dwell["average_seconds"],
                    "total_vehicles_seen": occ["total_seen"],
                    "vehicle_types": occ["vehicle_types"],
                },
            ))
            self._snapshot_history.append({
                "timestamp": context.timestamp,
                "occupied": occ["occupied"],
                "occupancy_pct": occ["occupancy_pct"],
                "status": current_status,
            })

        # ── Long dwell alerts ──────────────────────────────────────
        dwell = self._dwell.update(detections)
        for tid, seconds in dwell["per_vehicle"].items():
            if seconds >= self._dwell_alert_seconds and tid not in self._alerted_dwell_ids:
                det = next((d for d in detections if d["track_id"] == tid), None)
                bbox = det["bbox"] if det else [0, 0, 1, 1]
                events.append(AnalyticsEvent(
                    module=self.name,
                    camera_id=context.camera_id,
                    timestamp=context.timestamp,
                    event_type="long_dwell",
                    confidence=1.0,
                    bbox=bbox,
                    severity=Severity.MEDIUM,
                    frame_idx=context.frame_idx,
                    person_id=tid,
                    metadata={
                        "vehicle_id": tid,
                        "dwell_seconds": seconds,
                        "dwell_minutes": round(seconds / 60, 1),
                    },
                ))
                self._alerted_dwell_ids.add(tid)

        return events

    def shutdown(self) -> None:
        if self._detector:
            self._detector.shutdown()
        self._tracker.reset()
        self._dwell.reset()
        print(
            f"[ParkingModule] Session summary — "
            f"snapshots recorded: {len(self._snapshot_history)}"
        )

    # ── Optional: custom annotation ───────────────────────────────

    def annotate_frame(
        self, frame: np.ndarray, context: FrameContext
    ) -> np.ndarray:
        """Draw vehicle boxes and occupancy overlay."""
        annotated = frame.copy()
        occ = self._occupancy.update([])  # read last state without new detections
        label = (
            f"Parking: {occ['occupied']}/{occ['capacity']} "
            f"({occ['occupancy_pct']}%) [{occ['status'].upper()}]"
        )
        color = (0, 0, 255) if occ["status"] == "full" else \
                (0, 165, 255) if occ["status"] == "limited" else (0, 255, 0)
        cv2.putText(annotated, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return annotated

    # ── Optional: export session artifacts ────────────────────────

    def export_artifacts(self, session_dir: Path) -> None:
        if not self._snapshot_history:
            return
        path = Path(session_dir) / "parking_summary.json"
        summary = {
            "total_snapshots": len(self._snapshot_history),
            "peak_occupancy_pct": max(
                s["occupancy_pct"] for s in self._snapshot_history
            ),
            "avg_occupancy_pct": round(
                sum(s["occupancy_pct"] for s in self._snapshot_history)
                / len(self._snapshot_history), 1
            ),
            "snapshots": self._snapshot_history,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ParkingModule] Summary saved: {path}")

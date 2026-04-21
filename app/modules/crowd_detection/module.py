"""
Crowd Detection Module — AnalyticsModule Implementation
=========================================================
Wraps the CrowdTracker with the standard module contract.
Produces events for density changes, footfall, and exports
heat map + crowd summary at shutdown.

This module is zone-independent — it works on any camera feed
without requiring pre-defined ROI zones.
"""

from __future__ import annotations

import cv2
import torch
from pathlib import Path

from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity
from .crowd_tracker import CrowdTracker, DensityLevel
from .iou_tracker import IoUTracker


# ── Severity mapping by density level ─────────────────────────────────
DENSITY_SEVERITY = {
    DensityLevel.LOW: Severity.LOW,
    DensityLevel.MODERATE: Severity.LOW,
    DensityLevel.HIGH: Severity.MEDIUM,
    DensityLevel.CRITICAL: Severity.HIGH,
}


class CrowdDetectionModule(AnalyticsModule):
    """
    Detects crowd patterns — occupancy, footfall, density, and heatmaps.

    Features:
      - Real-time person counting (no ML model needed, uses shared detections)
      - Footfall tracking via frame-edge entry/exit detection
      - Spatial heat map accumulation
      - Crowd density level estimation with alert events
      - Movement trajectory recording
      - Video overlay with all crowd insights

    This module does NOT need its own YOLO model. It operates entirely
    on the shared person_tracks provided by the orchestrator.
    """

    def __init__(self):
        self._tracker: CrowdTracker | None = None
        self._config: dict = {}
        self._cameras: list[str] = ["*"]
        self._initialized: bool = False

        # Own person detector (optional — uses dedicated model for better crowd detection)
        self._own_detector = None           # YOLO model instance
        self._own_tracker_config: str | None = None
        self._own_conf: float = 0.10
        self._own_iou: float = 0.45
        self._own_imgsz: int = 1280
        self._use_own_detector: bool = False

        # SAHI (Sliced Aided Hyper Inference)
        self._use_sahi: bool = False
        self._sahi_model = None             # SAHI AutoDetectionModel
        self._sahi_slice_w: int = 640
        self._sahi_slice_h: int = 640
        self._sahi_overlap: float = 0.25
        self._iou_tracker: IoUTracker | None = None

        # Alert cooldowns
        self._last_density_alert_time: float = -999.0
        self._density_alert_cooldown: float = 10.0
        self._critical_alert_cooldown: float = 5.0
        self._alert_on_density_change: bool = True

        # Export paths (set at shutdown by engine's session_dir)
        self._session_dir: Path | None = None

    @property
    def name(self) -> str:
        return "crowd_detection"

    def initialize(self, config: dict) -> None:
        """
        Parse config and optionally load a dedicated person detection model.
        CrowdTracker creation is deferred until first frame (needs dimensions).
        """
        self._config = config
        self._cameras = config.get("cameras", ["*"])
        self._alert_on_density_change = config.get("alert_on_density_change", True)
        self._density_alert_cooldown = config.get("high_density_alert_sec", 10.0)
        self._critical_alert_cooldown = config.get("critical_density_alert_sec", 5.0)

        # ── Load dedicated person model (if configured) ───────────
        model_path = config.get("person_model_path", "")
        if model_path:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._own_conf = config.get("person_conf", 0.10)
            self._own_iou = config.get("person_iou", 0.45)
            self._own_imgsz = config.get("person_imgsz", 1280)
            self._own_tracker_config = config.get("tracker_config", None)

            # ── SAHI mode ────────────────────────────────────────
            self._use_sahi = config.get("use_sahi", False)

            if self._use_sahi:
                from sahi import AutoDetectionModel

                self._sahi_slice_w = config.get("sahi_slice_width", 640)
                self._sahi_slice_h = config.get("sahi_slice_height", 640)
                self._sahi_overlap = config.get("sahi_overlap_ratio", 0.25)

                print(f"[CrowdDetectionModule] Loading SAHI model: {model_path} on {device}")
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=model_path,
                    confidence_threshold=self._own_conf,
                    device=device,
                )
                # IoU tracker for persistent IDs (SAHI only does detection)
                self._iou_tracker = IoUTracker(
                    iou_threshold=config.get("sahi_track_iou", 0.25),
                    max_lost=config.get("sahi_track_max_lost", 30),
                    min_hits=1,
                )
                self._use_own_detector = True
                print(f"[CrowdDetectionModule] SAHI ready "
                      f"(slices={self._sahi_slice_w}x{self._sahi_slice_h}, "
                      f"overlap={self._sahi_overlap}, conf={self._own_conf})")
            else:
                # ── Direct model.track() mode ─────────────────────
                from ultralytics import YOLO
                print(f"[CrowdDetectionModule] Loading dedicated model: {model_path} on {device}")
                self._own_detector = YOLO(model_path).to(device)
                self._use_own_detector = True
                print(f"[CrowdDetectionModule] Dedicated detector ready "
                      f"(conf={self._own_conf}, iou={self._own_iou}, imgsz={self._own_imgsz})")
        else:
            print(f"[CrowdDetectionModule] Using shared person detections (no dedicated model)")

        print(f"[CrowdDetectionModule] Initialized (cameras={self._cameras})")

    def applicable_cameras(self) -> list[str]:
        """Crowd detection runs on all cameras by default."""
        return self._cameras

    def _ensure_tracker(self, context: FrameContext) -> None:
        """Lazy-initialize CrowdTracker on first frame (needs dimensions)."""
        if self._tracker is not None:
            return

        cfg = self._config
        self._tracker = CrowdTracker(
            frame_width=context.frame_width,
            frame_height=context.frame_height,
            fps=context.fps,
            density_low_max=cfg.get("density_low_max", 3),
            density_moderate_max=cfg.get("density_moderate_max", 8),
            density_high_max=cfg.get("density_high_max", 15),
            edge_margin_ratio=cfg.get("edge_margin_ratio", 0.05),
            entry_exit_cooldown_sec=cfg.get("entry_exit_cooldown_sec", 1.0),
            heatmap_resolution=cfg.get("heatmap_resolution", 100),
            heatmap_decay=cfg.get("heatmap_decay", 0.998),
            heatmap_alpha=cfg.get("heatmap_alpha", 0.4),
            trajectory_max_length=cfg.get("trajectory_max_length", 300),
            trajectory_draw_length=cfg.get("trajectory_draw_length", 60),
        )
        self._initialized = True
        print(f"[CrowdDetectionModule] Tracker created "
              f"({context.frame_width}x{context.frame_height} @ {context.fps:.1f}fps)")

    def process_frame(
        self,
        frame,
        context: FrameContext,
    ) -> list[AnalyticsEvent]:
        """
        Analyse crowd patterns in the current frame.

        Returns events for:
          - Density level changes (LOW→HIGH, etc.)
          - Footfall entries/exits (informational, LOW severity)
        """
        self._ensure_tracker(context)

        # ── Get person tracks ─────────────────────────────────────
        if self._use_own_detector and self._own_detector is not None:
            # Run our own detection with the dedicated (larger) model
            person_tracks = self._detect_persons(frame)
        else:
            # Use shared detections from the orchestrator
            person_tracks = context.person_tracks

        # Run the crowd tracker update
        snapshot, footfall_events, density_changed = self._tracker.update(
            frame_idx=context.frame_idx,
            timestamp=context.timestamp,
            person_tracks=person_tracks,
        )

        events: list[AnalyticsEvent] = []

        # ── Density change events ─────────────────────────────────
        if density_changed and self._alert_on_density_change:
            level = snapshot.density_level
            severity = DENSITY_SEVERITY.get(level, Severity.LOW)

            # Apply cooldown (stricter for critical)
            cooldown = (self._critical_alert_cooldown
                        if level == DensityLevel.CRITICAL
                        else self._density_alert_cooldown)

            if (context.timestamp - self._last_density_alert_time) >= cooldown:
                self._last_density_alert_time = context.timestamp

                # Use frame center as bbox for density events
                cx, cy = context.frame_width // 2, context.frame_height // 2
                bbox_size = min(context.frame_width, context.frame_height) // 4

                events.append(AnalyticsEvent(
                    module=self.name,
                    camera_id=context.camera_id,
                    timestamp=context.timestamp,
                    event_type="density_change",
                    confidence=1.0,
                    bbox=[cx - bbox_size, cy - bbox_size,
                          cx + bbox_size, cy + bbox_size],
                    severity=severity,
                    frame_idx=context.frame_idx,
                    metadata={
                        "density_level": level.value,
                        "person_count": snapshot.total_persons,
                        "peak_occupancy": self._tracker.peak_occupancy,
                    },
                ))

        # ── Footfall events (informational) ───────────────────────
        for ff in footfall_events:
            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type=f"person_{ff.direction}",
                confidence=1.0,
                bbox=[ff.position[0] - 20, ff.position[1] - 20,
                      ff.position[0] + 20, ff.position[1] + 20],
                severity=Severity.LOW,
                frame_idx=context.frame_idx,
                person_id=ff.track_id,
                metadata={
                    "direction": ff.direction,
                    "edge": ff.edge,
                    "total_entries": self._tracker.total_entries,
                    "total_exits": self._tracker.total_exits,
                },
            ))

        return events

    def annotate_frame(self, frame, context: FrameContext):
        """
        Draw crowd overlay on the video frame.
        Called by the engine after process_frame.
        """
        if self._tracker is not None:
            return self._tracker.draw_overlay(frame)
        return frame

    def _detect_persons(self, frame) -> dict[int, dict]:
        """
        Run person detection on the frame.

        Two modes:
          1. SAHI: Slice frame into tiles → detect on each → merge → IoU track
          2. Direct: model.track() with ByteTrack

        Returns:
            { track_id: {"bbox": [x1,y1,x2,y2], "cls": int} }
        """
        if self._use_sahi and self._sahi_model is not None:
            return self._detect_sahi(frame)
        else:
            return self._detect_direct(frame)

    def _detect_sahi(self, frame) -> dict[int, dict]:
        """
        SAHI sliced inference: detect small/distant people by processing
        overlapping tiles at full resolution, then merge results.
        """
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=frame,
            detection_model=self._sahi_model,
            slice_height=self._sahi_slice_h,
            slice_width=self._sahi_slice_w,
            overlap_height_ratio=self._sahi_overlap,
            overlap_width_ratio=self._sahi_overlap,
            verbose=0,
        )

        # Parse SAHI results into raw detections
        raw_detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            cat_id = pred.category.id
            # Only keep persons (COCO class 0)
            if cat_id == 0:
                raw_detections.append({
                    "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                    "cls": 0,
                })

        # Run through IoU tracker for persistent IDs
        if self._iou_tracker is not None:
            return self._iou_tracker.update(raw_detections)

        # Fallback: return without tracking (no persistent IDs)
        return {i: d for i, d in enumerate(raw_detections)}

    def _detect_direct(self, frame) -> dict[int, dict]:
        """Direct model.track() with ByteTrack — used when SAHI is disabled."""
        with torch.no_grad():
            results = self._own_detector.track(
                source=frame,
                persist=True,
                classes=[0],        # persons only
                conf=self._own_conf,
                iou=self._own_iou,
                imgsz=self._own_imgsz,
                tracker=self._own_tracker_config,
                verbose=False,
            )

        tracks = {}
        if results and results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            cls_list = results[0].boxes.cls.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            for tid, c, box in zip(ids, cls_list, boxes):
                tracks[tid] = {"bbox": box, "cls": c}
        return tracks

    def shutdown(self) -> None:
        """Export heat map image and crowd summary, then release resources."""
        if self._tracker is not None:
            summary = self._tracker.get_summary()
            print(f"\n[CrowdDetectionModule] Session Summary:")
            print(f"  Total entries  : {summary['total_entries']}")
            print(f"  Total exits    : {summary['total_exits']}")
            print(f"  Peak occupancy : {summary['peak_occupancy']}")
            print(f"  Avg occupancy  : {summary['avg_occupancy']}")
            print(f"  Avg dwell time : {summary['avg_dwell_sec']}s")
            print(f"  Unique tracks  : {summary['total_unique_tracks']}")
            print(f"  Density level  : {summary['current_density']}")

        # Release dedicated model if loaded
        if self._own_detector is not None:
            del self._own_detector
            self._own_detector = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[CrowdDetectionModule] Dedicated detector released")

        self._tracker = None
        self._initialized = False
        print("[CrowdDetectionModule] Shut down")

    def export_artifacts(self, session_dir: Path) -> dict[str, str]:
        """
        Export heat map image and crowd CSVs to the session directory.

        Called by the engine after processing completes.

        Returns:
            Dict of artifact_name → file_path for exported files.
        """
        artifacts = {}

        if self._tracker is None:
            return artifacts

        import csv

        # 1. Heat map image
        heatmap_path = session_dir / "crowd_heatmap.png"
        hm_img = self._tracker.get_heatmap_image()
        cv2.imwrite(str(heatmap_path), hm_img)
        artifacts["heatmap"] = str(heatmap_path)
        print(f"[CrowdDetectionModule] Heat map saved: {heatmap_path}")

        # 2. Footfall events CSV
        ff_events = self._tracker.get_footfall_events()
        if ff_events:
            ff_path = session_dir / "crowd_footfall.csv"
            with open(ff_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "track_id", "direction", "timestamp", "frame_idx",
                    "edge", "position_x", "position_y",
                ])
                writer.writeheader()
                for ev in ff_events:
                    writer.writerow({
                        "track_id": ev.track_id,
                        "direction": ev.direction,
                        "timestamp": round(ev.timestamp, 2),
                        "frame_idx": ev.frame_idx,
                        "edge": ev.edge,
                        "position_x": round(ev.position[0], 1),
                        "position_y": round(ev.position[1], 1),
                    })
            artifacts["footfall_csv"] = str(ff_path)
            print(f"[CrowdDetectionModule] Footfall CSV: {ff_path} ({len(ff_events)} events)")

        # 3. Dwell records CSV
        dwells = self._tracker.get_dwell_records()
        if dwells:
            dwell_path = session_dir / "crowd_dwell.csv"
            with open(dwell_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "track_id", "entry_time", "exit_time", "duration_sec",
                    "entry_x", "entry_y", "exit_x", "exit_y",
                ])
                writer.writeheader()
                for d in dwells:
                    writer.writerow({
                        "track_id": d.track_id,
                        "entry_time": round(d.entry_time, 2),
                        "exit_time": round(d.exit_time, 2),
                        "duration_sec": round(d.duration, 2),
                        "entry_x": round(d.entry_position[0], 1),
                        "entry_y": round(d.entry_position[1], 1),
                        "exit_x": round(d.exit_position[0], 1),
                        "exit_y": round(d.exit_position[1], 1),
                    })
            artifacts["dwell_csv"] = str(dwell_path)
            print(f"[CrowdDetectionModule] Dwell CSV: {dwell_path} ({len(dwells)} records)")

        # 4. Summary JSON
        import json
        summary = self._tracker.get_summary()
        summary_path = session_dir / "crowd_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        artifacts["summary"] = str(summary_path)
        print(f"[CrowdDetectionModule] Summary JSON: {summary_path}")

        return artifacts

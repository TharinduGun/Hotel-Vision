"""
Gun Detection Module — AnalyticsModule Implementation
======================================================
Wraps the GunDetector with alert cooldowns, severity mapping,
and the standard module contract.

This is the first module built natively on the orchestrator architecture.
"""

from __future__ import annotations

from collections import defaultdict

from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity
from .detector import GunDetector
from .temporal_filter import TemporalFilter


# ── Severity mapping by weapon class ──────────────────────────────────
WEAPON_SEVERITY = {
    "Handgun": Severity.CRITICAL,
    "Rifle": Severity.CRITICAL,
    "Shotgun": Severity.CRITICAL,
    "Knife": Severity.HIGH,
    # Default for any unknown weapon class
    "default": Severity.HIGH,
}


class GunDetectionModule(AnalyticsModule):
    """
    Detects weapons (guns, knives) in video frames.

    Features:
      - Person-ROI gated detection (saves GPU, reduces false positives)
      - Per-person alert cooldown (prevents alert spam)
      - Severity mapping by weapon class
      - Snapshot + clip triggers for critical events
    """

    def __init__(self):
        self._detector: GunDetector | None = None
        self._config: dict = {}
        self._cameras: list[str] = ["*"]

        # Alert cooldown: { person_id: last_alert_timestamp }
        self._cooldowns: dict[int, float] = defaultdict(lambda: -999.0)
        self._cooldown_sec: float = 10.0
        self._temporal_filter: TemporalFilter | None = None

    @property
    def name(self) -> str:
        return "gun_detection"

    def initialize(self, config: dict) -> None:
        """Load the gun detection model and configure thresholds."""
        self._config = config

        model_path = config.get("model_path", "")
        conf_threshold = config.get("conf_threshold", 0.55)
        person_roi_only = config.get("person_roi_only", True)
        roi_padding = config.get("roi_padding", 0.30)
        self._cooldown_sec = config.get("alert_cooldown_sec", 10.0)
        self._cameras = config.get("cameras", ["*"])

        # Hand proximity filter settings
        hand_proximity_filter = config.get("hand_proximity_filter", True)
        pose_model_path = config.get("pose_model_path", "pycode/src/yolov8m-pose.pt")
        hand_radius_ratio = config.get("hand_radius_ratio", 0.4)

        # Size filter settings
        max_weapon_area_ratio = config.get("max_weapon_area_ratio", 0.40)
        
        # Temporal Filter Settings
        t_min_frames = config.get("temporal_min_frames", 3)
        t_window = config.get("temporal_window", 5)
        self._temporal_filter = TemporalFilter(min_frames=t_min_frames, window_size=t_window)

        # Resolve device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._detector = GunDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            person_roi_only=person_roi_only,
            roi_padding=roi_padding,
            hand_proximity_filter=hand_proximity_filter,
            pose_model_path=pose_model_path,
            hand_radius_ratio=hand_radius_ratio,
            max_weapon_area_ratio=max_weapon_area_ratio,
        )

        print(f"[GunDetectionModule] Initialized "
              f"(cooldown={self._cooldown_sec}s, cameras={self._cameras})")

    def applicable_cameras(self) -> list[str]:
        """Gun detection runs on all cameras by default."""
        return self._cameras

    def process_frame(
        self,
        frame,
        context: FrameContext,
    ) -> list[AnalyticsEvent]:
        """
        Detect weapons in the current frame.

        Returns events only for detections that pass the cooldown filter.
        """
        if self._detector is None:
            return []

        # Run detection (ROI-gated or full-frame)
        raw_detections = self._detector.detect(
            frame=frame,
            person_tracks=context.person_tracks,
        )

        # ── Temporal consistency filter ────────────────────────────
        if self._temporal_filter is not None:
            active_keys = set(context.person_tracks.keys())
            detected_keys = {d.person_id for d in raw_detections if d.person_id is not None}
            if any(d.person_id is None for d in raw_detections):
                detected_keys.add("global")
                active_keys.add("global")
            self._temporal_filter.update(active_keys, detected_keys)

        if not raw_detections:
            return []

        # ── Filter by cooldown & temporal ─────────────────────────
        events = []

        for det in raw_detections:
            key = det.person_id if det.person_id is not None else "global"
            
            # Apply Temporal Filter
            if self._temporal_filter is not None and not self._temporal_filter.is_consistent(key):
                continue
                
            # Apply per-person cooldown
            person_key = det.person_id if det.person_id is not None else -1
            last_alert = self._cooldowns[person_key]

            if (context.timestamp - last_alert) < self._cooldown_sec:
                continue  # Still in cooldown — skip

            # Update cooldown
            self._cooldowns[person_key] = context.timestamp

            # Map severity
            severity = WEAPON_SEVERITY.get(
                det.class_name,
                WEAPON_SEVERITY["default"],
            )

            # Build event
            event = AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type="weapon_detected",
                confidence=det.confidence,
                bbox=det.bbox,
                severity=severity,
                frame_idx=context.frame_idx,
                person_id=det.person_id,
                metadata={
                    "weapon_class": det.class_name,
                    "weapon_class_id": det.class_id,
                    "detection_mode": "person_roi" if det.person_id else "full_frame",
                },
            )

            events.append(event)

        return events

    def shutdown(self) -> None:
        """Release the gun detection model."""
        if self._detector:
            self._detector.shutdown()
            self._detector = None
        print("[GunDetectionModule] Shut down")

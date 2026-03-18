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

        # Resolve device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._detector = GunDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
            person_roi_only=person_roi_only,
            roi_padding=roi_padding,
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

        if not raw_detections:
            return []

        # ── Filter by cooldown ─────────────────────────────────────
        events = []

        for det in raw_detections:
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

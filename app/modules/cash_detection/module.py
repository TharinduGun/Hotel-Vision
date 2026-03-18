"""
Cash Detection Module — AnalyticsModule Implementation
========================================================
Wraps the existing 6-layer cash detection and interaction analysis
pipeline into the new modular architecture.

Layers:
  1. Cash Detection (cash_detector.py → YOLOv8)
  2. Cash-Person Association (cash_detector.associate_with_persons)
  3. Hand Detection & Interaction (hand_detector.py → YOLOv8-pose)
  4. Cash State Tracking (cash_tracker.py → state machine)
  5. Interaction Analysis (interaction_analyzer.py → signal fusion)
  6. Fraud Detection (fraud_detector.py → business rules)

This module delegates to the original utility classes, converting
their outputs into AnalyticsEvents for the orchestrator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity

# Add pycode to path so we can import the existing utilities
_PYCODE_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "pycode")
if _PYCODE_DIR not in sys.path:
    sys.path.insert(0, _PYCODE_DIR)

from utils.cash_detector import CashDetector
from utils.cash_tracker import CashTracker, CashEventType
from utils.hand_detector import HandDetector
from utils.interaction_analyzer import InteractionAnalyzer
from utils.fraud_detector import FraudDetector
from utils.role_classifier import RoleClassifier


# ── Event type → severity mapping ─────────────────────────────────────

CASH_EVENT_SEVERITY = {
    # Cash tracker events
    CashEventType.CASH_PICKUP: Severity.MEDIUM,
    CashEventType.CASH_DEPOSIT: Severity.LOW,
    CashEventType.CASH_HANDOVER: Severity.MEDIUM,
    CashEventType.CASH_POCKET: Severity.HIGH,
    CashEventType.CASH_OUTSIDE_ZONE: Severity.HIGH,
}

FRAUD_SEVERITY = {
    "CASH_POCKET": Severity.HIGH,
    "UNREGISTERED_CASH_HANDLING": Severity.HIGH,
    "CASH_HANDOVER_SUSPICIOUS": Severity.HIGH,
    "CASH_OUTSIDE_ZONE": Severity.MEDIUM,
}


class CashDetectionModule(AnalyticsModule):
    """
    Wraps the existing 6-layer cash handling & fraud detection pipeline
    as an AnalyticsModule.

    This module manages its own internal state (cash tracker, role classifier,
    interaction analyzer, etc.) and converts all outputs to AnalyticsEvent
    objects for the orchestrator.
    """

    def __init__(self):
        self._cash_detector: CashDetector | None = None
        self._cash_tracker: CashTracker | None = None
        self._hand_detector: HandDetector | None = None
        self._interaction_analyzer: InteractionAnalyzer | None = None
        self._fraud_detector: FraudDetector | None = None
        self._role_classifier: RoleClassifier | None = None
        self._config: dict = {}
        self._cameras: list[str] = ["*"]

    @property
    def name(self) -> str:
        return "cash_detection"

    def initialize(self, config: dict) -> None:
        """
        Initialize all 6 layers of the cash detection pipeline.

        Config keys:
            model_path: Path to trained cash detection YOLO model
            conf_threshold: Confidence threshold for cash detection
            cameras: List of camera IDs to process (default: all)
            fps: Frame rate for time-based calculations
        """
        self._config = config
        self._cameras = config.get("cameras", ["*"])

        model_path = config.get("model_path", "")
        conf_threshold = config.get("conf_threshold", 0.25)
        fps = config.get("fps", 25.0)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Layer 1: Cash Detector (YOLO) ─────────────────────────
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cash detection model not found: {model_path}\n"
                "Run pycode/scripts/train_cash_detector.py first."
            )

        self._cash_detector = CashDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
        )
        print(f"[CashModule] Layer 1: Cash Detector loaded ({model_path})")

        # ── Layer 2: Cash Tracker (state machine) ──────────────────
        self._cash_tracker = CashTracker(
            pickup_debounce=8,
            deposit_debounce=15,
            zone_alert_cooldown=30,
        )
        print("[CashModule] Layer 2: Cash Tracker initialized")

        # ── Layer 3: Hand Detector (YOLOv8-pose) ─────────────────
        self._hand_detector = HandDetector(device=device)
        print("[CashModule] Layer 3: Hand Detector loaded")

        # ── Layer 4: Role Classifier ──────────────────────────────
        self._role_classifier = RoleClassifier()
        print("[CashModule] Layer 4: Role Classifier initialized")

        # ── Layer 5: Interaction Analyzer ─────────────────────────
        self._interaction_analyzer = InteractionAnalyzer(fps=fps)
        print("[CashModule] Layer 5: Interaction Analyzer initialized")

        # ── Layer 6: Fraud Detector ──────────────────────────────
        self._fraud_detector = FraudDetector(fps=fps)
        print("[CashModule] Layer 6: Fraud Detector initialized")

        print(f"[CashModule] All 6 layers ready (cameras={self._cameras})")

    def applicable_cameras(self) -> list[str]:
        return self._cameras

    def process_frame(
        self,
        frame,
        context: FrameContext,
    ) -> list[AnalyticsEvent]:
        """
        Run the full 6-layer cash detection pipeline on one frame.

        Returns AnalyticsEvents for every cash event, exchange event,
        and fraud alert detected in this frame.
        """
        if self._cash_detector is None:
            return []

        events: list[AnalyticsEvent] = []
        person_tracks = context.person_tracks

        # ── Layer 1: Cash Detection ────────────────────────────────
        cash_detections = self._cash_detector.detect(
            frame,
            person_tracks=person_tracks,
            roi_manager=context.roi_manager,
        )

        # ── Layer 2: Cash-Person Association ───────────────────────
        cash_associations = self._cash_detector.associate_with_persons(
            cash_detections, person_tracks
        )

        # ── Update role classifier ─────────────────────────────────
        if context.roi_manager and hasattr(context.roi_manager, 'get_zone_with_type'):
            for tid, tdata in person_tracks.items():
                if tdata.get("cls") == 0:
                    cx = (tdata["bbox"][0] + tdata["bbox"][2]) / 2
                    cy = (tdata["bbox"][1] + tdata["bbox"][3]) / 2
                    zone_name, zone_type = context.roi_manager.get_zone_with_type(cx, cy)
                    self._role_classifier.update(tid, zone_name, zone_type)

        # Build roles dict
        current_roles = {
            tid: self._role_classifier.get_role(tid)
            for tid, tdata in person_tracks.items()
            if tdata.get("cls") == 0
        }

        # ── Layer 3: Cash State Tracking ───────────────────────────
        cash_events = []
        if self._cash_tracker is not None:
            cash_events = self._cash_tracker.update(
                frame_idx=context.frame_idx,
                current_time=context.timestamp,
                person_tracks=person_tracks,
                cash_associations=cash_associations,
                roi_manager=context.roi_manager,
            )

        # Convert cash events to AnalyticsEvents
        for ce in cash_events:
            severity = CASH_EVENT_SEVERITY.get(ce.event_type, Severity.MEDIUM)
            person_bbox = person_tracks.get(ce.person_id, {}).get("bbox", [0, 0, 0, 0])

            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type=ce.event_type.value,
                confidence=ce.confidence if hasattr(ce, 'confidence') else 0.7,
                bbox=person_bbox,
                severity=severity,
                frame_idx=context.frame_idx,
                person_id=ce.person_id,
                metadata={
                    "zone": ce.zone if hasattr(ce, 'zone') else "",
                    "role": current_roles.get(ce.person_id, "Unknown"),
                    "partner_id": ce.partner_id if hasattr(ce, 'partner_id') else None,
                },
            ))

        # ── Layer 4: Hand Detection & Interaction ──────────────────
        person_hands = {}
        hand_interactions = []
        if self._hand_detector:
            person_hands, hand_interactions = self._hand_detector.detect_and_analyze(
                frame=frame,
                person_tracks=person_tracks,
                roles=current_roles,
                frame_idx=context.frame_idx,
                roi_manager=context.roi_manager,
            )

        # ── Layer 5: Interaction Analysis ──────────────────────────
        exchange_events = []
        if self._interaction_analyzer:
            exchange_events = self._interaction_analyzer.update(
                frame_idx=context.frame_idx,
                current_time=context.timestamp,
                person_tracks=person_tracks,
                roles=current_roles,
                hand_interactions=hand_interactions,
                cash_detections=cash_detections,
                roi_manager=context.roi_manager,
            )

        # Convert exchange events to AnalyticsEvents
        for ex in exchange_events:
            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type="cash_exchange",
                confidence=ex.confidence if hasattr(ex, 'confidence') else 0.6,
                bbox=[0, 0, 0, 0],  # Exchange is between two people
                severity=Severity.MEDIUM,
                frame_idx=context.frame_idx,
                metadata={
                    "reason": ex.reason if hasattr(ex, 'reason') else "",
                    "persons": [ex.person_a, ex.person_b] if hasattr(ex, 'person_a') else [],
                },
            ))

        # ── Layer 6: Fraud Detection ──────────────────────────────
        fraud_alerts = []
        if self._fraud_detector:
            fraud_alerts = self._fraud_detector.evaluate(
                frame_idx=context.frame_idx,
                current_time=context.timestamp,
                exchange_events=exchange_events,
                cash_events=cash_events,
                person_hands=person_hands,
                roles=current_roles,
                roi_manager=context.roi_manager,
            )

        # Convert fraud alerts to AnalyticsEvents
        for alert in fraud_alerts:
            alert_type = alert.alert_type if hasattr(alert, 'alert_type') else "fraud"
            severity = FRAUD_SEVERITY.get(alert_type, Severity.HIGH)

            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type=f"fraud_{alert_type.lower()}",
                confidence=alert.confidence if hasattr(alert, 'confidence') else 0.8,
                bbox=[0, 0, 0, 0],
                severity=severity,
                frame_idx=context.frame_idx,
                person_id=alert.person_id if hasattr(alert, 'person_id') else None,
                metadata={
                    "description": alert.description if hasattr(alert, 'description') else "",
                    "alert_type": alert_type,
                },
            ))

        return events

    def shutdown(self) -> None:
        """Release all models and resources."""
        import torch

        self._cash_detector = None
        self._cash_tracker = None
        self._hand_detector = None
        self._interaction_analyzer = None
        self._fraud_detector = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[CashModule] Shut down (all 6 layers released)")

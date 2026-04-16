"""
Cash Detection Module — AnalyticsModule Implementation
========================================================
Wraps the 6-layer cash detection and interaction analysis
pipeline into the modular architecture.

Pipeline (executed in order):
  Layer 1: Cash Detection        — YOLOv8 detects cash objects
  Layer 2: Cash-Person Association — spatial overlap mapping
  Layer 3: Role Classification   — zone-based cashier/customer labelling
  Layer 4: Cash State Tracking   — per-person state machine (pickup/deposit/pocket)
  Layer 5: Hand Detection        — YOLOv8-pose wrist keypoints + interaction signals
  Layer 6: Interaction Analysis  — multi-signal fusion for exchange detection
  Layer 7: Fraud Detection       — business rules over timeline

This module delegates to utility classes and converts their outputs
into AnalyticsEvents for the orchestrator.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity

logger = logging.getLogger(__name__)

from .config import CASH_DETECTION_DEFAULTS
from .cash_detector import CashDetector
from .cash_tracker import CashTracker, CashEventType
from .hand_detector import HandDetector
from .interaction_analyzer import InteractionAnalyzer
from .fraud_detector import FraudDetector
from .role_classifier import RoleClassifier
from .temporal_filter import CashTemporalFilter


# ── Event type → severity mapping ─────────────────────────────────────

CASH_EVENT_SEVERITY = {
    CashEventType.CASH_PICKUP: Severity.MEDIUM,
    CashEventType.CASH_DEPOSIT: Severity.LOW,
    CashEventType.CASH_HANDOVER: Severity.MEDIUM,
    CashEventType.CASH_POCKET: Severity.HIGH,
    CashEventType.CASH_OUTSIDE_ZONE: Severity.HIGH,
}

FRAUD_SEVERITY = {
    "CASH_POCKET": Severity.HIGH,
    "CASH_POCKETED": Severity.HIGH,
    "POSSIBLE_POCKETING": Severity.HIGH,
    "UNREGISTERED_CASH_HANDLING": Severity.HIGH,
    "UNREGISTERED_CASH": Severity.HIGH,
    "CASH_HANDOVER_SUSPICIOUS": Severity.HIGH,
    "CASH_OUTSIDE_ZONE": Severity.MEDIUM,
}


def _cfg(config: dict, key: str):
    """Get a config value, falling back to CASH_DETECTION_DEFAULTS."""
    return config.get(key, CASH_DETECTION_DEFAULTS.get(key))


class CashDetectionModule(AnalyticsModule):
    """
    Wraps the 7-layer cash handling & fraud detection pipeline
    as an AnalyticsModule.

    All tunable parameters come from system_config.yaml → cash_detection
    section, with fallbacks defined in config.py.
    """

    def __init__(self):
        self._cash_detector: CashDetector | None = None
        self._cash_tracker: CashTracker | None = None
        self._hand_detector: HandDetector | None = None
        self._interaction_analyzer: InteractionAnalyzer | None = None
        self._fraud_detector: FraudDetector | None = None
        self._role_classifier: RoleClassifier | None = None
        self._temporal_filter: CashTemporalFilter | None = None
        self._config: dict = {}
        self._cameras: list[str] = ["*"]

    @property
    def name(self) -> str:
        return "cash_detection"

    def initialize(self, config: dict) -> None:
        """
        Initialize all layers of the cash detection pipeline.

        Config is merged with CASH_DETECTION_DEFAULTS so that every
        parameter has a sensible fallback.
        """
        self._config = config
        self._cameras = _cfg(config, "cameras")

        model_path = _cfg(config, "model_path")
        conf_threshold = _cfg(config, "conf_threshold")
        fps = _cfg(config, "fps")

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
            hand_region_ratio=_cfg(config, "hand_region_ratio"),
            hand_margin_px=_cfg(config, "hand_margin_px"),
            exchange_gap_px=_cfg(config, "exchange_gap_px"),
            counter_person_radius_px=_cfg(config, "counter_person_radius_px"),
            min_area_px=_cfg(config, "min_area_px"),
            max_area_ratio=_cfg(config, "max_area_ratio"),
            min_aspect_ratio=_cfg(config, "min_aspect_ratio"),
            max_aspect_ratio=_cfg(config, "max_aspect_ratio"),
            hand_reach_px=_cfg(config, "hand_reach_px"),
            exchange_vertical_slack_px=_cfg(config, "exchange_vertical_slack_px"),
            near_hands_offset_px=_cfg(config, "near_hands_offset_px"),
            iou_weight=_cfg(config, "iou_weight"),
            center_inside_weight=_cfg(config, "center_inside_weight"),
            near_hands_weight=_cfg(config, "near_hands_weight"),
            use_sahi=_cfg(config, "use_sahi"),
            sahi_slice_w=_cfg(config, "sahi_slice_width"),
            sahi_slice_h=_cfg(config, "sahi_slice_height"),
            sahi_overlap=_cfg(config, "sahi_overlap_ratio"),
        )
        logger.info(f"Layer 1: Cash Detector loaded ({model_path})")

        # ── Layer 2: Temporal Filter (flicker suppression) ────────
        self._temporal_filter = CashTemporalFilter(
            min_frames=_cfg(config, "temporal_min_frames"),
            window_size=_cfg(config, "temporal_window"),
        )
        logger.info("Layer 2: Temporal Filter initialized")

        # ── Layer 3: Role Classifier (behavioral) ──────────────────
        self._role_classifier = RoleClassifier(
            cashier_threshold=_cfg(config, "cashier_threshold"),
            zone_weight=_cfg(config, "role_zone_weight"),
            stationary_weight=_cfg(config, "role_stationary_weight"),
            visitor_weight=_cfg(config, "role_visitor_weight"),
            stationary_frames=_cfg(config, "role_stationary_frames"),
            movement_threshold_px=_cfg(config, "role_movement_threshold_px"),
            visitor_count_threshold=_cfg(config, "role_visitor_count_threshold"),
        )
        logger.info("Layer 3: Role Classifier initialized (behavioral)")

        # ── Layer 4: Cash Tracker (5-state machine) ────────────────
        self._cash_tracker = CashTracker(
            pickup_debounce=_cfg(config, "pickup_debounce"),
            deposit_debounce=_cfg(config, "deposit_debounce"),
            zone_alert_cooldown=_cfg(config, "zone_alert_cooldown"),
            occlusion_grace_frames=_cfg(config, "occlusion_grace_frames"),
            suspicious_confirm_count=_cfg(config, "suspicious_confirm_count"),
            stale_profile_frames=_cfg(config, "stale_profile_frames"),
            stationary_threshold_px=_cfg(config, "stationary_threshold_px"),
            proximity_threshold_px=_cfg(config, "proximity_threshold_px"),
        )
        logger.info("Layer 4: Cash Tracker initialized (zone-free)")

        # ── Layer 5: Hand Detector (YOLOv8-pose) ─────────────────
        self._hand_detector = HandDetector(
            model_path=_cfg(config, "pose_model_path"),
            device=device,
            keypoint_conf_threshold=_cfg(config, "keypoint_conf_threshold"),
            interaction_threshold_px=_cfg(config, "interaction_threshold_px"),
            iou_threshold=_cfg(config, "iou_threshold"),
        )
        logger.info("Layer 5: Hand Detector loaded")

        # ── Layer 6: Interaction Analyzer ─────────────────────────
        self._interaction_analyzer = InteractionAnalyzer(
            time_window_sec=_cfg(config, "interaction_time_window_sec"),
            fps=fps,
            required_interaction_frames=_cfg(config, "required_interaction_frames"),
            cooldown_frames=_cfg(config, "interaction_cooldown_frames"),
            inferred_exchange_sec=_cfg(config, "inferred_exchange_sec"),
        )
        logger.info("Layer 6: Interaction Analyzer initialized")

        # ── Layer 7: Fraud Detector ──────────────────────────────
        self._fraud_detector = FraudDetector(
            register_wait_sec=_cfg(config, "register_wait_sec"),
            pocketing_window_sec=_cfg(config, "pocketing_window_sec"),
            fps=fps,
        )
        logger.info("Layer 7: Fraud Detector initialized")

        logger.info(f"All layers ready (cameras={self._cameras})")

    def applicable_cameras(self) -> list[str]:
        return self._cameras

    def process_frame(
        self,
        frame,
        context: FrameContext,
    ) -> list[AnalyticsEvent]:
        """
        Run the full cash detection pipeline on one frame.

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

        # ── Layer 2: Cash-Person Association + Temporal Filter ─────
        cash_associations = self._cash_detector.associate_with_persons(
            cash_detections, person_tracks
        )

        # Apply temporal consistency filter to suppress flicker
        if self._temporal_filter is not None:
            active_persons = {
                pid for pid, data in person_tracks.items()
                if data.get("cls") == 0
            }
            persons_with_cash = set(cash_associations.get("assigned", {}).keys())
            self._temporal_filter.update(active_persons, persons_with_cash)
            cash_associations = self._temporal_filter.filter_associations(
                cash_associations
            )

        # ── Layer 3: Role Classification ───────────────────────────
        # Collect nearby IDs for visitor tracking (behavioral Cashier detection)
        for tid, tdata in person_tracks.items():
            if tdata.get("cls") == 0:
                bbox = tdata["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Get zone info (may be None if no zones configured)
                zone_name = "Outside"
                zone_type = None
                if context.roi_manager and context.roi_manager.has_zones:
                    zone_name, zone_type = context.roi_manager.get_zone_with_type(cx, cy)

                # Find nearby person IDs (for visitor signal)
                nearby_ids = set()
                for other_tid, other_tdata in person_tracks.items():
                    if other_tid != tid and other_tdata.get("cls") == 0:
                        other_bbox = other_tdata["bbox"]
                        other_cx = (other_bbox[0] + other_bbox[2]) / 2
                        other_cy = (other_bbox[1] + other_bbox[3]) / 2
                        dist = ((cx - other_cx) ** 2 + (cy - other_cy) ** 2) ** 0.5
                        if dist < 250:  # proximity threshold
                            nearby_ids.add(other_tid)

                self._role_classifier.update(
                    logical_id=tid,
                    zone_name=zone_name,
                    zone_type=zone_type,
                    bbox=bbox,
                    frame_idx=context.frame_idx,
                    nearby_ids=nearby_ids,
                )

        current_roles = {
            tid: self._role_classifier.get_role(tid)
            for tid, tdata in person_tracks.items()
            if tdata.get("cls") == 0
        }
        # Push roles to shared context for orchestrator drawing
        context.roles.update(current_roles)

        # ── Layer 4: Cash State Tracking ───────────────────────────
        cash_events = []
        if self._cash_tracker is not None:
            cash_events = self._cash_tracker.update(
                frame_idx=context.frame_idx,
                current_time=context.timestamp,
                person_tracks=person_tracks,
                cash_associations=cash_associations,
                roi_manager=context.roi_manager,
                roles=current_roles,
            )

        for ce in cash_events:
            # Skip NEUTRAL events — they are informational only, no alert
            if ce.event_type == CashEventType.NEUTRAL:
                continue

            severity = CASH_EVENT_SEVERITY.get(ce.event_type, Severity.MEDIUM)
            person_bbox = person_tracks.get(ce.person_id, {}).get("bbox", [0, 0, 0, 0])

            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type=ce.event_type.value,
                confidence=ce.confidence,
                bbox=person_bbox,
                severity=severity,
                frame_idx=context.frame_idx,
                person_id=ce.person_id,
                metadata={
                    "zone": ce.zone,
                    "role": current_roles.get(ce.person_id, "Unknown"),
                    "partner_id": ce.partner_id,
                },
            ))

        # ── Layer 5: Hand Detection & Interaction ──────────────────
        person_hands = {}
        hand_interactions = []
        
        # D5 Performance Optimization: Only run YOLO pose if strictly needed
        has_customer = "Customer" in current_roles.values()
        has_cashier = "Cashier" in current_roles.values()
        has_cash_activity = len(cash_detections) > 0 or len(cash_events) > 0
        has_pending_fraud = len(self._fraud_detector.pending_exchanges) > 0 if self._fraud_detector else False
        
        run_hand_pose = has_cash_activity or has_pending_fraud or (has_customer and has_cashier)

        if self._hand_detector and run_hand_pose:
            person_hands, hand_interactions = self._hand_detector.detect_and_analyze(
                frame=frame,
                person_tracks=person_tracks,
                roles=current_roles,
                frame_idx=context.frame_idx,
                roi_manager=context.roi_manager,
            )

        # ── Layer 6: Interaction Analysis ──────────────────────────
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

        for ex in exchange_events:
            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type="cash_exchange",
                confidence=ex.confidence,
                bbox=[0, 0, 0, 0],
                severity=Severity.MEDIUM,
                frame_idx=context.frame_idx,
                metadata={
                    "reason": ex.reason,
                    "customer_id": ex.customer_id,
                    "cashier_id": ex.cashier_id,
                },
            ))

        # ── Layer 7: Fraud Detection ──────────────────────────────
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

        for alert in fraud_alerts:
            severity = FRAUD_SEVERITY.get(alert.alert_type, Severity.HIGH)

            events.append(AnalyticsEvent(
                module=self.name,
                camera_id=context.camera_id,
                timestamp=context.timestamp,
                event_type=f"fraud_{alert.alert_type.lower()}",
                confidence=alert.confidence,
                bbox=[0, 0, 0, 0],
                severity=severity,
                frame_idx=context.frame_idx,
                person_id=alert.person_id,
                metadata={
                    "description": alert.description,
                    "alert_type": alert.alert_type,
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
        self._temporal_filter = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Shut down (all layers released)")

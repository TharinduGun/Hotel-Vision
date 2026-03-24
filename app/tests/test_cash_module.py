"""
Tests for the Cash Detection Module wrapper.
Uses mocked sub-components — no YOLO models required.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity
from app.modules.cash_detection.module import CashDetectionModule, CASH_EVENT_SEVERITY, FRAUD_SEVERITY


# ── Interface tests ───────────────────────────────────────────────────

def test_cash_module_is_analytics_module():
    """CashDetectionModule must implement AnalyticsModule."""
    assert issubclass(CashDetectionModule, AnalyticsModule)


def test_cash_module_name():
    """Module name must be 'cash_detection'."""
    m = CashDetectionModule()
    assert m.name == "cash_detection"


def test_cash_module_applicable_cameras_default():
    """Default should apply to all cameras."""
    m = CashDetectionModule()
    assert m.applicable_cameras() == ["*"]


def test_cash_module_process_frame_without_init():
    """Module should return empty list if not initialized."""
    m = CashDetectionModule()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx = FrameContext(
        camera_id="CAM-01",
        frame_idx=0,
        timestamp=0.0,
        fps=25.0,
        frame_width=640,
        frame_height=480,
    )
    events = m.process_frame(frame, ctx)
    assert events == []


def test_cash_module_shutdown_without_init():
    """Shutdown should not crash if module was never initialized."""
    m = CashDetectionModule()
    m.shutdown()  # Should not raise


# ── Severity mapping tests ───────────────────────────────────────────

def test_cash_event_severity_mapping():
    """Cash event types should map to appropriate severity levels."""
    from app.modules.cash_detection.cash_tracker import CashEventType
    
    assert CASH_EVENT_SEVERITY[CashEventType.CASH_PICKUP] == Severity.MEDIUM
    assert CASH_EVENT_SEVERITY[CashEventType.CASH_DEPOSIT] == Severity.LOW
    assert CASH_EVENT_SEVERITY[CashEventType.CASH_HANDOVER] == Severity.MEDIUM
    assert CASH_EVENT_SEVERITY[CashEventType.CASH_POCKET] == Severity.HIGH
    assert CASH_EVENT_SEVERITY[CashEventType.CASH_OUTSIDE_ZONE] == Severity.HIGH


def test_fraud_severity_mapping():
    """Fraud alert types should map to HIGH or above."""
    assert FRAUD_SEVERITY["CASH_POCKET"] == Severity.HIGH
    assert FRAUD_SEVERITY["UNREGISTERED_CASH_HANDLING"] == Severity.HIGH
    assert FRAUD_SEVERITY["CASH_HANDOVER_SUSPICIOUS"] == Severity.HIGH
    assert FRAUD_SEVERITY["CASH_OUTSIDE_ZONE"] == Severity.MEDIUM


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cash_module_is_analytics_module()
    test_cash_module_name()
    test_cash_module_applicable_cameras_default()
    test_cash_module_process_frame_without_init()
    test_cash_module_shutdown_without_init()
    test_cash_event_severity_mapping()
    test_fraud_severity_mapping()
    print("✅ All cash detection module tests passed!")

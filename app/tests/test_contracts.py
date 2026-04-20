"""
Tests for the Analytics Module contracts and event schema.
These tests validate the module interface without requiring any ML models.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity


# ── Dummy module for testing ──────────────────────────────────────────

class DummyModule(AnalyticsModule):
    """A test module that always returns one event per frame."""

    @property
    def name(self) -> str:
        return "dummy_test"

    def initialize(self, config: dict) -> None:
        self.threshold = config.get("threshold", 0.5)

    def applicable_cameras(self) -> list[str]:
        return ["*"]

    def process_frame(self, frame, context: FrameContext) -> list[AnalyticsEvent]:
        return [AnalyticsEvent(
            module=self.name,
            camera_id=context.camera_id,
            timestamp=context.timestamp,
            event_type="test_detection",
            confidence=0.99,
            bbox=[10, 20, 100, 200],
            severity=Severity.LOW,
            frame_idx=context.frame_idx,
        )]

    def shutdown(self) -> None:
        pass


# ── AnalyticsModule contract tests ────────────────────────────────────

def test_module_has_name():
    """Module must expose a name property."""
    m = DummyModule()
    assert m.name == "dummy_test"


def test_module_initialize():
    """Module must accept config in initialize()."""
    m = DummyModule()
    m.initialize({"threshold": 0.7})
    assert m.threshold == 0.7


def test_module_applicable_cameras():
    """Module must return list of camera IDs."""
    m = DummyModule()
    m.initialize({})
    cameras = m.applicable_cameras()
    assert isinstance(cameras, list)
    assert len(cameras) > 0


def test_module_process_frame():
    """Module must return list of AnalyticsEvent from process_frame()."""
    m = DummyModule()
    m.initialize({})

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    context = FrameContext(
        camera_id="TEST-CAM",
        frame_idx=42,
        timestamp=1.68,
        fps=25.0,
        frame_width=640,
        frame_height=480,
    )

    events = m.process_frame(frame, context)
    assert isinstance(events, list)
    assert len(events) == 1
    assert isinstance(events[0], AnalyticsEvent)


def test_module_shutdown():
    """Module must support shutdown without error."""
    m = DummyModule()
    m.initialize({})
    m.shutdown()  # Should not raise


# ── AnalyticsEvent tests ──────────────────────────────────────────────

def test_event_creation():
    """AnalyticsEvent must accept all required fields."""
    event = AnalyticsEvent(
        module="test",
        camera_id="CAM-01",
        timestamp=5.5,
        event_type="weapon_detected",
        confidence=0.91,
        bbox=[120, 200, 180, 340],
        severity=Severity.CRITICAL,
    )
    assert event.module == "test"
    assert event.severity == Severity.CRITICAL
    assert event.confidence == 0.91


def test_event_auto_iso_timestamp():
    """AnalyticsEvent must auto-populate ISO timestamp."""
    event = AnalyticsEvent(
        module="test",
        camera_id="CAM-01",
        timestamp=1.0,
        event_type="test",
        confidence=0.5,
        bbox=[0, 0, 100, 100],
        severity=Severity.LOW,
    )
    assert event.iso_timestamp is not None
    assert "T" in event.iso_timestamp  # ISO format contains T


def test_event_to_dict():
    """AnalyticsEvent.to_dict() must return a flat dictionary."""
    event = AnalyticsEvent(
        module="gun_detection",
        camera_id="CAM-02",
        timestamp=12.5,
        event_type="weapon_detected",
        confidence=0.88,
        bbox=[100, 200, 150, 300],
        severity=Severity.CRITICAL,
        frame_idx=312,
        person_id=5,
        metadata={"weapon_class": "Handgun"},
    )
    d = event.to_dict()
    assert d["module"] == "gun_detection"
    assert d["severity"] == "critical"
    assert d["meta_weapon_class"] == "Handgun"
    assert isinstance(d["bbox"], str)  # "100,200,150,300"


def test_event_repr():
    """AnalyticsEvent.__repr__ should be readable."""
    event = AnalyticsEvent(
        module="gun_detection",
        camera_id="CAM-01",
        timestamp=1.0,
        event_type="weapon_detected",
        confidence=0.95,
        bbox=[0, 0, 50, 50],
        severity=Severity.CRITICAL,
    )
    r = repr(event)
    assert "gun_detection" in r
    assert "weapon_detected" in r


def test_severity_values():
    """All severity levels must be string-comparable."""
    assert Severity.LOW.value == "low"
    assert Severity.MEDIUM.value == "medium"
    assert Severity.HIGH.value == "high"
    assert Severity.CRITICAL.value == "critical"


# ── FrameContext tests ────────────────────────────────────────────────

def test_frame_context_defaults():
    """FrameContext must have sensible defaults for optional fields."""
    ctx = FrameContext(
        camera_id="CAM-01",
        frame_idx=0,
        timestamp=0.0,
        fps=25.0,
        frame_width=1920,
        frame_height=1080,
    )
    assert ctx.person_tracks == {}
    assert ctx.roi_manager is None
    assert ctx.roles == {}


def test_frame_context_with_tracks():
    """FrameContext must accept person tracks."""
    tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 0},
        2: {"bbox": [400, 200, 600, 500], "cls": 0},
    }
    ctx = FrameContext(
        camera_id="CAM-01",
        frame_idx=10,
        timestamp=0.4,
        fps=25.0,
        frame_width=1920,
        frame_height=1080,
        person_tracks=tracks,
    )
    assert len(ctx.person_tracks) == 2
    assert 1 in ctx.person_tracks


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_module_has_name()
    test_module_initialize()
    test_module_applicable_cameras()
    test_module_process_frame()
    test_module_shutdown()
    test_event_creation()
    test_event_auto_iso_timestamp()
    test_event_to_dict()
    test_event_repr()
    test_severity_values()
    test_frame_context_defaults()
    test_frame_context_with_tracks()
    print("✅ All contract + event schema tests passed!")

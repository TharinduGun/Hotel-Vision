"""
Tests for the Gun Detection Module.
Uses mock detections — no YOLO model required.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity
from app.modules.gun_detection.module import GunDetectionModule, WEAPON_SEVERITY
from app.modules.gun_detection.detector import GunDetection


# ── Mock GunDetector (bypasses model loading) ─────────────────────────

class MockGunDetector:
    """Fake detector that returns configurable detections."""

    def __init__(self, detections=None):
        self._detections = detections or []

    def detect(self, frame, person_tracks=None):
        return self._detections

    def shutdown(self):
        pass


def _make_context(cam_id="CAM-01", frame_idx=0, timestamp=0.0):
    return FrameContext(
        camera_id=cam_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        fps=25.0,
        frame_width=1920,
        frame_height=1080,
        person_tracks={1: {"bbox": [100, 100, 300, 500], "cls": 0}},
    )


def _make_gun_detection(conf=0.85, cls_name="Handgun", person_id=1):
    return GunDetection(
        bbox=[150, 250, 200, 300],
        confidence=conf,
        class_name=cls_name,
        class_id=0,
        person_id=person_id,
    )


# ── Interface tests ───────────────────────────────────────────────────

def test_gun_module_is_analytics_module():
    """GunDetectionModule must be a subclass of AnalyticsModule."""
    assert issubclass(GunDetectionModule, AnalyticsModule)


def test_gun_module_name():
    """Module name must be 'gun_detection'."""
    m = GunDetectionModule()
    assert m.name == "gun_detection"


def test_gun_module_applicable_cameras_default():
    """By default, gun detection applies to all cameras."""
    m = GunDetectionModule()
    # Before initialize, should still have default
    assert m.applicable_cameras() == ["*"]


# ── Detection processing tests ────────────────────────────────────────

def test_process_frame_with_detection():
    """Module should return an event when a weapon is detected."""
    m = GunDetectionModule()
    # Inject mock detector
    m._detector = MockGunDetector([_make_gun_detection()])
    m._cooldown_sec = 0  # Disable cooldown

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _make_context(timestamp=1.0)
    events = m.process_frame(frame, ctx)

    assert len(events) == 1
    assert events[0].event_type == "weapon_detected"
    assert events[0].module == "gun_detection"
    assert events[0].severity == Severity.CRITICAL


def test_process_frame_no_detection():
    """Module should return empty list when nothing is detected."""
    m = GunDetectionModule()
    m._detector = MockGunDetector([])  # No detections

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _make_context()
    events = m.process_frame(frame, ctx)

    assert events == []


def test_process_frame_none_detector():
    """If detector is not initialized, return empty list."""
    m = GunDetectionModule()
    m._detector = None

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _make_context()
    events = m.process_frame(frame, ctx)

    assert events == []


# ── Cooldown tests ────────────────────────────────────────────────────

def test_cooldown_suppresses_repeated_alerts():
    """Same person should not trigger alerts faster than cooldown_sec."""
    m = GunDetectionModule()
    m._cooldown_sec = 10.0
    m._detector = MockGunDetector([_make_gun_detection(person_id=1)])

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # First detection at t=0 — should fire
    ctx1 = _make_context(timestamp=0.0)
    events1 = m.process_frame(frame, ctx1)
    assert len(events1) == 1

    # Second detection at t=5 — within cooldown, should suppress
    ctx2 = _make_context(timestamp=5.0, frame_idx=125)
    events2 = m.process_frame(frame, ctx2)
    assert len(events2) == 0

    # Third detection at t=15 — after cooldown, should fire
    ctx3 = _make_context(timestamp=15.0, frame_idx=375)
    events3 = m.process_frame(frame, ctx3)
    assert len(events3) == 1


def test_cooldown_per_person():
    """Different persons should have independent cooldowns."""
    m = GunDetectionModule()
    m._cooldown_sec = 10.0

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Person 1 detected at t=0
    m._detector = MockGunDetector([_make_gun_detection(person_id=1)])
    ctx1 = _make_context(timestamp=0.0)
    events1 = m.process_frame(frame, ctx1)
    assert len(events1) == 1

    # Person 2 detected at t=2 — different person, should fire
    m._detector = MockGunDetector([_make_gun_detection(person_id=2)])
    ctx2 = _make_context(timestamp=2.0, frame_idx=50)
    events2 = m.process_frame(frame, ctx2)
    assert len(events2) == 1


# ── Severity mapping tests ───────────────────────────────────────────

def test_handgun_severity():
    """Handgun should map to CRITICAL severity."""
    assert WEAPON_SEVERITY["Handgun"] == Severity.CRITICAL


def test_rifle_severity():
    """Rifle should map to CRITICAL severity."""
    assert WEAPON_SEVERITY["Rifle"] == Severity.CRITICAL


def test_knife_severity():
    """Knife should map to HIGH severity."""
    assert WEAPON_SEVERITY["Knife"] == Severity.HIGH


def test_default_severity():
    """Unknown weapon classes should get HIGH severity."""
    assert WEAPON_SEVERITY["default"] == Severity.HIGH


# ── Event metadata tests ─────────────────────────────────────────────

def test_event_has_weapon_class_metadata():
    """Events should include weapon_class in metadata."""
    m = GunDetectionModule()
    m._detector = MockGunDetector([_make_gun_detection(cls_name="Rifle")])
    m._cooldown_sec = 0

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _make_context(timestamp=1.0)
    events = m.process_frame(frame, ctx)

    assert events[0].metadata["weapon_class"] == "Rifle"
    assert events[0].metadata["detection_mode"] == "person_roi"


def test_event_has_person_id():
    """Events should include the associated person_id."""
    m = GunDetectionModule()
    m._detector = MockGunDetector([_make_gun_detection(person_id=7)])
    m._cooldown_sec = 0

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _make_context(timestamp=1.0)
    events = m.process_frame(frame, ctx)

    assert events[0].person_id == 7


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_gun_module_is_analytics_module()
    test_gun_module_name()
    test_gun_module_applicable_cameras_default()
    test_process_frame_with_detection()
    test_process_frame_no_detection()
    test_process_frame_none_detector()
    test_cooldown_suppresses_repeated_alerts()
    test_cooldown_per_person()
    test_handgun_severity()
    test_rifle_severity()
    test_knife_severity()
    test_default_severity()
    test_event_has_weapon_class_metadata()
    test_event_has_person_id()
    print("✅ All gun detection module tests passed!")

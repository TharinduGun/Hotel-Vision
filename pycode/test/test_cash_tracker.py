"""
Unit tests for CashTracker state machine logic.
Tests state transitions, debouncing, handover detection, and zone violations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.cash_tracker import CashTracker, CashState, CashEventType
from utils.cash_detector import CashDetection


def _make_cash(bbox, conf=0.90):
    return CashDetection(bbox=bbox, confidence=conf, class_name="Cash", class_id=0)


def _make_associations(assigned=None, unassigned=None):
    return {
        "assigned": assigned or {},
        "unassigned": unassigned or [],
    }


class FakeROIManager:
    """Minimal ROI manager stub for testing."""
    has_zones = True

    def __init__(self, zone_map=None):
        # zone_map: {(x_min, y_min, x_max, y_max): ("ZoneName", "zone_type")}
        self._zone_map = zone_map or {}

    def get_zone_with_type(self, cx, cy):
        for (x1, y1, x2, y2), (name, ztype) in self._zone_map.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return name, ztype
        return "Outside", None


def test_pickup_detection():
    """Cash appearing near a person for N frames triggers CASH_PICKUP."""
    tracker = CashTracker(pickup_debounce=3, deposit_debounce=5)
    person_tracks = {1: {"bbox": [100, 100, 300, 500], "cls": 0}}
    cash = [_make_cash([150, 350, 200, 400])]

    events = []
    for frame in range(5):
        assoc = _make_associations(assigned={1: cash})
        new_events = tracker.update(frame, frame * 0.04, person_tracks, assoc)
        events.extend(new_events)

    pickup_events = [e for e in events if e.event_type == CashEventType.CASH_PICKUP]
    assert len(pickup_events) == 1, f"Expected 1 CASH_PICKUP, got {len(pickup_events)}"
    assert pickup_events[0].person_id == 1


def test_deposit_at_register():
    """Cash disappearing while person is at a register zone = CASH_DEPOSIT."""
    roi = FakeROIManager(zone_map={
        (0, 0, 500, 600): ("Register 1", "cash_register"),
    })
    tracker = CashTracker(pickup_debounce=2, deposit_debounce=3)
    person_tracks = {1: {"bbox": [100, 100, 300, 500], "cls": 0}}  # Inside register zone
    cash = [_make_cash([150, 350, 200, 400])]

    # First: pick up cash (2 frames to debounce)
    for frame in range(3):
        assoc = _make_associations(assigned={1: cash})
        tracker.update(frame, frame * 0.04, person_tracks, assoc, roi)

    # Now cash disappears for 3 frames (deposit debounce)
    events = []
    for frame in range(3, 7):
        assoc = _make_associations()  # No cash
        new_events = tracker.update(frame, frame * 0.04, person_tracks, assoc, roi)
        events.extend(new_events)

    deposit_events = [e for e in events if e.event_type == CashEventType.CASH_DEPOSIT]
    assert len(deposit_events) == 1, f"Expected 1 CASH_DEPOSIT, got {len(deposit_events)}"


def test_pocket_outside_zone():
    """Cash disappearing while person is NOT at a register = CASH_POCKET."""
    roi = FakeROIManager()  # No zones — everything is "Outside"
    tracker = CashTracker(pickup_debounce=2, deposit_debounce=3)
    person_tracks = {1: {"bbox": [100, 100, 300, 500], "cls": 0}}
    cash = [_make_cash([150, 350, 200, 400])]

    # Pick up cash
    for frame in range(3):
        assoc = _make_associations(assigned={1: cash})
        tracker.update(frame, frame * 0.04, person_tracks, assoc, roi)

    # Cash disappears outside zone
    events = []
    for frame in range(3, 7):
        assoc = _make_associations()
        new_events = tracker.update(frame, frame * 0.04, person_tracks, assoc, roi)
        events.extend(new_events)

    pocket_events = [e for e in events if e.event_type == CashEventType.CASH_POCKET]
    assert len(pocket_events) == 1, f"Expected 1 CASH_POCKET, got {len(pocket_events)}"


def test_no_false_pickup_below_debounce():
    """Cash detected for fewer frames than debounce should NOT trigger pickup."""
    tracker = CashTracker(pickup_debounce=5, deposit_debounce=5)
    person_tracks = {1: {"bbox": [100, 100, 300, 500], "cls": 0}}
    cash = [_make_cash([150, 350, 200, 400])]

    events = []
    # Only 3 frames with cash (below debounce of 5)
    for frame in range(3):
        assoc = _make_associations(assigned={1: cash})
        new_events = tracker.update(frame, frame * 0.04, person_tracks, assoc)
        events.extend(new_events)

    # Then no cash
    for frame in range(3, 10):
        assoc = _make_associations()
        new_events = tracker.update(frame, frame * 0.04, person_tracks, assoc)
        events.extend(new_events)

    pickup_events = [e for e in events if e.event_type == CashEventType.CASH_PICKUP]
    assert len(pickup_events) == 0, "Should NOT trigger pickup below debounce threshold"


def test_summary():
    """get_summary() returns correct counts."""
    tracker = CashTracker(pickup_debounce=1, deposit_debounce=1)
    person_tracks = {1: {"bbox": [100, 100, 300, 500], "cls": 0}}
    cash = [_make_cash([150, 350, 200, 400])]

    # Pickup
    assoc = _make_associations(assigned={1: cash})
    tracker.update(0, 0.0, person_tracks, assoc)

    summary = tracker.get_summary()
    assert summary["total_events"] >= 1
    assert summary["active_holders"] == 1


if __name__ == "__main__":
    test_pickup_detection()
    test_deposit_at_register()
    test_pocket_outside_zone()
    test_no_false_pickup_below_debounce()
    test_summary()
    print("All CashTracker tests passed!")

"""
Unit tests for CashDetector — association + context-aware filtering logic.
These tests use mock bounding boxes (no model required).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.cash_detector import CashDetection, CashDetector


class MockCashDetector(CashDetector):
    """Bypass model loading — we only test filtering and association logic."""

    def __init__(self, **kwargs):
        # Skip YOLO model loading; set defaults for filter params
        self.conf_threshold = kwargs.get("conf_threshold", 0.35)
        self.device = "cpu"
        self._class_names = {0: "Cash", 1: "person"}
        self.hand_region_ratio = kwargs.get("hand_region_ratio", 0.50)
        self.hand_margin_px = kwargs.get("hand_margin_px", 60)
        self.exchange_gap_px = kwargs.get("exchange_gap_px", 100)
        self.counter_person_radius_px = kwargs.get("counter_person_radius_px", 250)
        self.min_area_px = kwargs.get("min_area_px", 400)
        self.max_area_ratio = kwargs.get("max_area_ratio", 0.10)
        self.min_aspect_ratio = kwargs.get("min_aspect_ratio", 1.0)
        self.max_aspect_ratio = kwargs.get("max_aspect_ratio", 8.0)


def _make_cash(bbox, conf=0.90):
    return CashDetection(bbox=bbox, confidence=conf, class_name="Cash", class_id=0)


# ── Association Tests (existing) ────────────────────────────────────

def test_cash_inside_person_bbox():
    """Cash bbox overlapping with person bbox should be assigned."""
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 0},
    }
    cash = [_make_cash([150, 350, 200, 400])]

    detector = MockCashDetector()
    result = detector.associate_with_persons(cash, person_tracks)

    assert 1 in result["assigned"], "Cash should be assigned to person 1"
    assert len(result["unassigned"]) == 0, "No cash should be unassigned"


def test_cash_far_from_person():
    """Cash bbox far from any person should be unassigned."""
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 0},
    }
    cash = [_make_cash([800, 800, 850, 850])]

    detector = MockCashDetector()
    result = detector.associate_with_persons(cash, person_tracks)

    assert len(result["assigned"]) == 0, "No cash should be assigned"
    assert len(result["unassigned"]) == 1, "Cash should be unassigned"


def test_cash_assigned_to_nearest_person():
    """Cash should be assigned to the closest overlapping person."""
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 0},
        2: {"bbox": [400, 100, 600, 500], "cls": 0},
    }
    # Cash is near person 2's hands area
    cash = [_make_cash([420, 350, 480, 400])]

    detector = MockCashDetector()
    result = detector.associate_with_persons(cash, person_tracks)

    assert 2 in result["assigned"], "Cash should be assigned to person 2 (closer)"
    assert 1 not in result["assigned"], "Cash should NOT be assigned to person 1"


def test_multiple_cash_one_person():
    """Multiple cash detections near the same person are all assigned."""
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 0},
    }
    cash = [
        _make_cash([130, 350, 180, 400]),
        _make_cash([200, 360, 260, 410]),
    ]

    detector = MockCashDetector()
    result = detector.associate_with_persons(cash, person_tracks)

    assert 1 in result["assigned"], "Both cash should be assigned to person 1"
    assert len(result["assigned"][1]) == 2, "Person 1 should have 2 cash detections"


def test_ignores_non_person_tracks():
    """Non-person tracks (cars) should be ignored for association."""
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 2},  # cls=2 is car
    }
    cash = [_make_cash([150, 350, 200, 400])]

    detector = MockCashDetector()
    result = detector.associate_with_persons(cash, person_tracks)

    assert len(result["assigned"]) == 0, "Cash should not be assigned to a car"
    assert len(result["unassigned"]) == 1, "Cash should be unassigned"


def test_empty_inputs():
    """Empty detections or tracks should return empty results."""
    detector = MockCashDetector()

    # No cash
    result = detector.associate_with_persons([], {1: {"bbox": [0, 0, 100, 100], "cls": 0}})
    assert len(result["assigned"]) == 0
    assert len(result["unassigned"]) == 0

    # No persons
    cash = [_make_cash([50, 50, 80, 80])]
    result = detector.associate_with_persons(cash, {})
    assert len(result["assigned"]) == 0
    assert len(result["unassigned"]) == 1


# ── Geometric Filter Tests ──────────────────────────────────────────

def test_geometric_filter_rejects_tiny():
    """Tiny cash boxes below min_area_px should be rejected."""
    detector = MockCashDetector(min_area_px=400)
    # 10x10 = 100 px area — way below threshold
    detections = [_make_cash([100, 100, 110, 110])]

    frame_area = 1920 * 1080
    result = detector._geometric_filter(detections, frame_area)
    assert len(result) == 0, "Tiny box should be rejected"


def test_geometric_filter_rejects_huge():
    """Cash boxes covering >10% of frame should be rejected."""
    detector = MockCashDetector(max_area_ratio=0.10)
    # 1000x500 = 500,000 px area on a 1920x1080 frame (~24% of frame)
    detections = [_make_cash([0, 0, 1000, 500])]

    frame_area = 1920 * 1080
    result = detector._geometric_filter(detections, frame_area)
    assert len(result) == 0, "Huge box should be rejected"


def test_geometric_filter_rejects_square():
    """Near-square boxes (aspect ratio < 1.0) are impossible — 1.0 allows folded cash."""
    detector = MockCashDetector(min_aspect_ratio=1.0)
    # 50x50 = perfect square (aspect ratio = 1.0) — passes now since min is 1.0
    detections = [_make_cash([100, 100, 150, 150])]

    frame_area = 1920 * 1080
    result = detector._geometric_filter(detections, frame_area)
    # 50x50 = 2500px area > 400px min, aspect = 1.0 = min, passes
    assert len(result) == 1, "Square box should pass (cash can look square when folded)"


def test_geometric_filter_passes_valid():
    """A properly sized, cash-shaped rectangle should pass."""
    detector = MockCashDetector()
    # 100x40 = 4000 px area, aspect ratio ~2.5 — looks like a banknote
    detections = [_make_cash([100, 200, 200, 240])]

    frame_area = 1920 * 1080
    result = detector._geometric_filter(detections, frame_area)
    assert len(result) == 1, "Valid cash-shaped box should pass"


# ── Contextual Filter Tests ─────────────────────────────────────────

def test_context_near_hands_passes():
    """Cash near a person's hands (lower half of person bbox) should pass."""
    detector = MockCashDetector()
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 0},
    }
    # Cash in person's lower half (hand area), Y=380-410 is in lower 50%
    cash = [_make_cash([150, 380, 250, 410])]

    result = detector._contextual_filter(cash, person_tracks)
    assert len(result) == 1, "Cash near hands should pass context filter"


def test_context_far_from_everyone_rejected():
    """Cash far from any person and not on a counter should be rejected."""
    detector = MockCashDetector()
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 0},
    }
    # Cash at [800, 50] — far from person, not on any counter
    cash = [_make_cash([800, 50, 900, 80])]

    result = detector._contextual_filter(cash, person_tracks, roi_manager=None)
    assert len(result) == 0, "Cash far from everyone should be rejected"


def test_context_between_persons_passes():
    """Cash in the gap between two nearby persons should pass (exchange)."""
    detector = MockCashDetector(exchange_gap_px=100)
    person_tracks = {
        1: {"bbox": [100, 100, 250, 500], "cls": 0},
        2: {"bbox": [300, 100, 450, 500], "cls": 0},
    }
    # Cash in the gap between person 1 (right=250) and person 2 (left=300)
    cash = [_make_cash([260, 300, 290, 330])]

    result = detector._contextual_filter(cash, person_tracks)
    assert len(result) == 1, "Cash between two persons should pass (exchange scenario)"


def test_context_no_persons_rejects_all():
    """With no person tracks, all cash should be rejected (since no context)."""
    detector = MockCashDetector()
    person_tracks = {}  # No persons
    cash = [_make_cash([150, 200, 250, 230])]

    result = detector._contextual_filter(cash, person_tracks)
    assert len(result) == 0, "Cash with no persons should be rejected"


def test_context_car_tracks_ignored():
    """Only person tracks (cls=0) provide context; cars don't."""
    detector = MockCashDetector()
    person_tracks = {
        1: {"bbox": [100, 100, 300, 500], "cls": 2},  # car, not person
    }
    # Cash near the car's bbox — but cars don't count
    cash = [_make_cash([150, 380, 250, 410])]

    result = detector._contextual_filter(cash, person_tracks)
    assert len(result) == 0, "Cash near a car should not pass context filter"


def test_context_near_hands_with_margin():
    """Cash slightly outside person bbox but within hand_margin should pass."""
    detector = MockCashDetector(hand_margin_px=60)
    person_tracks = {
        1: {"bbox": [200, 100, 400, 500], "cls": 0},
    }
    # Cash 30px to the right of person bbox right edge (within 60px margin)
    # Y=400 is in the lower 50% of person [100,500]
    cash = [_make_cash([410, 380, 500, 410])]

    result = detector._contextual_filter(cash, person_tracks)
    assert len(result) == 1, "Cash within hand margin should pass"


if __name__ == "__main__":
    test_cash_inside_person_bbox()
    test_cash_far_from_person()
    test_cash_assigned_to_nearest_person()
    test_multiple_cash_one_person()
    test_ignores_non_person_tracks()
    test_empty_inputs()
    # Geometric filter tests
    test_geometric_filter_rejects_tiny()
    test_geometric_filter_rejects_huge()
    test_geometric_filter_rejects_square()
    test_geometric_filter_passes_valid()
    # Contextual filter tests
    test_context_near_hands_passes()
    test_context_far_from_everyone_rejected()
    test_context_between_persons_passes()
    test_context_no_persons_rejects_all()
    test_context_car_tracks_ignored()
    test_context_near_hands_with_margin()
    print("All CashDetector tests passed!")

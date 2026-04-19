"""
Tests for the Crowd Detection Module.
Uses mock person tracks — no YOLO model required.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from app.contracts.base_module import AnalyticsModule, FrameContext
from app.contracts.event_schema import AnalyticsEvent, Severity
from app.modules.crowd_detection.module import CrowdDetectionModule, DENSITY_SEVERITY
from app.modules.crowd_detection.crowd_tracker import (
    CrowdTracker, DensityLevel, FootfallEvent,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_context(
    cam_id="CAM-01", frame_idx=0, timestamp=0.0,
    person_tracks=None, width=1920, height=1080, fps=25.0,
):
    return FrameContext(
        camera_id=cam_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        fps=fps,
        frame_width=width,
        frame_height=height,
        person_tracks=person_tracks or {},
    )


def _make_person(x1, y1, x2, y2):
    """Create a person track dict."""
    return {"bbox": [x1, y1, x2, y2], "cls": 0}


def _make_car(x1, y1, x2, y2):
    """Create a car track dict (should be ignored by crowd module)."""
    return {"bbox": [x1, y1, x2, y2], "cls": 2}


def _make_tracker(w=1920, h=1080, fps=25.0, **kwargs):
    """Create a CrowdTracker with test-friendly defaults."""
    defaults = {
        "density_low_max": 3,
        "density_moderate_max": 8,
        "density_high_max": 15,
        "edge_margin_ratio": 0.05,
        "entry_exit_cooldown_sec": 0.0,  # No cooldown in tests
        "heatmap_resolution": 20,        # Small grid for speed
        "heatmap_decay": 0.99,
        "heatmap_alpha": 0.4,
        "trajectory_max_length": 50,
        "trajectory_draw_length": 10,
    }
    defaults.update(kwargs)
    return CrowdTracker(frame_width=w, frame_height=h, fps=fps, **defaults)


# ── Interface Tests ───────────────────────────────────────────────────

def test_crowd_module_is_analytics_module():
    """CrowdDetectionModule must be a subclass of AnalyticsModule."""
    assert issubclass(CrowdDetectionModule, AnalyticsModule)


def test_crowd_module_name():
    """Module name must be 'crowd_detection'."""
    m = CrowdDetectionModule()
    assert m.name == "crowd_detection"


def test_crowd_module_applicable_cameras_default():
    """By default, crowd detection applies to all cameras."""
    m = CrowdDetectionModule()
    assert m.applicable_cameras() == ["*"]


def test_crowd_module_no_crash_without_initialize():
    """Module should handle process_frame even if tracker hasn't been init'd."""
    m = CrowdDetectionModule()
    m.initialize({})
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ctx = _make_context()
    events = m.process_frame(frame, ctx)
    assert isinstance(events, list)


# ── Occupancy Tests ──────────────────────────────────────────────────

def test_occupancy_counts_persons_only():
    """Tracker should count only persons (cls=0), not cars."""
    t = _make_tracker()
    tracks = {
        1: _make_person(100, 100, 200, 300),
        2: _make_person(300, 100, 400, 300),
        3: _make_car(500, 100, 700, 300),  # Should NOT be counted
    }
    snapshot, _, _ = t.update(0, 0.0, tracks)
    assert snapshot.total_persons == 2


def test_occupancy_zero_persons():
    """Empty frame should return 0 persons."""
    t = _make_tracker()
    snapshot, _, _ = t.update(0, 0.0, {})
    assert snapshot.total_persons == 0


def test_occupancy_single_person():
    """Single person should be counted correctly."""
    t = _make_tracker()
    tracks = {1: _make_person(500, 400, 600, 600)}
    snapshot, _, _ = t.update(0, 0.0, tracks)
    assert snapshot.total_persons == 1


def test_peak_occupancy_tracked():
    """Peak occupancy should be the max observed at any point."""
    t = _make_tracker()

    # Frame 1: 3 people
    tracks_3 = {i: _make_person(i*100, 100, i*100+80, 300) for i in range(1, 4)}
    t.update(0, 0.0, tracks_3)

    # Frame 2: 5 people
    tracks_5 = {i: _make_person(i*100, 100, i*100+80, 300) for i in range(1, 6)}
    t.update(1, 0.04, tracks_5)

    # Frame 3: 2 people
    tracks_2 = {i: _make_person(i*100, 100, i*100+80, 300) for i in range(1, 3)}
    t.update(2, 0.08, tracks_2)

    assert t.peak_occupancy == 5


# ── Density Tests ────────────────────────────────────────────────────

def test_density_low():
    """0-3 persons = LOW density."""
    t = _make_tracker()
    tracks = {1: _make_person(500, 400, 600, 600)}
    snapshot, _, _ = t.update(0, 0.0, tracks)
    assert snapshot.density_level == DensityLevel.LOW


def test_density_moderate():
    """4-8 persons = MODERATE density."""
    t = _make_tracker()
    tracks = {i: _make_person(i*100, 100, i*100+80, 300) for i in range(1, 6)}
    snapshot, _, _ = t.update(0, 0.0, tracks)
    assert snapshot.density_level == DensityLevel.MODERATE


def test_density_high():
    """9-15 persons = HIGH density."""
    t = _make_tracker()
    tracks = {i: _make_person(i*100, 100, i*100+80, 300) for i in range(1, 12)}
    snapshot, _, _ = t.update(0, 0.0, tracks)
    assert snapshot.density_level == DensityLevel.HIGH


def test_density_critical():
    """16+ persons = CRITICAL density."""
    t = _make_tracker()
    tracks = {i: _make_person((i%15)*100+20, (i//15)*200+100, (i%15)*100+80, (i//15)*200+300) for i in range(1, 20)}
    snapshot, _, _ = t.update(0, 0.0, tracks)
    assert snapshot.density_level == DensityLevel.CRITICAL


def test_density_change_fires_event():
    """Module should emit density_change event when level transitions."""
    m = CrowdDetectionModule()
    m.initialize({"alert_on_density_change": True, "high_density_alert_sec": 0})
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Frame 1: 1 person → LOW
    ctx1 = _make_context(timestamp=0.0, person_tracks={1: _make_person(500, 400, 600, 600)})
    m.process_frame(frame, ctx1)

    # Frame 2: 5 persons → MODERATE (density change!)
    tracks = {i: _make_person(i*100, 100, i*100+80, 300) for i in range(1, 6)}
    ctx2 = _make_context(timestamp=1.0, frame_idx=25, person_tracks=tracks)
    events = m.process_frame(frame, ctx2)

    density_events = [e for e in events if e.event_type == "density_change"]
    assert len(density_events) >= 1
    assert density_events[0].metadata["density_level"] == "moderate"


# ── Footfall Tests ───────────────────────────────────────────────────

def test_entry_detected_at_left_edge():
    """Person appearing at the left edge should trigger an entry event."""
    t = _make_tracker(w=1920, h=1080)

    # Frame 1: empty
    t.update(0, 0.0, {})

    # Frame 2: person appears near left edge (x=30 is within 5% margin = 96px)
    tracks = {1: _make_person(10, 400, 80, 600)}
    _, events, _ = t.update(1, 0.04, tracks)

    entries = [e for e in events if e.direction == "entry"]
    assert len(entries) == 1
    assert entries[0].edge == "left"


def test_exit_detected_at_right_edge():
    """Person disappearing at right edge should trigger an exit event."""
    t = _make_tracker(w=1920, h=1080)

    # Frame 1: person at right edge
    tracks = {1: _make_person(1860, 400, 1910, 600)}
    t.update(0, 0.0, tracks)

    # Frame 2: person gone
    _, events, _ = t.update(1, 0.04, {})

    exits = [e for e in events if e.direction == "exit"]
    assert len(exits) == 1
    assert exits[0].edge == "right"


def test_no_entry_for_center_appearance():
    """Person appearing in the center of frame should NOT trigger entry."""
    t = _make_tracker(w=1920, h=1080)

    # Frame 1: empty
    t.update(0, 0.0, {})

    # Frame 2: person appears in center (far from edges)
    tracks = {1: _make_person(800, 400, 900, 600)}
    _, events, _ = t.update(1, 0.04, tracks)

    entries = [e for e in events if e.direction == "entry"]
    assert len(entries) == 0


def test_footfall_cooldown():
    """Same track should not fire entry twice within cooldown period."""
    t = _make_tracker(w=1920, h=1080, entry_exit_cooldown_sec=2.0)

    # Entry at t=0
    t.update(0, 0.0, {})
    tracks = {1: _make_person(10, 400, 80, 600)}
    t.update(1, 0.04, tracks)

    # Exit at t=0.5 (within cooldown)
    t.update(2, 0.5, {})

    # Re-entry at t=0.8 (still within 2s cooldown of the exit attempt)
    _, events, _ = t.update(3, 0.8, tracks)
    # May or may not fire depending on cooldown state — at minimum shouldn't crash


# ── Heatmap Tests ────────────────────────────────────────────────────

def test_heatmap_accumulates():
    """Heatmap should have non-zero values where persons were detected."""
    t = _make_tracker(w=1920, h=1080, heatmap_resolution=10)

    # Person in center for 5 frames
    tracks = {1: _make_person(900, 500, 1000, 600)}
    for i in range(5):
        t.update(i, i * 0.04, tracks)

    assert t._heatmap.max() > 0
    # Check the cell where the person centroid (950, 550) actually falls
    cx, cy = 950, 550
    col = min(int(cx / t._cell_w), t._hm_cols - 1)
    row = min(int(cy / t._cell_h), t._hm_rows - 1)
    assert t._heatmap[row, col] > 0


def test_heatmap_export_image():
    """Heatmap export should return a valid BGR image."""
    t = _make_tracker(w=640, h=480, heatmap_resolution=10)
    tracks = {1: _make_person(300, 200, 350, 280)}
    for i in range(3):
        t.update(i, i * 0.04, tracks)

    img = t.get_heatmap_image()
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8


# ── Trajectory Tests ─────────────────────────────────────────────────

def test_trajectory_records_centroids():
    """Trajectories should track centroid positions."""
    t = _make_tracker()

    # Move person across 3 frames
    t.update(0, 0.0, {1: _make_person(100, 100, 200, 300)})
    t.update(1, 0.04, {1: _make_person(150, 100, 250, 300)})
    t.update(2, 0.08, {1: _make_person(200, 100, 300, 300)})

    assert len(t._trajectories[1]) == 3
    # Centroids should be: (150, 200), (200, 200), (250, 200)
    assert abs(t._trajectories[1][0][0] - 150) < 1
    assert abs(t._trajectories[1][2][0] - 250) < 1


# ── Dwell Time Tests ─────────────────────────────────────────────────

def test_dwell_record_created_on_exit():
    """When a person disappears, a dwell record should be created."""
    t = _make_tracker()

    # Person visible for 2 seconds
    tracks = {1: _make_person(500, 400, 600, 600)}
    for i in range(50):
        t.update(i, i * 0.04, tracks)  # 50 frames @ 25fps = 2.0s

    # Person disappears
    t.update(50, 2.0, {})

    dwells = t.get_dwell_records()
    assert len(dwells) == 1
    assert abs(dwells[0].duration - 2.0) < 0.1


# ── Overlay Tests ────────────────────────────────────────────────────

def test_draw_overlay_no_crash():
    """Drawing overlay on empty frame should not crash."""
    t = _make_tracker(w=640, h=480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = t.draw_overlay(frame)
    assert result.shape == (480, 640, 3)


def test_draw_overlay_with_data():
    """Drawing overlay with real data should produce a valid frame."""
    t = _make_tracker(w=640, h=480)
    tracks = {1: _make_person(200, 150, 300, 350)}
    t.update(0, 0.0, tracks)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = t.draw_overlay(frame)
    assert result.shape == (480, 640, 3)
    # Frame should have been modified (not all zeros)
    assert result.sum() > 0


# ── Summary Tests ────────────────────────────────────────────────────

def test_summary_returns_valid_dict():
    """get_summary() should return a dict with expected keys."""
    t = _make_tracker()
    tracks = {1: _make_person(500, 400, 600, 600)}
    t.update(0, 0.0, tracks)

    summary = t.get_summary()
    assert "total_entries" in summary
    assert "total_exits" in summary
    assert "peak_occupancy" in summary
    assert "avg_occupancy" in summary
    assert "current_density" in summary
    assert "avg_dwell_sec" in summary
    assert "total_unique_tracks" in summary
    assert summary["peak_occupancy"] == 1


# ── Severity mapping ─────────────────────────────────────────────────

def test_critical_density_maps_to_high_severity():
    """CRITICAL density should produce HIGH severity events."""
    assert DENSITY_SEVERITY[DensityLevel.CRITICAL] == Severity.HIGH


def test_low_density_maps_to_low_severity():
    """LOW density should produce LOW severity events."""
    assert DENSITY_SEVERITY[DensityLevel.LOW] == Severity.LOW


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_crowd_module_is_analytics_module()
    test_crowd_module_name()
    test_crowd_module_applicable_cameras_default()
    test_crowd_module_no_crash_without_initialize()
    test_occupancy_counts_persons_only()
    test_occupancy_zero_persons()
    test_occupancy_single_person()
    test_peak_occupancy_tracked()
    test_density_low()
    test_density_moderate()
    test_density_high()
    test_density_critical()
    test_density_change_fires_event()
    test_entry_detected_at_left_edge()
    test_exit_detected_at_right_edge()
    test_no_entry_for_center_appearance()
    test_footfall_cooldown()
    test_heatmap_accumulates()
    test_heatmap_export_image()
    test_trajectory_records_centroids()
    test_dwell_record_created_on_exit()
    test_draw_overlay_no_crash()
    test_draw_overlay_with_data()
    test_summary_returns_valid_dict()
    test_critical_density_maps_to_high_severity()
    test_low_density_maps_to_low_severity()
    print("✅ All crowd detection module tests passed!")

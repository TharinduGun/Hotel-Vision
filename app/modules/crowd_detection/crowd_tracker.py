"""
Crowd Tracker — Internal Engine
=================================
Zone-independent crowd analytics that operates on shared person detections.

Features:
  1. Real-time occupancy counting
  2. Footfall tracking (frame-edge entry/exit)
  3. Spatial heat map accumulation
  4. Crowd density estimation
  5. Per-track movement trajectories
  6. Video overlay rendering

This class does NOT implement AnalyticsModule — it is the internal
engine used by CrowdDetectionModule.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


# ── Data Structures ───────────────────────────────────────────────────

class DensityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FootfallEvent:
    """Records a person entering or exiting the frame."""
    track_id: int
    direction: str          # "entry" | "exit"
    timestamp: float
    frame_idx: int
    edge: str               # "top" | "bottom" | "left" | "right"
    position: tuple[float, float]   # (cx, cy) where crossing happened


@dataclass
class DwellRecord:
    """Tracks how long a person was visible in the scene."""
    track_id: int
    entry_time: float
    exit_time: float = 0.0
    duration: float = 0.0
    entry_position: tuple[float, float] = (0, 0)
    exit_position: tuple[float, float] = (0, 0)


@dataclass
class CrowdSnapshot:
    """Per-frame crowd state summary."""
    frame_idx: int
    timestamp: float
    total_persons: int
    density_level: DensityLevel
    active_tracks: set = field(default_factory=set)


# ── Crowd Tracker ─────────────────────────────────────────────────────

class CrowdTracker:
    """
    Zone-independent crowd analytics engine.

    Consumes person_tracks from FrameContext each frame and produces
    occupancy counts, footfall events, heat maps, and density levels.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 25.0,
        # Density
        density_low_max: int = 3,
        density_moderate_max: int = 8,
        density_high_max: int = 15,
        # Footfall
        edge_margin_ratio: float = 0.05,
        entry_exit_cooldown_sec: float = 1.0,
        # Heatmap
        heatmap_resolution: int = 100,
        heatmap_decay: float = 0.998,
        heatmap_alpha: float = 0.4,
        # Trajectory
        trajectory_max_length: int = 300,
        trajectory_draw_length: int = 60,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        # ── Density thresholds ────────────────────────────────────
        self._density_low_max = density_low_max
        self._density_moderate_max = density_moderate_max
        self._density_high_max = density_high_max

        # ── Footfall ──────────────────────────────────────────────
        self._edge_margin = int(max(frame_width, frame_height) * edge_margin_ratio)
        self._entry_cooldown = entry_exit_cooldown_sec
        self._track_first_seen: dict[int, float] = {}       # track_id → timestamp
        self._track_last_seen: dict[int, float] = {}        # track_id → timestamp
        self._track_entry_pos: dict[int, tuple] = {}        # track_id → (cx, cy)
        self._last_entry_exit: dict[int, float] = {}        # track_id → last event time
        self._prev_tracks: set[int] = set()                 # tracks seen last frame

        # Accumulated results
        self._footfall_events: list[FootfallEvent] = []
        self._dwell_records: list[DwellRecord] = []
        self._total_entries: int = 0
        self._total_exits: int = 0

        # ── Heat map ─────────────────────────────────────────────
        self._hm_alpha = heatmap_alpha
        self._hm_decay = heatmap_decay
        # Grid resolution: scale longest axis to heatmap_resolution cells
        aspect = frame_width / frame_height
        if aspect >= 1.0:
            self._hm_cols = heatmap_resolution
            self._hm_rows = max(1, int(heatmap_resolution / aspect))
        else:
            self._hm_rows = heatmap_resolution
            self._hm_cols = max(1, int(heatmap_resolution * aspect))
        self._heatmap = np.zeros((self._hm_rows, self._hm_cols), dtype=np.float32)
        self._cell_w = frame_width / self._hm_cols
        self._cell_h = frame_height / self._hm_rows

        # ── Trajectories ─────────────────────────────────────────
        self._traj_max = trajectory_max_length
        self._traj_draw = trajectory_draw_length
        self._trajectories: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=trajectory_max_length)
        )

        # ── Occupancy history (for rolling average) ──────────────
        self._occupancy_window = deque(maxlen=int(fps * 2))   # 2-second window
        self._peak_occupancy: int = 0
        self._current_density: DensityLevel = DensityLevel.LOW
        self._density_history: list[tuple[float, DensityLevel]] = []

        # ── Snapshots (time-series for export) ───────────────────
        self._snapshots: list[CrowdSnapshot] = []

    # ── Main per-frame update ─────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        person_tracks: dict[int, dict],
    ) -> CrowdSnapshot:
        """
        Process one frame of person tracks and return crowd state.

        Args:
            frame_idx: Current frame index.
            timestamp: Seconds since stream start.
            person_tracks: { track_id: { "bbox": [x1,y1,x2,y2], "cls": int } }
                           Only cls==0 (persons) are used.

        Returns:
            CrowdSnapshot with current crowd state.
        """
        # Filter to persons only
        persons = {
            tid: t for tid, t in person_tracks.items()
            if t["cls"] == 0
        }
        current_ids = set(persons.keys())
        count = len(persons)

        # 1. Occupancy
        self._occupancy_window.append(count)
        self._peak_occupancy = max(self._peak_occupancy, count)

        # 2. Centroids + Heatmap + Trajectories
        centroids: dict[int, tuple[float, float]] = {}
        for tid, tdata in persons.items():
            bx = tdata["bbox"]
            cx = (bx[0] + bx[2]) / 2
            cy = (bx[1] + bx[3]) / 2
            centroids[tid] = (cx, cy)

            # Heatmap accumulation
            col = min(int(cx / self._cell_w), self._hm_cols - 1)
            row = min(int(cy / self._cell_h), self._hm_rows - 1)
            self._heatmap[row, col] += 1.0

            # Trajectory
            self._trajectories[tid].append((cx, cy))

        # Decay heatmap
        self._heatmap *= self._hm_decay

        # 3. Footfall detection (entry/exit)
        new_events = self._detect_footfall(current_ids, centroids, timestamp, frame_idx)

        # 4. Density level
        new_density = self._classify_density(count)
        density_changed = (new_density != self._current_density)
        if density_changed:
            self._density_history.append((timestamp, new_density))
        self._current_density = new_density

        # 5. Track lifetime (for dwell records)
        for tid in current_ids:
            if tid not in self._track_first_seen:
                self._track_first_seen[tid] = timestamp
                self._track_entry_pos[tid] = centroids.get(tid, (0, 0))
            self._track_last_seen[tid] = timestamp

        # Detect exits: tracks that were present last frame but are gone now
        exited = self._prev_tracks - current_ids
        for tid in exited:
            if tid in self._track_first_seen:
                entry_t = self._track_first_seen[tid]
                exit_t = self._track_last_seen.get(tid, timestamp)
                duration = exit_t - entry_t
                if duration > 0.5:  # Only record meaningful dwell times
                    self._dwell_records.append(DwellRecord(
                        track_id=tid,
                        entry_time=entry_t,
                        exit_time=exit_t,
                        duration=duration,
                        entry_position=self._track_entry_pos.get(tid, (0, 0)),
                        exit_position=centroids.get(tid, (0, 0)),
                    ))

        self._prev_tracks = current_ids.copy()

        # Clean up trajectories for tracks that have exited
        # Keep trajectory briefly for visual fade, then remove
        stale_tids = [tid for tid in self._trajectories
                      if tid not in current_ids and tid not in self._prev_tracks]

        # Build snapshot
        snapshot = CrowdSnapshot(
            frame_idx=frame_idx,
            timestamp=timestamp,
            total_persons=count,
            density_level=new_density,
            active_tracks=current_ids,
        )

        return snapshot, new_events, density_changed

    # ── Footfall Detection ────────────────────────────────────────

    def _detect_footfall(
        self,
        current_ids: set[int],
        centroids: dict[int, tuple],
        timestamp: float,
        frame_idx: int,
    ) -> list[FootfallEvent]:
        """Detect entry/exit events based on frame-edge appearance/disappearance."""
        events = []
        margin = self._edge_margin

        # New tracks (appeared this frame)
        new_tracks = current_ids - self._prev_tracks
        for tid in new_tracks:
            if tid in centroids:
                cx, cy = centroids[tid]
                edge = self._classify_edge(cx, cy, margin)
                if edge is not None:
                    # Cooldown check
                    last_t = self._last_entry_exit.get(tid, -999)
                    if (timestamp - last_t) >= self._entry_cooldown:
                        event = FootfallEvent(
                            track_id=tid,
                            direction="entry",
                            timestamp=timestamp,
                            frame_idx=frame_idx,
                            edge=edge,
                            position=(cx, cy),
                        )
                        events.append(event)
                        self._footfall_events.append(event)
                        self._total_entries += 1
                        self._last_entry_exit[tid] = timestamp

        # Lost tracks (disappeared this frame)
        lost_tracks = self._prev_tracks - current_ids
        for tid in lost_tracks:
            # Use last known position from trajectory
            if tid in self._trajectories and self._trajectories[tid]:
                cx, cy = self._trajectories[tid][-1]
                edge = self._classify_edge(cx, cy, margin)
                if edge is not None:
                    last_t = self._last_entry_exit.get(tid, -999)
                    if (timestamp - last_t) >= self._entry_cooldown:
                        event = FootfallEvent(
                            track_id=tid,
                            direction="exit",
                            timestamp=timestamp,
                            frame_idx=frame_idx,
                            edge=edge,
                            position=(cx, cy),
                        )
                        events.append(event)
                        self._footfall_events.append(event)
                        self._total_exits += 1
                        self._last_entry_exit[tid] = timestamp

        return events

    def _classify_edge(
        self, cx: float, cy: float, margin: int,
    ) -> Optional[str]:
        """Classify which frame edge a point is near (or None)."""
        if cx <= margin:
            return "left"
        if cx >= self.frame_width - margin:
            return "right"
        if cy <= margin:
            return "top"
        if cy >= self.frame_height - margin:
            return "bottom"
        return None

    # ── Density Classification ────────────────────────────────────

    def _classify_density(self, person_count: int) -> DensityLevel:
        """Map person count to a density level."""
        if person_count <= self._density_low_max:
            return DensityLevel.LOW
        elif person_count <= self._density_moderate_max:
            return DensityLevel.MODERATE
        elif person_count <= self._density_high_max:
            return DensityLevel.HIGH
        else:
            return DensityLevel.CRITICAL

    # ── Video Overlay ─────────────────────────────────────────────

    def draw_overlay(
        self,
        frame: np.ndarray,
        show_count: bool = True,
        show_density: bool = True,
        show_heatmap: bool = True,
        show_trajectories: bool = True,
    ) -> np.ndarray:
        """
        Draw crowd analytics overlay on the video frame.

        Args:
            frame: BGR numpy array (modified in-place).
            show_count: Show person count badge.
            show_density: Show density level indicator.
            show_heatmap: Blend heat map overlay.
            show_trajectories: Draw recent movement paths.

        Returns:
            The annotated frame.
        """
        h, w = frame.shape[:2]

        # 1. Heat map overlay
        if show_heatmap and self._heatmap.max() > 0:
            frame = self._draw_heatmap(frame)

        # 2. Trajectories
        if show_trajectories:
            self._draw_trajectories(frame)

        # 3. Count badge (top-right)
        if show_count:
            count = len(self._occupancy_window) and self._occupancy_window[-1] or 0
            avg = sum(self._occupancy_window) / max(len(self._occupancy_window), 1)
            label = f"People: {count}"
            avg_label = f"Avg: {avg:.1f} | Peak: {self._peak_occupancy}"

            # Background box
            pad = 12
            tw1, th1 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            tw2, th2 = cv2.getTextSize(avg_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            box_w = max(tw1, tw2) + pad * 2
            box_h = th1 + th2 + pad * 3
            bx = w - box_w - 10
            by = 10

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, label, (bx + pad, by + pad + th1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, avg_label, (bx + pad, by + pad * 2 + th1 + th2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 4. Density indicator (top-right, below count)
        if show_density:
            color_map = {
                DensityLevel.LOW: (0, 200, 0),       # Green
                DensityLevel.MODERATE: (0, 200, 255), # Yellow
                DensityLevel.HIGH: (0, 100, 255),     # Orange
                DensityLevel.CRITICAL: (0, 0, 255),   # Red
            }
            color = color_map.get(self._current_density, (200, 200, 200))
            density_text = f"Density: {self._current_density.value.upper()}"
            td_w, td_h = cv2.getTextSize(density_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            dy = 10 + (box_h if show_count else 0) + 8
            dx = w - td_w - 30

            # Indicator bar
            cv2.rectangle(frame, (dx - 8, dy), (dx + td_w + 8, dy + td_h + 12), color, -1)
            cv2.putText(frame, density_text, (dx, dy + td_h + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 5. Footfall counter (bottom-left)
        ff_text = f"In: {self._total_entries}  Out: {self._total_exits}"
        tw, th = cv2.getTextSize(ff_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fy = h - 20
        fx = 15

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (fx - 5, fy - th - 8), (fx + tw + 5, fy + 5), (30, 30, 30), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, ff_text, (fx, fy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _draw_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Blend the accumulated heat map onto the frame."""
        # Normalize heatmap to 0-255
        hm = self._heatmap.copy()
        max_val = hm.max()
        if max_val > 0:
            hm = (hm / max_val * 255).astype(np.uint8)
        else:
            return frame

        # Apply colormap
        hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # Resize to frame dimensions
        hm_resized = cv2.resize(hm_colored, (self.frame_width, self.frame_height),
                                interpolation=cv2.INTER_LINEAR)

        # Create mask: only show heatmap where there's actual heat
        mask = cv2.resize(hm, (self.frame_width, self.frame_height),
                          interpolation=cv2.INTER_LINEAR)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0

        # Blend: more heat = more overlay
        blended = frame.astype(np.float32)
        hm_float = hm_resized.astype(np.float32)
        alpha = mask_3ch * self._hm_alpha
        blended = blended * (1 - alpha) + hm_float * alpha

        return blended.astype(np.uint8)

    def _draw_trajectories(self, frame: np.ndarray) -> None:
        """Draw full movement trails for active tracks."""
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (200, 150, 50), (50, 200, 150), (150, 50, 200),
        ]

        for tid, traj in self._trajectories.items():
            # Only draw trajectories for currently active tracks
            if tid not in self._prev_tracks:
                continue
            if len(traj) < 2:
                continue

            # Draw the FULL stored trajectory (not just recent points)
            points = list(traj)
            # Optionally limit if configured
            if self._traj_draw > 0 and len(points) > self._traj_draw:
                points = points[-self._traj_draw:]

            color = colors[tid % len(colors)]
            total = len(points)

            for i in range(1, total):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                # Fade: older points are dimmer, newest are bright
                progress = i / total
                fade = int(200 * progress) + 55
                line_color = tuple(int(c * fade / 255) for c in color)
                # Thicker line near the current position
                thickness = 1 if progress < 0.7 else 2
                cv2.line(frame, pt1, pt2, line_color, thickness, lineType=cv2.LINE_AA)

    # ── Export / Summary Methods ──────────────────────────────────

    def get_heatmap_image(self) -> np.ndarray:
        """
        Generate a standalone heat map image (for export).

        Returns:
            BGR numpy array of the heat map at frame resolution.
        """
        hm = self._heatmap.copy()
        max_val = hm.max()
        if max_val > 0:
            hm = (hm / max_val * 255).astype(np.uint8)
        else:
            hm = np.zeros_like(hm, dtype=np.uint8)

        hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        return cv2.resize(hm_colored, (self.frame_width, self.frame_height),
                          interpolation=cv2.INTER_LINEAR)

    def get_footfall_events(self) -> list[FootfallEvent]:
        """Return all accumulated footfall events."""
        return list(self._footfall_events)

    def get_dwell_records(self) -> list[DwellRecord]:
        """Return all completed dwell records."""
        return list(self._dwell_records)

    def get_summary(self) -> dict:
        """
        Return aggregated crowd analytics summary.

        Returns:
            Dict with total entries/exits, peak/avg occupancy,
            density distribution, and dwell statistics.
        """
        # Average dwell
        dwells = [d.duration for d in self._dwell_records]
        avg_dwell = sum(dwells) / len(dwells) if dwells else 0.0
        max_dwell = max(dwells) if dwells else 0.0

        # Density time distribution
        density_dist = defaultdict(float)
        if self._density_history:
            for i in range(len(self._density_history)):
                level = self._density_history[i][1]
                t_start = self._density_history[i][0]
                t_end = (self._density_history[i + 1][0]
                         if i + 1 < len(self._density_history)
                         else t_start)
                density_dist[level.value] += (t_end - t_start)

        # Occupancy stats
        occ = list(self._occupancy_window) if self._occupancy_window else [0]

        return {
            "total_entries": self._total_entries,
            "total_exits": self._total_exits,
            "peak_occupancy": self._peak_occupancy,
            "avg_occupancy": round(sum(occ) / len(occ), 1),
            "current_density": self._current_density.value,
            "avg_dwell_sec": round(avg_dwell, 1),
            "max_dwell_sec": round(max_dwell, 1),
            "total_dwell_records": len(self._dwell_records),
            "total_unique_tracks": len(self._track_first_seen),
            "density_time_distribution": dict(density_dist),
            "footfall_events_count": len(self._footfall_events),
        }

    @property
    def current_density(self) -> DensityLevel:
        return self._current_density

    @property
    def peak_occupancy(self) -> int:
        return self._peak_occupancy

    @property
    def total_entries(self) -> int:
        return self._total_entries

    @property
    def total_exits(self) -> int:
        return self._total_exits

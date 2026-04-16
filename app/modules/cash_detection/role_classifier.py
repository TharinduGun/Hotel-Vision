"""
Role Classifier Module (v2 — Behavioral)
==========================================
Classifies tracked persons as "Cashier" (staff) or "Customer"
using a multi-signal approach that works with OR without ROI zones.

Signals:
  1. Zone presence (existing) — person in a cashier-type zone
  2. Stationarity — person hasn't moved much for a long time
  3. Visitor count — multiple unique people have come near this person

When zones are available, they dominate. When zones are missing,
behavioral signals provide a reasonable fallback.
"""

from collections import deque
from typing import Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


class RoleClassifier:
    """
    Multi-signal role classifier.

    Fuses three signals to determine if a person is staff:

      Score = zone_ratio × zone_weight
            + is_stationary × stationary_weight
            + has_visitors × visitor_weight

    If score >= role_threshold → "Cashier", else "Customer".

    When zones exist, zone_ratio dominates (weight 0.9).
    Without zones, stationarity (0.7) + visitors (0.5) can still
    reach the threshold (0.5).
    """

    # Zone types that count as "staff areas"
    STAFF_ZONE_TYPES = {"cashier", "cash_register", "money_exchange"}

    def __init__(
        self,
        cashier_threshold=0.50,
        # Signal weights
        zone_weight=0.9,
        stationary_weight=0.7,
        visitor_weight=0.5,
        # Stationarity params
        stationary_frames=450,        # ~18s at 25fps before considering stationary
        movement_threshold_px=20.0,   # Max center drift to be "stationary"
        # Visitor params
        visitor_count_threshold=3,    # Unique nearby IDs to count as "receiving visitors"
        visitor_proximity_px=250.0,   # How close someone must be to count as a visitor
        # Position tracking
        position_window=120,          # Rolling window for position tracking (~5s)
    ):
        self.cashier_threshold = cashier_threshold
        self.zone_weight = zone_weight
        self.stationary_weight = stationary_weight
        self.visitor_weight = visitor_weight
        self.stationary_frames = stationary_frames
        self.movement_threshold_px = movement_threshold_px
        self.visitor_count_threshold = visitor_count_threshold
        self.visitor_proximity_px = visitor_proximity_px
        self.position_window = position_window

        # { logical_id: { stats dict } }
        self._stats: Dict[int, dict] = {}

    def _init_stats(self, logical_id: int) -> dict:
        """Create a fresh stats entry for a new person."""
        stats = {
            "total": 0,
            "in_staff_zone": 0,
            "positions": deque(maxlen=self.position_window),
            "nearby_ids": set(),  # Unique person IDs that have been near this person
            "first_frame": 0,
            "has_zone_data": False,  # Have we ever received zone data for this person?
        }
        self._stats[logical_id] = stats
        return stats

    def update(
        self,
        logical_id: int,
        zone_name: str,
        zone_type: Optional[str],
        bbox: Optional[list] = None,
        frame_idx: int = 0,
        nearby_ids: Optional[Set[int]] = None,
    ):
        """
        Call once per frame per tracked person.

        Args:
            logical_id: The track's logical ID.
            zone_name: Name of the zone the track is in (or "Outside").
            zone_type: Type of the zone ("cashier", etc.) or None.
            bbox: Person bounding box [x1, y1, x2, y2] for position tracking.
            frame_idx: Current frame index.
            nearby_ids: Set of person IDs currently near this person.
        """
        if logical_id not in self._stats:
            stats = self._init_stats(logical_id)
            stats["first_frame"] = frame_idx
        else:
            stats = self._stats[logical_id]

        stats["total"] += 1

        # Signal 1: Zone occupancy
        if zone_type and zone_type in self.STAFF_ZONE_TYPES:
            stats["in_staff_zone"] += 1
            stats["has_zone_data"] = True
        elif zone_type is not None:
            # zone_type exists but isn't a staff zone — still means zones are active
            stats["has_zone_data"] = True

        # Signal 2: Position tracking (for stationarity)
        if bbox:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            stats["positions"].append((cx, cy, frame_idx))

        # Signal 3: Visitor tracking
        if nearby_ids:
            stats["nearby_ids"].update(nearby_ids - {logical_id})

    def get_role(self, logical_id: int) -> str:
        """
        Get the current role classification for a track.

        Returns:
            str: "Cashier" or "Customer"
        """
        if logical_id not in self._stats:
            return "Customer"

        stats = self._stats[logical_id]
        if stats["total"] == 0:
            return "Customer"

        score = 0.0

        # Signal 1: Zone ratio
        if stats["has_zone_data"]:
            zone_ratio = stats["in_staff_zone"] / stats["total"]
            score += zone_ratio * self.zone_weight

        # Signal 2: Stationarity
        if stats["total"] >= self.stationary_frames:
            is_stationary = self._check_stationary(stats)
            if is_stationary:
                score += self.stationary_weight

        # Signal 3: Visitor count
        if len(stats["nearby_ids"]) >= self.visitor_count_threshold:
            score += self.visitor_weight

        return "Cashier" if score >= self.cashier_threshold else "Customer"

    def _check_stationary(self, stats: dict) -> bool:
        """Check if the person's position has been stable over the tracking window."""
        positions = stats["positions"]
        if len(positions) < 30:
            return False

        recent = list(positions)[-60:]  # Last ~2.4s
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        return x_range < self.movement_threshold_px and y_range < self.movement_threshold_px

    def get_stats(self, logical_id: int) -> dict:
        """Get raw stats for debugging."""
        if logical_id not in self._stats:
            return {"in_staff_zone": 0, "total": 0, "visitors": 0, "stationary": False}

        stats = self._stats[logical_id]
        return {
            "in_staff_zone": stats["in_staff_zone"],
            "total": stats["total"],
            "visitors": len(stats["nearby_ids"]),
            "stationary": self._check_stationary(stats) if stats["total"] >= self.stationary_frames else False,
            "has_zone_data": stats["has_zone_data"],
        }

    def get_all_roles(self) -> Dict[int, str]:
        """Get a dict of all classified roles: {id: role}."""
        return {lid: self.get_role(lid) for lid in self._stats}

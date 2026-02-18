"""
Role Classifier Module
=======================
Classifies tracked persons as "Cashier" (staff) or "Customer"
based on what fraction of their tracked frames are spent inside
cashier-type ROI zones.
"""


class RoleClassifier:
    """
    Tracks per-ID zone occupancy and assigns a role label.
    
    Logic:
      - If a person spends >= `cashier_threshold` fraction of their
        tracked frames inside a "cashier"-type zone → role = "Cashier"
      - Otherwise → role = "Customer"
    """

    # Zone types that count as "staff areas"
    STAFF_ZONE_TYPES = {"cashier"}

    def __init__(self, cashier_threshold=0.60):
        """
        Args:
            cashier_threshold (float): Minimum fraction of frames a person
                must be in a cashier zone to be classified as staff.
                Default: 0.60 (60%).
        """
        self.cashier_threshold = cashier_threshold
        # { logical_id: { "in_staff_zone": int, "total": int } }
        self._stats = {}

    def update(self, logical_id, zone_name, zone_type):
        """
        Call once per frame per tracked person.
        
        Args:
            logical_id: The track's logical ID (after re-linking).
            zone_name (str): Name of the zone the track is in (or "Outside").
            zone_type (str|None): Type of the zone ("cashier", "money_exchange", etc.)
                                  None if outside all zones.
        """
        if logical_id not in self._stats:
            self._stats[logical_id] = {"in_staff_zone": 0, "total": 0}

        self._stats[logical_id]["total"] += 1

        if zone_type and zone_type in self.STAFF_ZONE_TYPES:
            self._stats[logical_id]["in_staff_zone"] += 1

    def get_role(self, logical_id):
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

        ratio = stats["in_staff_zone"] / stats["total"]
        return "Cashier" if ratio >= self.cashier_threshold else "Customer"

    def get_stats(self, logical_id):
        """Get raw stats for debugging."""
        return self._stats.get(logical_id, {"in_staff_zone": 0, "total": 0})

    def get_all_roles(self):
        """Get a dict of all classified roles: {id: role}."""
        return {lid: self.get_role(lid) for lid in self._stats}

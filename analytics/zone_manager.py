"""
Zone manager using Shapely polygons for point-in-polygon checks.

Loads zone definitions from zones.json and classifies (x, y) positions
into named zones. Also tracks which zones are designated as break zones.
"""

import json
from shapely.geometry import Point, Polygon


class ZoneManager:

    def __init__(self, config_path="configs/zones.json"):
        with open(config_path) as f:
            data = json.load(f)

        self.zones = {}
        zone_data = data.get("zones", data)

        for name, coords in zone_data.items():
            if coords and len(coords) >= 3:
                self.zones[name] = Polygon(coords)

        self.break_zones = set(data.get("break_zones", []))

    def get_zone(self, x, y):
        """
        Determine which zone a point falls in.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.

        Returns:
            Zone name string, or 'unknown' if outside all zones.
        """
        point = Point(x, y)
        for name, polygon in self.zones.items():
            if polygon.contains(point):
                return name
        return "unknown"

    def is_break_zone(self, zone_name):
        """Check if a zone is designated as a break zone."""
        return zone_name in self.break_zones
import json
import time


class JSONExporter:
    """
    FIX L-02: All field names updated from road-traffic to parking domain.
      traffic_state   →  occupancy_status
      lane_counts     →  space_counts
      total_vehicles  →  occupied  (current, not cumulative)
      flow_direction  →  removed (replaced by dwell_times)
    schema_version added for downstream consumers to detect breaking changes.
    """

    SCHEMA_VERSION = "2.0"

    def build(self, analytics: dict) -> str:

        payload = {
            "schema_version":   self.SCHEMA_VERSION,
            "timestamp":        time.time(),
            "occupied":         analytics.get("occupied", 0),
            "available":        analytics.get("available", 0),
            "capacity":         analytics.get("capacity", 0),
            "occupancy_pct":    analytics.get("occupancy_pct", 0.0),
            "occupancy_status": analytics.get("status", "unknown"),
            "space_counts":     analytics.get("space_counts", {}),
            "dwell_times":      analytics.get("dwell_times", {}),
            "vehicle_types":    analytics.get("vehicle_types", {}),
            "entry_count":      analytics.get("entry_count", 0),
        }

        return json.dumps(payload, indent=2)
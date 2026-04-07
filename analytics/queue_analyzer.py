"""
Counts people in the queue zone and tracks queue statistics over time.
"""


class QueueAnalyzer:

    def __init__(self, zone_manager):
        """
        Args:
            zone_manager: ZoneManager instance for point-in-zone checks.
        """
        self.zone_manager = zone_manager
        self.history = []

    def count(self, detections):
        """
        Count how many detected people are in the queue zone.

        Args:
            detections: List of detection dicts, each with a 'center' key (cx, cy).

        Returns:
            Integer count of people in the queue zone.
        """
        queue_count = 0
        for det in detections:
            cx, cy = det["center"]
            if self.zone_manager.get_zone(cx, cy) == "queue":
                queue_count += 1
        return queue_count

    def update(self, count):
        """Record a queue count for historical tracking."""
        self.history.append(count)

    def report(self):
        """Generate summary statistics from recorded queue counts."""
        if not self.history:
            return {"max_queue_length": 0, "average_queue_length": 0.0}

        return {
            "max_queue_length": max(self.history),
            "average_queue_length": round(sum(self.history) / len(self.history), 2),
        }
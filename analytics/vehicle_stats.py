class VehicleStats:
    """
    FIX M-04: The inflation bug (same vehicle counted ~30x) was in pipeline.py
    where the entire frame buffer was flattened before calling this.
    That deduplication is now done in pipeline.py before this is called,
    so this class receives one entry per unique vehicle per interval.
    """

    CLASS_MAP = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def compute(self, detections: list) -> dict:

        stats = {label: 0 for label in self.CLASS_MAP.values()}

        for det in detections:
            cls = det["class_id"]
            if cls in self.CLASS_MAP:
                stats[self.CLASS_MAP[cls]] += 1

        return stats
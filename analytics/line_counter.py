import cv2


class LineCounter:
    """
    FIX M-01: Entry line now loaded from configs/parking_spaces.py:ENTRY_LINE
    instead of being hardcoded at pixel (200,350,1000,350) which was specific
    to video1.mp4. Returns 0 safely if ENTRY_LINE is not yet configured.
    """

    def __init__(self):

        try:
            from configs.parking_spaces import ENTRY_LINE
            self.line = ENTRY_LINE      # (x1, y1, x2, y2) or None
        except (ImportError, AttributeError):
            self.line = None

        self.counted_ids: set = set()
        self.count = 0


    def check_crossing(self, detections: list) -> int:

        if self.line is None:
            return 0

        _, y1, _, _ = self.line

        for det in detections:
            tid = det["track_id"]
            bx1, by1, bx2, by2 = det["bbox"]
            cy = int((by1 + by2) / 2)

            if cy > y1 and tid not in self.counted_ids:
                self.count += 1
                self.counted_ids.add(tid)

        return self.count


    def draw_line(self, frame):
        """Optional: overlay entry line on frame for debugging."""
        if self.line is None:
            return frame
        x1, y1, x2, y2 = self.line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Entry: {self.count}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        return frame
    
    def reset_count(self):
        """Call this at midnight or start of each shift to reset daily entry count."""
        self.counted_ids.clear()
        self.count = 0
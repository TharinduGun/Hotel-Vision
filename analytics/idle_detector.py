"""
Detects when a tracked person has been stationary for too long.

Compares consecutive center positions. If movement stays below a pixel
threshold for more than frame_threshold consecutive frames, the person
is considered idle.
"""


class IdleDetector:

    def __init__(self, frame_threshold=150, movement_threshold=5):
        """
        Args:
            frame_threshold: Consecutive low-movement frames before flagging idle.
            movement_threshold: Max pixel movement that still counts as stationary.
        """
        self.frame_threshold = frame_threshold
        self.movement_threshold = movement_threshold
        self.positions = {}
        self.idle_frames = {}

    def update(self, track_id, position):
        """
        Update position for a track and check if idle.

        Args:
            track_id: Integer track ID.
            position: Tuple (cx, cy) center coordinates.

        Returns:
            True if the person has been idle beyond the threshold.
        """
        cx, cy = position

        if track_id not in self.positions:
            self.positions[track_id] = (cx, cy)
            self.idle_frames[track_id] = 0
            return False

        prev_x, prev_y = self.positions[track_id]
        dx = abs(cx - prev_x)
        dy = abs(cy - prev_y)

        if dx < self.movement_threshold and dy < self.movement_threshold:
            self.idle_frames[track_id] += 1
        else:
            self.idle_frames[track_id] = 0

        self.positions[track_id] = (cx, cy)

        return self.idle_frames[track_id] > self.frame_threshold

    def cleanup(self, active_track_ids):
        """Remove stale tracks that are no longer being detected."""
        stale = set(self.positions.keys()) - set(active_track_ids)
        for tid in stale:
            self.positions.pop(tid, None)
            self.idle_frames.pop(tid, None)
"""
Temporal Consistency Filter for Cash Detection.
Requires cash to be detected near a person in N out of M consecutive
recent frames before confirming the detection as real.

Ported from gun_detection/temporal_filter.py and adapted for cash
association keys (person IDs from cash_associations).
"""

from collections import deque


class CashTemporalFilter:
    """
    Sliding-window filter that suppresses flickering cash detections.

    A person is only considered to be "holding cash" if cash was
    associated with them in at least `min_frames` out of the last
    `window_size` frames.
    """

    def __init__(self, min_frames: int = 2, window_size: int = 5):
        """
        Args:
            min_frames: Min frames with a cash detection to confirm.
            window_size: Rolling window size (frames).
        """
        self.min_frames = min_frames
        self.window_size = window_size
        # { person_id: deque[bool] } — True = cash detected near this person
        self._history: dict[int, deque[bool]] = {}

    def update(
        self,
        active_person_ids: set[int],
        persons_with_cash: set[int],
    ) -> None:
        """
        Update the sliding window for all active persons.

        Args:
            active_person_ids: All person track IDs visible this frame.
            persons_with_cash: Person IDs that have cash associated this frame.
        """
        # Cleanup stale tracks (people no longer in frame)
        stale = [k for k in self._history if k not in active_person_ids]
        for k in stale:
            del self._history[k]

        # Update all active tracks
        for pid in active_person_ids:
            if pid not in self._history:
                self._history[pid] = deque(maxlen=self.window_size)
            self._history[pid].append(pid in persons_with_cash)

    def is_consistent(self, person_id: int) -> bool:
        """
        Returns True if cash was detected near `person_id` in at least
        `min_frames` out of the current window.
        """
        if person_id not in self._history:
            return False
        return sum(self._history[person_id]) >= self.min_frames

    def filter_associations(self, cash_associations: dict) -> dict:
        """
        Filter cash_associations to only include persons that pass
        temporal consistency.

        Args:
            cash_associations: {"assigned": {pid: [CashDetection, ...]},
                                "unassigned": [...]}

        Returns:
            Filtered copy with inconsistent persons moved to unassigned.
        """
        filtered_assigned = {}
        extra_unassigned = []

        for pid, detections in cash_associations.get("assigned", {}).items():
            if self.is_consistent(pid):
                filtered_assigned[pid] = detections
            else:
                extra_unassigned.extend(detections)

        return {
            "assigned": filtered_assigned,
            "unassigned": cash_associations.get("unassigned", []) + extra_unassigned,
        }

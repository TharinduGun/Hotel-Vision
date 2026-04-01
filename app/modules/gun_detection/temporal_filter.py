"""
Temporal Consistency Filter for Gun Detection.
Requires a weapon to be detected in N out of M consecutive recent frames
for the same person (or globally if person tracking is off).
"""

from collections import deque

class TemporalFilter:
    def __init__(self, min_frames: int = 3, window_size: int = 5):
        self.min_frames = min_frames
        self.window_size = window_size
        # Maps person_id or 'global' -> deque of booleans
        self._history: dict[int | str, deque[bool]] = {}
        
    def update(self, active_keys: set[int | str], detected_keys: set[int | str]):
        """
        Update the sliding window for all active subjects.
        """
        # 1. Cleanup stale tracks (people no longer in frame)
        current_keys = list(self._history.keys())
        for k in current_keys:
            if k not in active_keys and k != "global":
                del self._history[k]
                
        # 2. Update all active tracks
        for k in active_keys:
            if k not in self._history:
                self._history[k] = deque(maxlen=self.window_size)
            self._history[k].append(k in detected_keys)
            
        # 3. Explicit update for 'global' if there's no person tracking
        if "global" in detected_keys or "global" in self._history:
            if "global" not in self._history:
                self._history["global"] = deque(maxlen=self.window_size)
            if "global" not in active_keys:
                self._history["global"].append("global" in detected_keys)
            
    def is_consistent(self, key: int | str) -> bool:
        """
        Returns True if the specified key has been detected at least
        min_frames times within the current window.
        """
        if key not in self._history:
            return False
        
        detected_count = sum(self._history[key])
        return detected_count >= self.min_frames

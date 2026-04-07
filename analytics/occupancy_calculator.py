import time
from collections import deque


class OccupancyCalculator:
   
    def __init__(self, total_spaces: int):
        if total_spaces <= 0:
            raise ValueError("Total spaces must be a positive integer")
        self.total_spaces = total_spaces
        self.current_ids: set = set()
        self.total_seen:  set = set()
        self.entry_times: dict = {}
        self._count_buffer = deque(maxlen=10)


    def update(self, detections: list) -> dict:

        now = time.time()
        self.current_ids = {d["track_id"] for d in detections}

        # Record first-seen time for new arrivals
        for tid in self.current_ids:
            if tid not in self.entry_times:
                self.entry_times[tid] = now

        # Remove entry records for vehicles that left
        for tid in set(self.entry_times) - self.current_ids:
            del self.entry_times[tid]

        self.total_seen.update(self.current_ids)

        raw_occupied  = len(self.current_ids)
        self._count_buffer.append(raw_occupied)
        sorted_buf = sorted(self._count_buffer)
        mid        = len(sorted_buf) // 2
        if len(sorted_buf) % 2 == 0:
            occupied = round((sorted_buf[mid - 1] + sorted_buf[mid]) / 2)
        else:
            occupied = sorted_buf[mid]
        available = max(0, self.total_spaces - occupied)
        pct       = round(occupied / self.total_spaces * 100, 1)

        if pct >= 90:
            status = "full"
        elif pct >= 70:
            status = "limited"
        else:
            status = "available"

        return {
            "occupied":      occupied,
            "available":     available,
            "capacity":      self.total_spaces,
            "occupancy_pct": pct,
            "status":        status,
            "total_seen":    len(self.total_seen),
        }
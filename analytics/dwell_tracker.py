import time


class ParkingDwellTracker:
    """
    FIX C-03: Replaces FlowAnalyzer entirely.
    FlowAnalyzer tracked "up/down" pixel movement — pure noise for parked
    vehicles. This tracks how long each vehicle has been in the lot (dwell time),
    which is the actually useful metric for a cafe/hotel parking context.
    """

    def __init__(self):
        self.entry_times: dict = {}   # tid → unix timestamp of first detection


    def update(self, detections: list) -> dict:

        now         = time.time()
        current_ids = {d["track_id"] for d in detections}

        # Register new arrivals
        for tid in current_ids:
            if tid not in self.entry_times:
                self.entry_times[tid] = now

        # Remove departed vehicles
        for tid in set(self.entry_times) - current_ids:
            del self.entry_times[tid]

        dwell_times = {
            tid: round(now - t, 1)
            for tid, t in self.entry_times.items()
        }

        avg_dwell = round(sum(dwell_times.values()) / len(dwell_times), 1) \
                    if dwell_times else 0.0

        return {
            "per_vehicle":      dwell_times,   # {track_id: seconds_parked}
            "average_seconds":  avg_dwell,
        }
import csv
import os
import time


class CSVLogger:
    """
    FIX H-06: File is now opened once in __init__ and kept open.
    Old code did open()/close() on every single log() call — hundreds of
    cycles per hour, plus corruption risk on mid-write crash.
    Now flushes to disk after each row instead.

    FIX C-05/L-02: Output file renamed parking_data.csv, columns updated
    to parking domain (occupancy_pct, status, avg_dwell_seconds, etc.)
    Old columns: timestamp, total, congestion, cars, buses, trucks, motorcycles
    """

    FIELDNAMES = [
        "timestamp",
        "occupied",
        "available",
        "capacity",
        "occupancy_pct",
        "status",
        "avg_dwell_seconds",
        "cars",
        "buses",
        "trucks",
        "motorcycles",
        "entry_count",
    ]

    def __init__(self, file_path: str = "data/outputs/parking_data.csv"):

        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        write_header = not os.path.exists(self.file_path)

        self._file   = open(self.file_path, mode='a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)

        if write_header:
            self._writer.writeheader()
            self._file.flush()


    def log(self, analytics: dict):

        dwell     = analytics.get("dwell_times", {})
        avg_dwell = dwell.get("average_seconds", 0.0) if isinstance(dwell, dict) else 0.0
        vtypes    = analytics.get("vehicle_types", {})

        self._writer.writerow({
            "timestamp":         time.time(),
            "occupied":          analytics.get("occupied", 0),
            "available":         analytics.get("available", 0),
            "capacity":          analytics.get("capacity", 0),
            "occupancy_pct":     analytics.get("occupancy_pct", 0.0),
            "status":            analytics.get("status", "unknown"),
            "avg_dwell_seconds": avg_dwell,
            "cars":              vtypes.get("car", 0),
            "buses":             vtypes.get("bus", 0),
            "trucks":            vtypes.get("truck", 0),
            "motorcycles":       vtypes.get("motorcycle", 0),
            "entry_count":       analytics.get("entry_count", 0),
        })
        self._file.flush()


    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
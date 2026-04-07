import time

from models.detector import VehicleDetector
from analytics.tracker import IouTracker
from analytics.parking_space_analyzer import ParkingSpaceAnalyzer
from analytics.occupancy_calculator import OccupancyCalculator
from analytics.dwell_tracker import ParkingDwellTracker
from analytics.line_counter import LineCounter
from analytics.json_exporter import JSONExporter
from analytics.vehicle_stats import VehicleStats
from utils.visualizer import draw
from utils.csv_logger import CSVLogger

# from services.websocket_server import update_data   # uncomment when ready


class ParkingPipeline:

    def __init__(self, total_spaces: int = 20):

        self.last_update_time = time.time()
        self.interval         = 1.0
        self.buffer           = []

        self.detector = VehicleDetector()
        self.tracker  = IouTracker()

        self.space_analyzer = ParkingSpaceAnalyzer()
        self.occupancy      = OccupancyCalculator(total_spaces)
        self.dwell          = ParkingDwellTracker()
        self.line_counter   = LineCounter()
        self.vehicle_stats  = VehicleStats()
        self.csv_logger = CSVLogger()


    def process(self, frame):

        detections = self.detector.detect(frame)
        detections = self.tracker.update(detections)

        self.buffer.append(detections)

        current_time = time.time()
        analytics    = None

        if current_time - self.last_update_time >= self.interval:

            # Deduplicate: one entry per unique vehicle (most recent bbox)
            latest_by_id = {}
            for frame_dets in self.buffer:
                for d in frame_dets:
                    latest_by_id[d["track_id"]] = d
            current_detections = list(latest_by_id.values())

            space_counts   = self.space_analyzer.analyze(current_detections)
            occupancy_data = self.occupancy.update(current_detections)
            dwell_data     = self.dwell.update(current_detections)
            vehicle_types  = self.vehicle_stats.compute(current_detections)
            entry_count    = self.line_counter.check_crossing(current_detections)

            analytics = {
                "occupied":      occupancy_data["occupied"],
                "available":     occupancy_data["available"],
                "capacity":      occupancy_data["capacity"],
                "occupancy_pct": occupancy_data["occupancy_pct"],
                "status":        occupancy_data["status"],
                "space_counts":  space_counts,
                "dwell_times":   dwell_data,
                "vehicle_types": vehicle_types,
                "entry_count":   entry_count,
            }

            self.buffer           = []
            self.last_update_time = current_time
            self.csv_logger.log(analytics)

            # update_data(analytics)   # uncomment when WebSocket is ready

        frame = draw(frame, detections)
        return frame, analytics
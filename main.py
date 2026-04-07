import cv2
import signal
import sys
import yaml
import logging

from core.video_loader import VideoLoader
from core.pipeline import ParkingPipeline

# ── Load config ───────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, cfg["logging"]["level"], logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg["logging"]["file"])
    ]
)

VIDEO_PATH           = cfg["video"]["source"]
TOTAL_PARKING_SPACES = cfg["parking"]["total_spaces"]


def main():

    loader   = VideoLoader(VIDEO_PATH)
    pipeline = ParkingPipeline(total_spaces=TOTAL_PARKING_SPACES)

    # ── Graceful shutdown — defined inside main() so it can see loader/pipeline
    def shutdown(sig, frame):
        print("\nShutting down...")
        pipeline.csv_logger.close()
        loader.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    frame_count = 0

    while True:

        frame = loader.read()
        if frame is None:
            break

        frame_count += 1

        if frame_count % 2 == 0:
            processed_frame, analytics = pipeline.process(frame)
        else:
            processed_frame = frame
            analytics       = None

        if analytics is not None:
            print("=== Parking Analytics ===")
            print(analytics)

        cv2.imshow("Parking Lot Monitor", processed_frame)

        if cv2.waitKey(1) == 27:   # ESC to quit
            break

    loader.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
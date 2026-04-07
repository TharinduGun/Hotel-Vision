"""
Offline single-video analysis tool.

Processes a video file with person detection + ByteTrack tracking,
zone classification, idle detection, and queue analysis.
Outputs an annotated video and a JSON report.
"""

import sys

import cv2
import json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_loader import VideoLoader
from core.detector import PersonDetector
from core.tracker import ByteTrackTracker
from analytics.zone_manager import ZoneManager
from analytics.idle_detector import IdleDetector
from analytics.queue_analyzer import QueueAnalyzer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(_ROOT, "data", "videos", "coffee_shop2.mp4")
ZONE_CONFIG = os.path.join(_ROOT, "configs", "zones.json")


def main():
    try:
        video = VideoLoader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    detector = PersonDetector()
    tracker = ByteTrackTracker(frame_rate=int(video.fps))
    zone_manager = ZoneManager(ZONE_CONFIG)
    idle_detector = IdleDetector()
    queue_analyzer = QueueAnalyzer(zone_manager)

    ret, frame = video.read_frame()
    if not ret:
        print("Error: Could not read first frame from video.")
        return

    os.makedirs("outputs/processed_video", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "outputs/processed_video/processed_output.mp4",
        fourcc, video.fps, (width, height)
    )

    frame_num = 0

    while ret:
        frame_num += 1

        raw_detections = detector.detect(frame)
        detections = tracker.update(raw_detections)

        queue_count = 0

        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]

            zone = zone_manager.get_zone(cx, cy)

            if zone == "queue":
                queue_count += 1

            is_idle = idle_detector.update(track_id, (cx, cy))

            color = (0, 0, 255) if is_idle else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id} | {zone}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            if is_idle:
                cv2.putText(frame, "IDLE", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        queue_analyzer.update(queue_count)

        cv2.putText(frame, f"Queue Count: {queue_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)

        cv2.imshow("Offline Analysis", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if cv2.getWindowProperty("Offline Analysis", cv2.WND_PROP_VISIBLE) < 1:
            break

        ret, frame = video.read_frame()

    report = queue_analyzer.report()
    with open("outputs/reports/report.json", "w") as f:
        json.dump(report, f, indent=4)

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processing complete — {frame_num} frames.")
    print(f"  Video  -> outputs/processed_video/processed_output.mp4")
    print(f"  Report -> outputs/reports/report.json")


if __name__ == "__main__":
    main()
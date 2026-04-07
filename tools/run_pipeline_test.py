"""
Quick integration test — runs detection + tracking + face recognition
on a single video without multi-threading, for debugging.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from core.detector import PersonDetector
from core.tracker import ByteTrackTracker
from Identity.face_detector import FaceDetector
from Identity.face_recognizer import FaceRecognizer
from Identity.identity_manager import IdentityManager
from events.event_builder import EventBuilder
from utils.track_matcher import match_face_to_track


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(_ROOT, "data", "videos", "coffee_shop2.mp4")
FACE_DETECT_INTERVAL = 15


def main():
    detector = PersonDetector()
    tracker = ByteTrackTracker()
    face_detector = FaceDetector()
    recognizer = FaceRecognizer()
    identity_manager = IdentityManager()
    event_builder = EventBuilder()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open {VIDEO_PATH}")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        raw_detections = detector.detect(frame)
        detections = tracker.update(raw_detections)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            track_id = det["track_id"]
            identity = identity_manager.get(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} {identity}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if frame_count % FACE_DETECT_INTERVAL == 0:
            faces = face_detector.detect(frame)

            for face in faces:
                identity = recognizer.recognize(face.embedding)
                bbox = face.bbox.astype(int)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (255, 0, 0), 2)
                cv2.putText(frame, identity, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if identity != "unknown":
                    matched_track = match_face_to_track(bbox.tolist(), detections)
                    if matched_track is not None:
                        identity_manager.assign(matched_track, identity)

                    event_builder.write_event(
                        employee_id=identity,
                        camera_id="CAM_TEST",
                        zone="counter",
                        status="ON_DUTY",
                    )

        try:
            cv2.imshow("Pipeline Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        except cv2.error:
            pass  # headless environment — skip display

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print(f"Test complete — {frame_count} frames processed.")


if __name__ == "__main__":
    main()
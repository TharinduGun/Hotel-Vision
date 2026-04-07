"""
Per-camera processing pipeline.

Runs YOLOv8 detection, ByteTrack tracking, face recognition, zone analysis,
idle detection, status resolution, and event logging in a single loop.
"""

import cv2
from core.detector import PersonDetector
from core.tracker import ByteTrackTracker
from Identity.face_detector import FaceDetector
from Identity.face_recognizer import FaceRecognizer
from Identity.identity_manager import IdentityManager
from analytics.zone_manager import ZoneManager
from analytics.queue_analyzer import QueueAnalyzer
from analytics.idle_detector import IdleDetector
from analytics.status_resolver import StatusResolver
from events.event_builder import EventBuilder
from utils.track_matcher import match_face_to_track


class CameraPipeline:

    def __init__(self, camera_id, source, zone_config="configs/zones.json",
                 face_detect_interval=15, queue_threshold=5,
                 enable_display=True, event_builder=None):
        """
        Args:
            camera_id: Unique camera identifier string.
            source: Video file path or RTSP URL.
            zone_config: Path to zones.json.
            face_detect_interval: Run face detection every N frames.
            queue_threshold: Queue count above this triggers a LONG_QUEUE event.
            enable_display: Whether to show the annotated video window.
            event_builder: Shared EventBuilder instance (creates one if None).
        """
        self.camera_id = camera_id
        self.source = source
        self.face_detect_interval = face_detect_interval
        self.queue_threshold = queue_threshold
        self.enable_display = enable_display

        # Detection + tracking (separate)
        self.detector = PersonDetector()
        self.tracker = ByteTrackTracker()

        # Face recognition
        self.face_detector = FaceDetector()
        self.recognizer = FaceRecognizer()

        # State managers
        self.identity_manager = IdentityManager()
        self.zone_manager = ZoneManager(zone_config)
        self.idle_detector = IdleDetector()
        self.queue_analyzer = QueueAnalyzer(self.zone_manager)
        self.status_resolver = StatusResolver()

        # Event output
        self.event_builder = event_builder or EventBuilder()

        self.frame_count = 0
        self.seen_employee_ids = set()

    def run(self):
        """Main processing loop. Blocks until the video ends or ESC is pressed."""
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[{self.camera_id}] Error: Cannot open source {self.source}")
            return

        print(f"[{self.camera_id}] Pipeline started — source: {self.source}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            self._process_frame(frame)

            if self.enable_display:
                if not self._show_frame(frame):
                    break

        cap.release()
        if self.enable_display:
            cv2.destroyWindow(self.camera_id)

        print(f"[{self.camera_id}] Pipeline stopped after {self.frame_count} frames.")

    def _process_frame(self, frame):
        """Run all analytics on a single frame."""

        # --- Step 1: Detect people (YOLOv8) ---
        raw_detections = self.detector.detect(frame)

        # --- Step 2: Track people (ByteTrack) ---
        detections = self.tracker.update(raw_detections)
        active_track_ids = [d["track_id"] for d in detections]

        # --- Step 3: Face recognition (every N frames, unidentified tracks only) ---
        if self.frame_count % self.face_detect_interval == 0 and len(detections) > 0:
            self._run_face_recognition(frame, detections)

        # --- Step 4: Per-person analytics ---
        for det in detections:
            track_id = det["track_id"]
            cx, cy = det["center"]
            x1, y1, x2, y2 = det["bbox"]

            identity = self.identity_manager.get(track_id)

            # Zone classification
            zone = self.zone_manager.get_zone(cx, cy)

            # Idle detection
            is_idle = self.idle_detector.update(track_id, (cx, cy))

            # Draw person bounding box
            color = (0, 0, 255) if is_idle else (0, 255, 0)
            label = f"{track_id}:{identity}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Status resolution and event logging (employees only)
            if identity != "unknown":
                self.seen_employee_ids.add(identity)

                status = self.status_resolver.resolve(
                    identity, zone, is_idle, self.zone_manager
                )

                self.event_builder.write_event(
                    employee_id=identity,
                    camera_id=self.camera_id,
                    zone=zone,
                    status=status,
                )

                # Draw status tag
                cv2.putText(frame, status, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Step 5: Clean up stale idle entries ---
        self.idle_detector.cleanup(active_track_ids)

        # --- Step 6: Queue analysis ---
        queue_count = self.queue_analyzer.count(detections)
        self.queue_analyzer.update(queue_count)

        if queue_count > self.queue_threshold:
            self.event_builder.write_event(
                employee_id="SYSTEM",
                camera_id=self.camera_id,
                zone="queue",
                status="LONG_QUEUE",
            )

        cv2.putText(frame, f"Queue: {queue_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- Step 7: Check for OFFLINE employees ---
        offline_employees = self.status_resolver.get_offline_employees(self.seen_employee_ids)
        for emp_id in offline_employees:
            self.event_builder.write_event(
                employee_id=emp_id,
                camera_id=self.camera_id,
                zone="unknown",
                status="OFFLINE",
            )

    def _run_face_recognition(self, frame, detections):
        """Detect faces and match them to tracked people."""
        faces = self.face_detector.detect(frame)

        for face in faces:
            identity = self.recognizer.recognize(face.embedding)

            if identity == "unknown":
                continue

            # Match face bbox to the correct person track
            face_bbox = face.bbox.astype(int).tolist()
            matched_track_id = match_face_to_track(face_bbox, detections)

            if matched_track_id is not None:
                self.identity_manager.assign(matched_track_id, identity)

            # Draw face bbox
            fb = face.bbox.astype(int)
            cv2.rectangle(frame, (fb[0], fb[1]), (fb[2], fb[3]), (255, 0, 0), 2)
            cv2.putText(frame, identity, (fb[0], fb[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def _show_frame(self, frame):
        """Display the annotated frame. Returns False if user pressed ESC."""
        cv2.imshow(self.camera_id, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        if cv2.getWindowProperty(self.camera_id, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True
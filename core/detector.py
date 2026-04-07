"""
Person detection using YOLOv8 (detection only, no built-in tracking).

Outputs detections in the format ByteTrack expects.
"""

import numpy as np
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        """
        Args:
            model_path: Path to YOLOv8 weights. Downloads automatically if not found.
            confidence: Minimum detection confidence threshold.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run person detection on a single frame.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            numpy array of shape (N, 6) where each row is
            [x1, y1, x2, y2, confidence, class_id].
            Returns empty array if no detections.
        """
        results = self.model(
            frame,
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)

        return np.hstack([xyxy, conf, cls])
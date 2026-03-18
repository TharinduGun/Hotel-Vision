"""
Gun Detector — Core Inference Engine
======================================
Loads a YOLOv8 model trained on weapon detection and runs inference.

Key feature: Person-ROI gated detection — only scans within person
bounding boxes to save compute and reduce false positives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class GunDetection:
    """A single gun/weapon detection in a frame."""
    bbox: list[float]           # [x1, y1, x2, y2] in full-frame coordinates
    confidence: float
    class_name: str             # "Handgun", "Rifle", etc.
    class_id: int
    person_id: int | None       # Associated person track ID (if ROI-gated)
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        self.center = (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


class GunDetector:
    """
    YOLOv8-based gun/weapon detector.

    Two operating modes:
      1. **Person-ROI mode** (default): Crop each person's bounding box
         region, run detection on the crop, then map results back to
         full-frame coordinates. Saves GPU compute and massively reduces
         false positives (walls, signs, etc.).

      2. **Full-frame mode**: Run detection on the entire frame.
         Used when no person tracks are available.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.55,
        device: str = "cuda",
        person_roi_only: bool = True,
        roi_padding: float = 0.30,      # Expand person bbox by 30% for context (arms/weapons)
        imgsz: int = 640,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Gun detection model not found: {model_path}\n"
                "Run pycode/scripts/train_gun_detector.py first."
            )

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.person_roi_only = person_roi_only
        self.roi_padding = roi_padding
        self.imgsz = imgsz
        self._class_names = self.model.names

        print(f"[GunDetector] Loaded model: {model_path}")
        print(f"[GunDetector] Classes: {self._class_names}")
        print(f"[GunDetector] Confidence: {conf_threshold}, "
              f"ROI-only: {person_roi_only}")

    def detect(
        self,
        frame: np.ndarray,
        person_tracks: dict[int, dict] | None = None,
    ) -> list[GunDetection]:
        """
        Detect weapons in a frame.

        Args:
            frame: BGR numpy array (full-frame).
            person_tracks: Optional person tracks for ROI-gated detection.
                          { track_id: { "bbox": [x1,y1,x2,y2], "cls": 0 } }

        Returns:
            List of GunDetection objects.
        """
        if self.person_roi_only and person_tracks:
            return self._detect_in_person_rois(frame, person_tracks)
        else:
            return self._detect_full_frame(frame)

    def _detect_full_frame(self, frame: np.ndarray) -> list[GunDetection]:
        """Run detection on the entire frame."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=0.45,                 # Lower IoU to prevent suppressing close weapons
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
        )

        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = self._class_names.get(cls_id, f"weapon_{cls_id}")

                detections.append(GunDetection(
                    bbox=bbox,
                    confidence=conf,
                    class_name=cls_name,
                    class_id=cls_id,
                    person_id=None,
                ))

        return detections

    def _detect_in_person_rois(
        self,
        frame: np.ndarray,
        person_tracks: dict[int, dict],
    ) -> list[GunDetection]:
        """
        Run detection only within each person's bounding box.

        For each person:
          1. Crop the person region (with padding for context)
          2. Run YOLO on the crop
          3. Map detections back to full-frame coordinates
          4. Tag each detection with the person_id
        """
        frame_h, frame_w = frame.shape[:2]
        detections = []

        # Only process person tracks (cls == 0)
        persons = {
            pid: data for pid, data in person_tracks.items()
            if data.get("cls") == 0
        }

        for pid, pdata in persons.items():
            px1, py1, px2, py2 = pdata["bbox"]
            pw = px2 - px1
            ph = py2 - py1

            # Expand bbox with padding for context
            pad_x = pw * self.roi_padding
            pad_y = ph * self.roi_padding
            cx1 = max(0, int(px1 - pad_x))
            cy1 = max(0, int(py1 - pad_y))
            cx2 = min(frame_w, int(px2 + pad_x))
            cy2 = min(frame_h, int(py2 + pad_y))

            # Skip tiny crops
            crop_w = cx2 - cx1
            crop_h = cy2 - cy1
            if crop_w < 50 or crop_h < 50:
                continue

            # Crop the person region
            crop = frame[cy1:cy2, cx1:cx2]

            # Run detection on the crop
            results = self.model.predict(
                source=crop,
                conf=self.conf_threshold,
                iou=0.45,             # Lower IoU to prevent suppressing close weapons
                device=self.device,
                verbose=False,
                imgsz=self.imgsz,
            )

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    # Get detection in crop coordinates
                    crop_bbox = boxes.xyxy[i].cpu().tolist()
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())
                    cls_name = self._class_names.get(cls_id, f"weapon_{cls_id}")

                    # Map back to full-frame coordinates
                    full_bbox = [
                        crop_bbox[0] + cx1,
                        crop_bbox[1] + cy1,
                        crop_bbox[2] + cx1,
                        crop_bbox[3] + cy1,
                    ]

                    detections.append(GunDetection(
                        bbox=full_bbox,
                        confidence=conf,
                        class_name=cls_name,
                        class_id=cls_id,
                        person_id=pid,
                    ))

        return detections

    def shutdown(self):
        """Release model and GPU memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[GunDetector] Shut down")

"""
Gun Detector — Core Inference Engine
======================================
Loads a YOLOv8 model trained on weapon detection and runs inference.

Key features:
  - Person-ROI gated detection (only scans within person bounding boxes)
  - Hand proximity filter (YOLOv8-pose wrist keypoints)
  - Bbox size validation (rejects oversized / extreme-ratio detections)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.shared.model_manager import model_manager


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

    Three layers of false-positive filtering:
      1. **Person-ROI mode**: Only scan within person bounding boxes.
      2. **Bbox size filter**: Reject detections that are too large relative
         to the person, or have extreme aspect ratios.
      3. **Hand proximity filter**: Use YOLOv8-pose wrist keypoints to only
         accept detections near a person's hands.
    """

    # COCO Pose keypoint indices for wrists
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.55,
        device: str = "cuda",
        person_roi_only: bool = True,
        roi_padding: float = 0.30,
        imgsz: int = 640,
        # ── Hand proximity filtering ──────────────────
        hand_proximity_filter: bool = True,
        pose_model_path: str = "yolov8m-pose.pt",
        hand_radius_ratio: float = 0.4,
        # ── Bbox size filtering ───────────────────────
        max_weapon_area_ratio: float = 0.40,
        max_aspect_ratio: float = 5.0,
        # ── Minimum size filtering ────────────────────
        min_weapon_pixels: int = 900,
        min_weapon_height_ratio: float = 0.05,
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

        # Size filter params
        self.max_weapon_area_ratio = max_weapon_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_weapon_pixels = min_weapon_pixels
        self.min_weapon_height_ratio = min_weapon_height_ratio

        # Hand proximity params
        self.hand_proximity_filter = hand_proximity_filter
        self.hand_radius_ratio = hand_radius_ratio
        self._pose_model: YOLO | None = None
        self._pose_model_path = pose_model_path

        print(f"[GunDetector] Loaded model: {model_path}")
        print(f"[GunDetector] Classes: {self._class_names}")
        print(f"[GunDetector] Confidence: {conf_threshold}, "
              f"ROI-only: {person_roi_only}")
        print(f"[GunDetector] Hand proximity filter: {hand_proximity_filter}")
        print(f"[GunDetector] Max weapon/person area ratio: {max_weapon_area_ratio}")
        print(f"[GunDetector] Min weapon pixels: {min_weapon_pixels}, "
              f"min height ratio: {min_weapon_height_ratio}")

    # ── Lazy pose model loading ───────────────────────────────────────

    def _get_pose_model(self):
        """Deprecated: Model loading removed, using shared pose keypoints."""
        pass

    # ── Public API ────────────────────────────────────────────────────

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

    # ── Full-frame detection (fallback) ───────────────────────────────

    def _detect_full_frame(self, frame: np.ndarray) -> list[GunDetection]:
        """Run detection on the entire frame."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=0.45,
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

    # ── Person-ROI gated detection (primary mode) ─────────────────────

    def _detect_in_person_rois(
        self,
        frame: np.ndarray,
        person_tracks: dict[int, dict],
    ) -> list[GunDetection]:
        """
        Run detection only within each person's bounding box.

        Filtering pipeline per detection:
          1. Bbox size check — reject if weapon area > max_weapon_area_ratio
             of person area, or if aspect ratio is extreme.
          2. Hand proximity check — reject if weapon center is too far from
             any wrist keypoint of the associated person.
        """
        frame_h, frame_w = frame.shape[:2]

        # Only process person tracks (cls == 0)
        persons = {
            pid: data for pid, data in person_tracks.items()
            if data.get("cls") == 0
        }

        if not persons:
            return []

        # ── Get wrist keypoints for all persons (one pose pass) ───────
        person_wrists: dict[int, list[tuple[float, float]]] = {}
        if self.hand_proximity_filter:
            person_wrists = self._get_person_wrists(frame, persons)

        # ── Run weapon detection per person ROI ───────────────────────
        detections = []

        for pid, pdata in persons.items():
            px1, py1, px2, py2 = pdata["bbox"]
            pw = px2 - px1
            ph = py2 - py1
            person_area = pw * ph

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
                iou=0.45,
                device=self.device,
                verbose=False,
                imgsz=self.imgsz,
            )

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue

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

                # ── FILTER 1: Bbox size check ─────────────────────
                wep_w = full_bbox[2] - full_bbox[0]
                wep_h = full_bbox[3] - full_bbox[1]
                wep_area = wep_w * wep_h

                # Reject if weapon area > max ratio of person area
                if person_area > 0 and (wep_area / person_area) > self.max_weapon_area_ratio:
                    continue

                # Reject extreme aspect ratios
                aspect = max(wep_w, wep_h) / max(min(wep_w, wep_h), 1)
                if aspect > self.max_aspect_ratio:
                    continue

                # Reject tiny detections (noise)
                if wep_area < self.min_weapon_pixels:
                    continue

                # Reject if weapon height is too small relative to person
                if ph > 0 and (wep_h / ph) < self.min_weapon_height_ratio:
                    continue

                # ── FILTER 2: Hand proximity check ────────────────
                if self.hand_proximity_filter and pid in person_wrists:
                    wrists = person_wrists[pid]
                    if wrists:
                        # Weapon center
                        wep_cx = (full_bbox[0] + full_bbox[2]) / 2
                        wep_cy = (full_bbox[1] + full_bbox[3]) / 2

                        # Max allowed distance = hand_radius_ratio * person height
                        max_dist = self.hand_radius_ratio * ph

                        # Check if weapon center is near ANY wrist
                        near_hand = False
                        for wx, wy in wrists:
                            dist = math.hypot(wep_cx - wx, wep_cy - wy)
                            if dist <= max_dist:
                                near_hand = True
                                break

                        if not near_hand:
                            continue  # Too far from hands → reject

                detections.append(GunDetection(
                    bbox=full_bbox,
                    confidence=conf,
                    class_name=cls_name,
                    class_id=cls_id,
                    person_id=pid,
                ))

        return detections

    # ── Pose estimation helpers ───────────────────────────────────────

    def _get_person_wrists(
        self,
        frame: np.ndarray,
        persons: dict[int, dict],
    ) -> dict[int, list[tuple[float, float]]]:
        """
        Extract wrist keypoints from shared person tracks.

        Returns:
            { person_id: [(wrist_x, wrist_y), ...] }  (up to 2 wrists per person)
        """
        person_wrists: dict[int, list[tuple[float, float]]] = {
            pid: [] for pid in persons
        }

        for pid, pdata in persons.items():
            kpts = pdata.get("keypoints")
            if not kpts:
                continue
                
            # Extract valid wrists
            if len(kpts) > self.LEFT_WRIST_IDX:
                lw_x, lw_y, lw_conf = kpts[self.LEFT_WRIST_IDX]
                if lw_conf >= 0.3:
                    person_wrists[pid].append((float(lw_x), float(lw_y)))

            if len(kpts) > self.RIGHT_WRIST_IDX:
                rw_x, rw_y, rw_conf = kpts[self.RIGHT_WRIST_IDX]
                if rw_conf >= 0.3:
                    person_wrists[pid].append((float(rw_x), float(rw_y)))

        return person_wrists

    @staticmethod
    def _compute_iou(box1, box2) -> float:
        """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 1e-6)
        area2 = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 1e-6)
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def shutdown(self):
        """Release model and GPU memory."""
        del self.model
        if self._pose_model is not None:
            self._pose_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[GunDetector] Shut down")

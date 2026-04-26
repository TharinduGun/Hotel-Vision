"""
Person Detector (Shared Service)
=================================
Wraps YOLOv8 person/object detection and tracking so that all modules
can share the same person tracks without running detection multiple times.

Extracted from the original pycode/src/main.py logic.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from ultralytics import YOLO


class PersonDetector:
    """
    Shared YOLO-based person (and optionally vehicle) detector with tracking.

    Runs once per frame. All modules receive the resulting person_tracks
    through the FrameContext — no duplicate inference.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'shared' section from system_config.yaml.
        """
        shared = config.get("shared", {})
        model_path = shared.get("person_model_path", "yolov8m.pt")
        self.device = self._resolve_device(shared.get("device", "auto"))
        self.conf = shared.get("person_conf", 0.20)
        self.iou = shared.get("person_iou", 0.5)
        self.imgsz = shared.get("person_imgsz", 960)
        self.classes = shared.get("person_classes", [0, 2])
        self.tracker_config = shared.get("tracker_config")

        print(f"[PersonDetector] Loading model: {model_path} on {self.device}")
        self.model = YOLO(model_path).to(self.device)
        print(f"[PersonDetector] Ready (conf={self.conf}, iou={self.iou}, "
              f"imgsz={self.imgsz}, classes={self.classes})")

    def track(self, source: str) -> Any:
        """
        Run tracking on a video source (used in streaming mode).

        Args:
            source: Path to video file or RTSP URL.

        Returns:
            YOLO results generator (streamed, one result per frame).
        """
        with torch.no_grad():
            return self.model.track(
                source=source,
                persist=True,
                classes=self.classes,
                tracker=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                stream=True,
                verbose=False,
            )

    def detect_single(self, frame) -> dict[int, dict]:
        """
        Run detection on a single frame (no tracking persistence).

        Returns:
            { track_id: { "bbox": [x1,y1,x2,y2], "cls": int } }
        """
        with torch.no_grad():
            results = self.model.track(
                source=frame,
                persist=True,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )

        tracks = {}
        if results and results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            cls = results[0].boxes.cls.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            kpts = None
            if results[0].keypoints is not None and results[0].keypoints.data is not None:
                kpts = results[0].keypoints.data.cpu().tolist()
                
            for i, (tid, c, box) in enumerate(zip(ids, cls, boxes)):
                track_data = {"bbox": box, "cls": c}
                if kpts is not None and i < len(kpts):
                    track_data["keypoints"] = kpts[i]
                tracks[tid] = track_data

        return tracks

    @staticmethod
    def parse_result(result) -> dict[int, dict]:
        """
        Extract track data from a single YOLO result object.

        Args:
            result: One item from the model.track() generator.

        Returns:
            { track_id: { "bbox": [x1,y1,x2,y2], "cls": int } }
        """
        tracks = {}
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            cls_list = result.boxes.cls.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            kpts = None
            if result.keypoints is not None and result.keypoints.data is not None:
                kpts = result.keypoints.data.cpu().tolist()
                
            for i, (tid, c, box) in enumerate(zip(ids, cls_list, boxes)):
                track_data = {"bbox": box, "cls": c}
                if kpts is not None and i < len(kpts):
                    track_data["keypoints"] = kpts[i]
                tracks[tid] = track_data
        return tracks

    def shutdown(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("[PersonDetector] Shut down")

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str

"""
Camera Manager
===============
Manages video sources (local files today, RTSP streams later).
Provides a clean iterator interface so modules never touch OpenCV directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass
class CameraInfo:
    """Static info about a camera source."""
    camera_id: str
    name: str
    source: str
    source_type: str        # "file" | "rtsp"
    fps: float = 25.0
    width: int = 0
    height: int = 0
    total_frames: int = 0


class CameraStream:
    """
    Wraps a single cv2.VideoCapture with metadata.

    Usage:
        stream = CameraStream("CAM-01", "lobby", "/path/to/video.mp4")
        for frame_idx, frame in stream:
            ...  # process frame
        stream.close()
    """

    def __init__(
        self,
        camera_id: str,
        name: str,
        source: str,
        source_type: str = "file",
        max_seconds: float | None = None,
    ):
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.source_type = source_type
        self.max_seconds = max_seconds

        self._cap: cv2.VideoCapture | None = None
        self._info: CameraInfo | None = None

    def open(self) -> CameraInfo:
        """Open the video source and return camera info."""
        if not os.path.exists(self.source) and self.source_type == "file":
            raise FileNotFoundError(f"Video not found: {self.source}")

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.source}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-3:
            fps = 25.0

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Limit frames if max_seconds is set
        if self.max_seconds is not None:
            max_frames = int(fps * self.max_seconds)
            total_frames = min(total_frames, max_frames)

        self._info = CameraInfo(
            camera_id=self.camera_id,
            name=self.name,
            source=self.source,
            source_type=self.source_type,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
        )

        print(f"[CameraManager] Opened {self.camera_id}: "
              f"{width}x{height} @ {fps:.1f}fps, {total_frames} frames")

        return self._info

    @property
    def info(self) -> CameraInfo:
        if self._info is None:
            raise RuntimeError("Stream not opened. Call open() first.")
        return self._info

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yields (frame_idx, frame_bgr) tuples.

        The stream stops at max_seconds (if set) or end of video.
        """
        if self._cap is None:
            raise RuntimeError("Stream not opened. Call open() first.")

        max_frames = self._info.total_frames if self._info else float("inf")
        frame_idx = 0

        while self._cap.isOpened() and frame_idx < max_frames:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def close(self):
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print(f"[CameraManager] Closed {self.camera_id}")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class CameraManager:
    """
    Factory that creates CameraStream instances from config.

    Usage:
        mgr = CameraManager(cfg)
        for cam_id, stream in mgr.streams():
            with stream:
                for idx, frame in stream.frames():
                    ...
    """

    def __init__(self, config: dict):
        self._cameras_cfg = config.get("cameras", {})

    def get_stream(self, camera_id: str) -> CameraStream:
        """Create a CameraStream for a single camera."""
        cam_cfg = self._cameras_cfg.get(camera_id)
        if cam_cfg is None:
            raise KeyError(f"Camera '{camera_id}' not found in config")

        return CameraStream(
            camera_id=camera_id,
            name=cam_cfg.get("name", camera_id),
            source=cam_cfg["source"],
            source_type=cam_cfg.get("type", "file"),
            max_seconds=cam_cfg.get("run_seconds"),
        )

    def streams(self) -> Iterator[Tuple[str, CameraStream]]:
        """Yield (camera_id, CameraStream) for all configured cameras."""
        for cam_id in self._cameras_cfg:
            yield cam_id, self.get_stream(cam_id)

    @property
    def camera_ids(self) -> list[str]:
        """All configured camera IDs."""
        return list(self._cameras_cfg.keys())

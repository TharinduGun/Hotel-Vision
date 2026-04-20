"""
Snapshot Saver (Shared Service)
================================
Saves annotated frame snapshots and short video clips when
high-severity events (like gun detection) are triggered.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import deque

import cv2
import numpy as np

from app.contracts.event_schema import AnalyticsEvent


class SnapshotSaver:
    """
    Saves evidence for analytics events:
      - Single frame snapshots (JPEG)
      - Short video clips around the detection moment

    The saver maintains a rolling frame buffer so it can save
    a few seconds BEFORE the detection, not just after.
    """

    def __init__(
        self,
        session_dir: str | Path,
        buffer_seconds: float = 3.0,
        fps: float = 25.0,
    ):
        self.session_dir = Path(session_dir)
        self.snapshots_dir = self.session_dir / "snapshots"
        self.clips_dir = self.session_dir / "clips"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.buffer_size = int(buffer_seconds * fps)
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self._post_capture: dict[str, dict] = {}  # event_id -> capture state

        print(f"[SnapshotSaver] Ready (buffer={buffer_seconds}s, "
              f"snapshots={self.snapshots_dir})")

    def buffer_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the rolling buffer (call every frame)."""
        self._frame_buffer.append(frame.copy())

    def save_snapshot(
        self,
        frame: np.ndarray,
        event: AnalyticsEvent,
        annotate: bool = True,
    ) -> str:
        """
        Save a single frame as a JPEG snapshot.

        Args:
            frame: The BGR frame to save.
            event: The event that triggered the snapshot.
            annotate: If True, draw the detection bbox on the snapshot.

        Returns:
            Path to the saved snapshot image.
        """
        if annotate:
            frame = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in event.bbox]
            # Red box for weapons, orange for others
            color = (0, 0, 255) if event.severity.value == "critical" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"{event.event_type} ({event.confidence:.0%})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        filename = (f"{event.module}_{event.camera_id}_"
                    f"f{event.frame_idx}_{event.event_type}.jpg")
        path = self.snapshots_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def start_clip_capture(
        self,
        event: AnalyticsEvent,
        post_seconds: float = 2.0,
        frame_size: tuple[int, int] = (1920, 1080),
    ) -> str:
        """
        Start capturing a short video clip.

        Saves the pre-buffer frames immediately and commits to
        capturing post_seconds more frames.

        Returns:
            Path where the clip will be saved.
        """
        filename = (f"{event.module}_{event.camera_id}_"
                    f"f{event.frame_idx}_{event.event_type}.mp4")
        clip_path = self.clips_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(clip_path), fourcc, self.fps, frame_size
        )

        # Write buffered (pre-detection) frames
        for buffered_frame in self._frame_buffer:
            if buffered_frame.shape[1] == frame_size[0]:
                writer.write(buffered_frame)

        post_frames = int(post_seconds * self.fps)
        self._post_capture[str(clip_path)] = {
            "writer": writer,
            "remaining": post_frames,
            "path": str(clip_path),
        }

        return str(clip_path)

    def feed_post_frame(self, frame: np.ndarray) -> None:
        """
        Feed a frame to any active clip captures.
        Call this every frame after start_clip_capture().
        """
        completed = []
        for clip_path, state in self._post_capture.items():
            if state["remaining"] > 0:
                state["writer"].write(frame)
                state["remaining"] -= 1
            else:
                state["writer"].release()
                completed.append(clip_path)
                print(f"[SnapshotSaver] Clip saved: {clip_path}")

        for path in completed:
            del self._post_capture[path]

    def shutdown(self):
        """Finalize and release any open clip writers."""
        for state in self._post_capture.values():
            state["writer"].release()
        self._post_capture.clear()
        print(f"[SnapshotSaver] Shut down")

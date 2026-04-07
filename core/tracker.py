"""
Standalone ByteTrack tracker using the supervision library.

Separates tracking from detection so each can be configured
and swapped independently.
"""

import numpy as np
import supervision as sv


class ByteTrackTracker:

    def __init__(self, track_activation_threshold=0.25,
                 lost_track_buffer=30,
                 minimum_matching_threshold=0.8,
                 frame_rate=30):
        """
        Args:
            track_activation_threshold: Minimum confidence to activate a new track.
            lost_track_buffer: Frames to keep a lost track before removing it.
            minimum_matching_threshold: IoU threshold for matching detections to tracks.
            frame_rate: Video frame rate (used internally by ByteTrack).
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections_array):
        """
        Update tracker with new detections.

        Args:
            detections_array: numpy array of shape (N, 6) from PersonDetector.
                              Each row: [x1, y1, x2, y2, confidence, class_id].

        Returns:
            List of dicts, each with:
                track_id (int), bbox [x1,y1,x2,y2], center (cx,cy), confidence (float).
        """
        if len(detections_array) == 0:
            return []

        sv_detections = sv.Detections(
            xyxy=detections_array[:, :4],
            confidence=detections_array[:, 4],
            class_id=detections_array[:, 5].astype(int),
        )

        tracked = self.tracker.update_with_detections(sv_detections)

        results = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            track_id = int(tracked.tracker_id[i])
            conf = float(tracked.confidence[i])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            results.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "confidence": conf,
            })

        return results

    def reset(self):
        """Reset all tracks."""
        self.tracker.reset()
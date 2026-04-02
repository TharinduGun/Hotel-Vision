"""
Lightweight IoU Tracker
========================
Simple IoU-based multi-object tracker for use with SAHI detections.
Assigns persistent integer IDs to bounding boxes across frames.

This replaces ByteTrack when using SAHI (which only does detection,
not tracking). It's lightweight and optimized for crowd scenarios.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class _Track:
    """Internal track state."""
    track_id: int
    bbox: list[float]       # [x1, y1, x2, y2]
    cls: int
    age: int = 0            # frames since creation
    lost: int = 0           # consecutive frames not matched
    hits: int = 1           # total matched frames


class IoUTracker:
    """
    Simple IoU-based tracker.

    Matches new detections to existing tracks by IoU overlap.
    Assigns new IDs for unmatched detections.
    Removes tracks after max_lost consecutive missed frames.

    Args:
        iou_threshold: Minimum IoU to consider a match (0.0 – 1.0).
        max_lost: Remove track after this many consecutive missed frames.
        min_hits: Track must be matched this many times before being reported.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_lost: int = 30,
        min_hits: int = 1,
    ):
        self._iou_thresh = iou_threshold
        self._max_lost = max_lost
        self._min_hits = min_hits
        self._next_id: int = 1
        self._tracks: list[_Track] = []

    def update(
        self,
        detections: list[dict],
    ) -> dict[int, dict]:
        """
        Match detections to tracks and return active tracked objects.

        Args:
            detections: List of {"bbox": [x1,y1,x2,y2], "cls": int}

        Returns:
            { track_id: {"bbox": [...], "cls": int} }
        """
        if not detections:
            # Age all tracks
            for t in self._tracks:
                t.lost += 1
            self._tracks = [t for t in self._tracks if t.lost <= self._max_lost]
            return {}

        det_bboxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        det_classes = [d["cls"] for d in detections]

        if not self._tracks:
            # No existing tracks — create new ones for all detections
            for i, det in enumerate(detections):
                self._tracks.append(_Track(
                    track_id=self._next_id,
                    bbox=det["bbox"],
                    cls=det["cls"],
                ))
                self._next_id += 1
        else:
            # Compute IoU matrix: tracks × detections
            track_bboxes = np.array([t.bbox for t in self._tracks], dtype=np.float32)
            iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)

            # Greedy matching (highest IoU first)
            matched_tracks = set()
            matched_dets = set()

            # Sort all IoU pairs by descending value
            n_tracks, n_dets = iou_matrix.shape
            pairs = []
            for ti in range(n_tracks):
                for di in range(n_dets):
                    if iou_matrix[ti, di] >= self._iou_thresh:
                        pairs.append((iou_matrix[ti, di], ti, di))
            pairs.sort(reverse=True)

            for iou_val, ti, di in pairs:
                if ti in matched_tracks or di in matched_dets:
                    continue
                # Match track ti ← detection di
                self._tracks[ti].bbox = detections[di]["bbox"]
                self._tracks[ti].cls = detections[di]["cls"]
                self._tracks[ti].lost = 0
                self._tracks[ti].hits += 1
                self._tracks[ti].age += 1
                matched_tracks.add(ti)
                matched_dets.add(di)

            # Unmatched tracks → increment lost
            for ti in range(n_tracks):
                if ti not in matched_tracks:
                    self._tracks[ti].lost += 1
                    self._tracks[ti].age += 1

            # Unmatched detections → create new tracks
            for di in range(n_dets):
                if di not in matched_dets:
                    self._tracks.append(_Track(
                        track_id=self._next_id,
                        bbox=detections[di]["bbox"],
                        cls=detections[di]["cls"],
                    ))
                    self._next_id += 1

        # Prune lost tracks
        self._tracks = [t for t in self._tracks if t.lost <= self._max_lost]

        # Return active tracks (that meet minimum hit threshold)
        result = {}
        for t in self._tracks:
            if t.lost == 0 and t.hits >= self._min_hits:
                result[t.track_id] = {"bbox": t.bbox, "cls": t.cls}
        return result

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: np.ndarray,
        boxes_b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute IoU matrix between two sets of bboxes.

        Args:
            boxes_a: (N, 4) array of [x1, y1, x2, y2]
            boxes_b: (M, 4) array of [x1, y1, x2, y2]

        Returns:
            (N, M) IoU matrix
        """
        n = boxes_a.shape[0]
        m = boxes_b.shape[0]
        iou = np.zeros((n, m), dtype=np.float32)

        for i in range(n):
            ax1, ay1, ax2, ay2 = boxes_a[i]
            a_area = (ax2 - ax1) * (ay2 - ay1)

            for j in range(m):
                bx1, by1, bx2, by2 = boxes_b[j]
                b_area = (bx2 - bx1) * (by2 - by1)

                # Intersection
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter = (ix2 - ix1) * (iy2 - iy1)
                union = a_area + b_area - inter
                iou[i, j] = inter / union if union > 0 else 0.0

        return iou

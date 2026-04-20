"""
Shared utilities for the cash detection module.
Eliminates duplicate helper functions across sub-components.
"""


def compute_iou(box1, box2) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2] bounding box.
        box2: [x1, y1, x2, y2] bounding box.

    Returns:
        float: IoU value in [0.0, 1.0].
    """
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

"""
Matches a face bounding box to the closest person track by checking
if the face bbox falls inside a person's body bbox.
"""


def match_face_to_track(face_bbox, detections):
    """
    Find which tracked person a detected face belongs to.

    Args:
        face_bbox: [x1, y1, x2, y2] from InsightFace.
        detections: List of detection dicts from PersonDetector/ByteTrack,
                    each with 'track_id' and 'bbox' keys.

    Returns:
        track_id (int) of the matching person, or None if no match.
    """
    fx1, fy1, fx2, fy2 = map(int, face_bbox)

    best_track_id = None
    best_overlap = 0

    for det in detections:
        px1, py1, px2, py2 = det["bbox"]

        if fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2:
            overlap_area = (fx2 - fx1) * (fy2 - fy1)
            if overlap_area > best_overlap:
                best_overlap = overlap_area
                best_track_id = det["track_id"]

    return best_track_id
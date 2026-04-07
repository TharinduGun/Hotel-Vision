import cv2
import numpy as np   # ← was wrongly placed at the bottom of the file

CLASS_LABELS = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


def draw(frame, detections: list):

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid   = det.get("track_id", -1)
        label = CLASS_LABELS.get(det.get("class_id", -1), "Vehicle")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ID:{tid}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    return frame


def draw_spaces(frame, space_counts: dict, spaces: dict):

    for name, info in space_counts.items():
        poly = spaces.get(name)
        if not poly:
            continue

        color   = (0, 0, 255) if info["occupied"] else (0, 255, 0)
        pts     = np.array(poly, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2)

        cx    = sum(p[0] for p in poly) // len(poly)
        cy    = sum(p[1] for p in poly) // len(poly)
        label = "OCC" if info["occupied"] else "FREE"
        cv2.putText(frame, f"{name}:{label}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame


def draw_occupancy_overlay(frame, analytics: dict):

    if not analytics:
        return frame

    lines = [
        f"Occupied : {analytics.get('occupied', 0)} / {analytics.get('capacity', 0)}",
        f"Available: {analytics.get('available', 0)}",
        f"Status   : {analytics.get('status', '-').upper()}",
        f"Load     : {analytics.get('occupancy_pct', 0.0)}%",
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame
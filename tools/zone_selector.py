"""
Interactive zone coordinate selector.

Opens a video frame and lets you click polygon vertices.
Press ESC when done — coordinates are saved to a JSON file.

Usage:
    python tools/zone_selector.py
    python tools/zone_selector.py --video data/videos/video.mp4 --output configs/zone_points.json
"""

import argparse
import cv2
import json

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"  Point {len(points)}: [{x}, {y}]")


def main():
    parser = argparse.ArgumentParser(description="Interactive zone coordinate selector")
    parser.add_argument("--video", type=str, default="data/videos/coffee_shop2.mp4",
                        help="Video file to extract a frame from")
    parser.add_argument("--output", type=str, default="zone_points.json",
                        help="Output JSON file for the polygon coordinates")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame from {args.video}")
        return

    print("Click points to define a zone polygon. Press ESC when done.")

    cv2.namedWindow("Zone Selector")
    cv2.setMouseCallback("Zone Selector", mouse_callback)

    while True:
        display = frame.copy()

        for i, p in enumerate(points):
            cv2.circle(display, tuple(p), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display, tuple(points[i - 1]), tuple(p), (0, 255, 0), 2)

        if len(points) > 2:
            cv2.line(display, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 1)

        cv2.imshow("Zone Selector", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    print(f"\nPolygon points ({len(points)} vertices): {points}")

    with open(args.output, "w") as f:
        json.dump(points, f, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
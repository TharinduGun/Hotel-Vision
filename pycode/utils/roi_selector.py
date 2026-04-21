"""
ROI Selector Tool
==================
Opens a reference image and lets the user draw rectangular ROIs
using OpenCV's selectROI. Saves all zone coordinates to config/zones.json.

Usage:
    python roi_selector.py                          # uses default reference image
    python roi_selector.py --image path/to/img.png  # custom image
    python roi_selector.py --from-video path/to/v.mp4  # extract first frame
"""

import cv2
import json
import os
import argparse
import sys

# Default reference image path (relative to this file)
DEFAULT_REF_IMAGE = os.path.join(os.path.dirname(__file__), "refernce_image.png")

# Default output config path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "zones.json")

# Zones to select — edit this list to add/remove zones
ZONES_TO_SELECT = [
    {"name": "Cashier 1", "type": "cashier"},
    {"name": "Cashier 2", "type": "cashier"},
    {"name": "Money Exchange Counter", "type": "money_exchange"},
]


def extract_first_frame(video_path):
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    return frame


def select_rois(image, zones_config):
    """
    Opens a window for each zone and lets the user draw a rectangle.
    
    Args:
        image: BGR numpy array (the reference image).
        zones_config: List of dicts with 'name' and 'type' keys.
    
    Returns:
        List of zone dicts with 'name', 'type', and 'roi' (x, y, w, h).
    """
    selected_zones = []
    overlay = image.copy()

    # Color map for different zone types
    color_map = {
        "cashier": (0, 255, 0),       # Green
        "money_exchange": (0, 165, 255),  # Orange
    }
    default_color = (255, 255, 255)  # White fallback

    for i, zone in enumerate(zones_config):
        zone_name = zone["name"]
        zone_type = zone["type"]
        color = color_map.get(zone_type, default_color)

        # Draw previously selected zones on the display image
        display = overlay.copy()
        for prev in selected_zones:
            px, py, pw, ph = prev["roi"]
            cv2.rectangle(display, (px, py), (px + pw, py + ph), 
                          color_map.get(prev["type"], default_color), 2)
            cv2.putText(display, prev["name"], (px, py - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        color_map.get(prev["type"], default_color), 2)

        window_title = f"Select ROI for: {zone_name} ({i+1}/{len(zones_config)}) | ENTER=confirm, ESC=skip"
        print(f"\n>>> Draw rectangle for: {zone_name}")
        print("    Press ENTER/SPACE to confirm, ESC to skip this zone, C to cancel all.")

        roi = cv2.selectROI(window_title, display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_title)

        x, y, w, h = roi
        if w > 0 and h > 0:
            selected_zones.append({
                "name": zone_name,
                "type": zone_type,
                "roi": [int(x), int(y), int(w), int(h)]
            })
            print(f"    ✓ {zone_name}: x={x}, y={y}, w={w}, h={h}")

            # Update overlay with the new selection
            cv2.rectangle(overlay, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            cv2.putText(overlay, zone_name, (int(x), int(y) - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            print(f"    ✗ {zone_name}: Skipped (no selection)")

    return selected_zones


def show_preview(image, zones):
    """Show a final preview of all selected zones."""
    preview = image.copy()
    
    color_map = {
        "cashier": (0, 255, 0),
        "money_exchange": (0, 165, 255),
    }

    for zone in zones:
        x, y, w, h = zone["roi"]
        color = color_map.get(zone["type"], (255, 255, 255))
        
        # Semi-transparent fill
        sub_img = preview[y:y+h, x:x+w]
        overlay_color = [color[0], color[1], color[2]]
        rect_overlay = sub_img.copy()
        rect_overlay[:] = overlay_color
        cv2.addWeighted(rect_overlay, 0.2, sub_img, 0.8, 0, sub_img)
        
        # Border and label
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        
        label = f"{zone['name']} ({zone['type']})"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(preview, (x, y - t_size[1] - 10), (x + t_size[0] + 4, y), color, -1)
        cv2.putText(preview, label, (x + 2, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("ROI Preview — Press any key to save, ESC to cancel", preview)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return key != 27  # True = save, False = cancelled


def save_config(zones, image_path, image_shape, output_path):
    """Save zone configuration to JSON."""
    h, w = image_shape[:2]
    config = {
        "source_image": os.path.basename(image_path),
        "image_size": [w, h],
        "zones": zones
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Configuration saved to: {output_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Interactive ROI Selector for Video Analytics")
    parser.add_argument("--image", type=str, default=DEFAULT_REF_IMAGE,
                        help="Path to reference image")
    parser.add_argument("--from-video", type=str, default=None,
                        help="Extract first frame from this video as reference")
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Output path for zones.json config")
    args = parser.parse_args()

    # Load image
    if args.from_video:
        print(f"Extracting first frame from: {args.from_video}")
        image = extract_first_frame(args.from_video)
    else:
        image_path = os.path.abspath(args.image)
        if not os.path.exists(image_path):
            print(f"ERROR: Reference image not found: {image_path}")
            sys.exit(1)
        print(f"Loading reference image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Could not read image: {image_path}")
            sys.exit(1)

    image_path = args.from_video or args.image
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"\nZones to define: {len(ZONES_TO_SELECT)}")
    for z in ZONES_TO_SELECT:
        print(f"  - {z['name']} ({z['type']})")

    # Select ROIs
    zones = select_rois(image, ZONES_TO_SELECT)

    if not zones:
        print("\nNo zones selected. Exiting without saving.")
        sys.exit(0)

    # Preview
    print(f"\n{len(zones)} zone(s) selected. Showing preview...")
    confirmed = show_preview(image, zones)

    if confirmed:
        output_path = os.path.abspath(args.output)
        save_config(zones, image_path, image.shape, output_path)
    else:
        print("\nCancelled. No configuration saved.")


if __name__ == "__main__":
    main()

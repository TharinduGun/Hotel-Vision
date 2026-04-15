"""
Cash Detection Diagnostic Tool
================================
Runs the cash detection model on a video and saves detailed diagnostic output
so you can see EXACTLY what the model thinks is "cash" and why.

Outputs:
  1. Annotated frames (every N-th frame) with ALL detections drawn
  2. Cropped images of every detection, organized by confidence range
  3. CSV log with per-detection metadata for analysis
  4. Summary stats printed at the end

Usage:
  python pycode/scripts/diagnose_cash_detections.py
  python pycode/scripts/diagnose_cash_detections.py --video path/to/video.mp4
  python pycode/scripts/diagnose_cash_detections.py --conf 0.20 --max-frames 500
"""

import os
import sys
import csv
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose cash detection false positives")
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "resources" / "videos" / "Indoor_Original_VideoStream.mp4"),
        help="Path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "pycode" / "models" / "cash_detector_v2" / "train_v2" / "weights" / "best.pt"),
        help="Path to cash detection model",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold (use LOW to catch all detections, default: 0.15)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Max frames to process (default: 1000 = ~40s at 25fps)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every N-th frame (default: 5)",
    )
    parser.add_argument(
        "--save-annotated-every",
        type=int,
        default=25,
        help="Save annotated frame every N processed frames (default: 25)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "pycode" / "diagnostics" / "cash_detection"),
        help="Output directory for diagnostic results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    return parser.parse_args()


def setup_output_dirs(output_dir: str) -> dict:
    """Create organized output directory structure."""
    dirs = {
        "root": Path(output_dir),
        "annotated": Path(output_dir) / "annotated_frames",
        "crops_high": Path(output_dir) / "crops" / "conf_0.70_plus",
        "crops_medium": Path(output_dir) / "crops" / "conf_0.50_to_0.70",
        "crops_low": Path(output_dir) / "crops" / "conf_0.30_to_0.50",
        "crops_very_low": Path(output_dir) / "crops" / "conf_below_0.30",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_crop_dir(dirs: dict, confidence: float) -> Path:
    """Get the right crop directory based on confidence value."""
    if confidence >= 0.70:
        return dirs["crops_high"]
    elif confidence >= 0.50:
        return dirs["crops_medium"]
    elif confidence >= 0.30:
        return dirs["crops_low"]
    else:
        return dirs["crops_very_low"]


def draw_detection(frame, bbox, confidence, det_idx, color=(0, 255, 0)):
    """Draw a detection on the frame with confidence label."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Background for text
    label = f"#{det_idx} conf={confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw center point
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    return frame


def compute_geometric_pass(bbox, frame_w, frame_h,
                           min_area=800, max_area_ratio=0.05,
                           min_aspect=1.0, max_aspect=8.0):
    """Check if detection passes geometric filters (Phase 0A values)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h
    frame_area = frame_w * frame_h

    if area < min_area:
        return False, f"too_small (area={area:.0f} < {min_area})"
    if area / frame_area > max_area_ratio:
        return False, f"too_large (ratio={area/frame_area:.3f} > {max_area_ratio})"

    long_side = max(w, h)
    short_side = max(min(w, h), 1)
    aspect = long_side / short_side

    if aspect < min_aspect or aspect > max_aspect:
        return False, f"bad_aspect (aspect={aspect:.1f})"

    return True, "pass"


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    # Setup
    dirs = setup_output_dirs(args.output_dir)
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 70)
    print("CASH DETECTION DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"Video:        {args.video}")
    print(f"Model:        {args.model}")
    print(f"Resolution:   {frame_w}x{frame_h} @ {fps:.1f}fps")
    print(f"Total frames: {total_frames}")
    print(f"Conf threshold: {args.conf} (intentionally low to catch all detections)")
    print(f"Processing:   every {args.frame_skip}-th frame, max {args.max_frames} frames")
    print(f"Output:       {args.output_dir}")
    print("=" * 70)

    # CSV log
    csv_path = dirs["root"] / "detections_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "det_idx", "x1", "y1", "x2", "y2",
        "width", "height", "area", "aspect_ratio",
        "confidence", "class_id", "class_name",
        "geometric_pass", "geometric_reason",
        "center_x", "center_y",
        "frame_region",  # top/middle/bottom thirds
    ])

    # Stats
    stats = {
        "frames_processed": 0,
        "total_detections": 0,
        "detections_by_conf": defaultdict(int),
        "geometric_pass": 0,
        "geometric_fail": 0,
        "fail_reasons": defaultdict(int),
        "crops_saved": 0,
        "annotated_saved": 0,
    }

    frame_idx = 0
    processed = 0
    det_global_idx = 0

    print(f"\nProcessing...")

    while processed < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed
        if frame_idx % args.frame_skip != 0:
            continue

        processed += 1
        stats["frames_processed"] += 1

        # Run YOLO at LOW confidence to catch everything
        results = model.predict(
            source=frame,
            conf=args.conf,
            classes=[0],  # Class 0 only (whatever the model calls it)
            device=args.device,
            verbose=False,
            imgsz=640,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            # No detections on this frame
            if processed % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames} ({processed} processed) — 0 detections")
            continue

        boxes = results[0].boxes
        num_dets = len(boxes)
        stats["total_detections"] += num_dets

        # Should we save an annotated frame?
        save_annotated = (processed % args.save_annotated_every == 0) or num_dets >= 3
        annotated_frame = frame.copy() if save_annotated else None

        for i in range(num_dets):
            det_global_idx += 1
            bbox = boxes.xyxy[i].cpu().tolist()
            conf = float(boxes.conf[i].cpu())
            cls_id = int(boxes.cls[i].cpu())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")

            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            area = w * h
            long_s = max(w, h)
            short_s = max(min(w, h), 1)
            aspect = long_s / short_s
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Confidence bucket
            if conf >= 0.70:
                stats["detections_by_conf"]["0.70+"] += 1
            elif conf >= 0.50:
                stats["detections_by_conf"]["0.50-0.70"] += 1
            elif conf >= 0.30:
                stats["detections_by_conf"]["0.30-0.50"] += 1
            else:
                stats["detections_by_conf"]["<0.30"] += 1

            # Geometric filter check
            geo_pass, geo_reason = compute_geometric_pass(
                bbox, frame_w, frame_h
            )
            if geo_pass:
                stats["geometric_pass"] += 1
            else:
                stats["geometric_fail"] += 1
                stats["fail_reasons"][geo_reason.split(" ")[0]] += 1

            # Frame region (vertical thirds)
            if cy < frame_h / 3:
                region = "top"
            elif cy < 2 * frame_h / 3:
                region = "middle"
            else:
                region = "bottom"

            # CSV log
            csv_writer.writerow([
                frame_idx, det_global_idx,
                int(x1), int(y1), int(x2), int(y2),
                int(w), int(h), int(area), f"{aspect:.2f}",
                f"{conf:.4f}", cls_id, cls_name,
                "PASS" if geo_pass else "FAIL", geo_reason,
                int(cx), int(cy),
                region,
            ])

            # Save crop
            crop_x1 = max(0, int(x1) - 15)
            crop_y1 = max(0, int(y1) - 15)
            crop_x2 = min(frame_w, int(x2) + 15)
            crop_y2 = min(frame_h, int(y2) + 15)
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            if crop.size > 0:
                crop_dir = get_crop_dir(dirs, conf)
                crop_name = f"frame{frame_idx:06d}_det{det_global_idx:04d}_conf{conf:.2f}.jpg"
                cv2.imwrite(str(crop_dir / crop_name), crop)
                stats["crops_saved"] += 1

            # Draw on annotated frame
            if annotated_frame is not None:
                color = (0, 255, 0) if geo_pass else (0, 0, 255)  # Green=pass, Red=fail
                draw_detection(annotated_frame, bbox, conf, det_global_idx, color)

        # Save annotated frame
        if annotated_frame is not None:
            ann_name = f"frame_{frame_idx:06d}_{num_dets}dets.jpg"
            cv2.imwrite(str(dirs["annotated"] / ann_name), annotated_frame)
            stats["annotated_saved"] += 1

        if processed % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({processed} processed) — "
                  f"{stats['total_detections']} total detections so far")

    # Cleanup
    cap.release()
    csv_file.close()

    # Print summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Frames processed:     {stats['frames_processed']}")
    print(f"Total detections:     {stats['total_detections']}")
    if stats['frames_processed'] > 0:
        dets_per_frame = stats['total_detections'] / stats['frames_processed']
        print(f"Avg detections/frame: {dets_per_frame:.2f}")
    print()

    print("Detections by confidence range:")
    for bucket in ["0.70+", "0.50-0.70", "0.30-0.50", "<0.30"]:
        count = stats["detections_by_conf"].get(bucket, 0)
        pct = (count / stats["total_detections"] * 100) if stats["total_detections"] > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>10s}: {count:5d} ({pct:5.1f}%) {bar}")
    print()

    print("Geometric filter results:")
    print(f"  PASS: {stats['geometric_pass']}")
    print(f"  FAIL: {stats['geometric_fail']}")
    if stats["fail_reasons"]:
        print("  Failure breakdown:")
        for reason, count in sorted(stats["fail_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    print()

    print(f"Outputs saved to: {dirs['root']}")
    print(f"  Annotated frames: {stats['annotated_saved']}")
    print(f"  Detection crops:  {stats['crops_saved']}")
    print(f"  CSV log:          {csv_path}")
    print()

    print("NEXT STEPS:")
    print("  1. Open the 'crops/' folders to see what the model thinks is 'cash'")
    print("  2. Check 'annotated_frames/' to see detections in context")
    print("  3. Open 'detections_log.csv' in Excel to filter/sort by confidence")
    print("  4. Crops in 'conf_0.70_plus/' are what currently passes the pipeline")
    print("  5. Use these insights to build proper negatives for the v3 dataset")
    print("=" * 70)


if __name__ == "__main__":
    main()

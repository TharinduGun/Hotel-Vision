"""
Collect Hard Negatives for Cash Detection (Source 3)
=====================================================
The easiest approach: extract frames from the hotel CCTV video
and add them to the training set WITHOUT labels.

YOLO treats images with no corresponding .txt label file as
"background" — meaning "nothing to detect here." This teaches
the model that counter surfaces, hands, phones, screens, etc.
are NOT cash.

Your 30-minute hotel video is the perfect negative source because
it contains all the things that cause false positives:
  - Hands at the counter (no cash)
  - Phones, cards, papers on counter
  - Monitor screens
  - Empty counter surfaces
  - People walking around

Usage:
  python pycode/scripts/collect_hard_negatives.py

  # Fewer/more negatives:
  python pycode/scripts/collect_hard_negatives.py --count 500
  
  # Use a different video:
  python pycode/scripts/collect_hard_negatives.py --video path/to/video.mp4
"""

import os
import sys
import random
import argparse
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hard negatives from hotel video")
    parser.add_argument(
        "--video",
        type=str,
        default=str(PROJECT_ROOT / "resources" / "videos" / "Indoor_Original_VideoStream.mp4"),
        help="Path to CCTV video (default: hotel lobby video)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=600,
        help="Number of negative frames to extract (default: 600)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "resources" / "datasets" / "cash-detection-v3" / "train" / "images"),
        help="Output directory (default: cash-detection-v3 train images)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("HARD NEGATIVE COLLECTOR")
    print("=" * 70)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video:          {args.video}")
    print(f"Total frames:   {total_frames} ({total_frames/fps:.0f}s)")
    print(f"Extracting:     {args.count} random frames as hard negatives")
    print(f"Output:         {args.output_dir}")
    print()

    # Pick random frame indices (spread across the full video)
    random.seed(args.seed)
    frame_indices = sorted(random.sample(
        range(0, total_frames),
        min(args.count, total_frames)
    ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check what label dir would be (to make sure we DON'T create labels)
    label_dir = output_dir.parent / "labels"

    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save image with "neg_" prefix (easy to identify later)
        img_name = f"neg_hotel_{saved:05d}.jpg"
        img_path = output_dir / img_name

        cv2.imwrite(str(img_path), frame)

        # CRITICAL: Do NOT create a label file for this image.
        # When YOLO encounters an image with no matching .txt label,
        # it treats it as a "background" image — no objects to detect.
        # This is exactly what we want: teach the model these scenes are NOT cash.

        # Double-check: remove any accidental label file
        label_path = label_dir / f"neg_hotel_{saved:05d}.txt"
        if label_path.exists():
            label_path.unlink()

        saved += 1
        if saved % 100 == 0:
            print(f"  Saved {saved}/{args.count} negatives...")

    cap.release()

    print(f"\nDone! Saved {saved} hard negative frames.")
    print(f"  Location: {output_dir}")
    print(f"  Prefix:   neg_hotel_*")
    print()
    print("HOW THIS WORKS:")
    print("  - These images have NO label files (.txt)")
    print("  - YOLO sees them as 'background' during training")
    print("  - The model learns: hands, phones, counters, screens = NOT cash")
    print()
    print("WHAT'S IN THESE NEGATIVES:")
    print("  - Hotel counter surfaces (the #1 FP source)")
    print("  - Hands at counter without cash")
    print("  - Phones, cards, papers on counter")
    print("  - Monitor/screen reflections")
    print("  - People walking in lobby")
    print("  - Empty scenes")
    print()
    print("You can now train: python pycode/scripts/train_cash_v3.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

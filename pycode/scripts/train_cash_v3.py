"""
Train Cash Detection Model v3
==============================
Trains a YOLOv8m model from scratch (COCO pretrained) on the new
cash-detection-v3 dataset.

Dataset composition:
  - 3088 public images (merged from 4 Roboflow datasets)
  - 800 synthetic CCTV images (cut-paste banknotes on hotel backgrounds)
  - 600 hard negatives (hotel video frames, no labels)
  - 579 validation images
  - 193 test images

Usage:
  python pycode/scripts/train_cash_v3.py

  # Resume from checkpoint:
  python pycode/scripts/train_cash_v3.py --resume
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train cash detection model v3")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs (default: 80)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce to 8 if OOM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (default: 0 = first GPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = PROJECT_ROOT / "resources" / "datasets" / "cash-detection-v3" / "data.yaml"
    output_dir = PROJECT_ROOT / "pycode" / "models" / "cash_detector_v3"

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run merge_cash_datasets.py and create_synthetic_cash_data.py first!")
        sys.exit(1)

    print("=" * 70)
    print("CASH DETECTION MODEL v3 — TRAINING")
    print("=" * 70)
    print(f"Dataset:  {dataset_path}")
    print(f"Output:   {output_dir}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}")
    print(f"ImgSize:  {args.imgsz}")
    print(f"Device:   {args.device}")
    print(f"Resume:   {args.resume}")
    print("=" * 70)

    if args.resume:
        # Resume from last checkpoint — YOLO reads epoch + all args from the checkpoint
        last_pt = output_dir / "train" / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"ERROR: No checkpoint found at {last_pt}")
            sys.exit(1)
        model = YOLO(str(last_pt))
        print(f"Resuming from: {last_pt}")
        print("\nResuming training...\n")
        model.train(resume=True)
    else:
        # Fresh start from COCO-pretrained YOLOv8m
        model = YOLO("yolov8m.pt")
        print("Starting from COCO-pretrained YOLOv8m")
        print("\nStarting training...\n")

        model.train(
            data=str(dataset_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=15,

            # Project output
            project=str(output_dir),
            name="train",
            exist_ok=True,

            # Optimizer
            optimizer="auto",
            lr0=0.01,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3.0,

            # ── CCTV-optimized augmentation ──
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.5,
            degrees=15.0,
            translate=0.15,
            scale=0.7,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.3,
            erasing=0.3,

            # Training config
            close_mosaic=10,
            amp=True,
            workers=0,
            deterministic=True,
            seed=42,
            plots=True,
            verbose=True,
            val=True,
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best weights:  {output_dir / 'train' / 'weights' / 'best.pt'}")
    print(f"Last weights:  {output_dir / 'train' / 'weights' / 'last.pt'}")
    print()
    print("NEXT STEPS:")
    print("  1. Check results: open pycode/models/cash_detector_v3/train/results.png")
    print("  2. Check confusion matrix: open pycode/models/cash_detector_v3/train/confusion_matrix.png")
    print("  3. Test on hotel video: update system_config.yaml model_path")
    print(f'     model_path: "pycode/models/cash_detector_v3/train/weights/best.pt"')
    print("  4. Run full pipeline to verify FP reduction")
    print("=" * 70)


if __name__ == "__main__":
    main()

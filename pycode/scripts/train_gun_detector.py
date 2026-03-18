"""
Train Gun Detection Model
============================
Fine-tunes YOLOv8 on a weapon detection dataset for the
gun detection module.

Usage:
    cd d:/Work/jwinfotech/Videoanalystics/video-analytics
    python pycode/scripts/train_gun_detector.py

Output:
    Trained weights → pycode/models/gun_detector/weights/best.pt

Dataset:
    Download a weapon detection dataset from Roboflow and place it in:
    resources/datasets/weapon-detection/

    Recommended dataset options:
    1. WeSecure (Roboflow, ~7.8k images — Handgun, Shotgun, Knife, Rifle)
       https://universe.roboflow.com/wesecure/weapon-detection-rlvhb
    2. Hand Weapon Dataset (HuggingFace, 6k images)

    After downloading, the structure should be:
    resources/datasets/weapon-detection/
        data.yaml
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
"""

import os
import sys
import argparse
import torch
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIG — edit these as needed
# ──────────────────────────────────────────────

# Paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Base model — transfer learning from our existing YOLOv8m
BASE_MODEL = os.path.join(PROJECT_ROOT, "pycode", "src", "yolov8m.pt")

# Checkpoint from previous interrupted training
LAST_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "pycode", "models", "gun_detector", "weights", "last.pt"
)

# Dataset config (update this after downloading your dataset)
DATASET_YAML = os.path.join(
    PROJECT_ROOT, "resources", "datasets", "weapon-detection", "data.yaml"
)

# Training output directory
OUTPUT_PROJECT = os.path.join(PROJECT_ROOT, "pycode", "models")
OUTPUT_NAME = "gun_detector"

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────

EPOCHS = 80             # More epochs for weapon detection (harder task)
IMG_SIZE = 640          # Image size
BATCH_SIZE = 8          # 8 is safe for 16GB RAM; use 4 if OOM
PATIENCE = 15           # Early stopping — stop if no improvement for 15 epochs
WORKERS = 0             # 0 = main process only (avoids Windows multiprocessing errors)
DEVICE = "0"            # "0" for GPU, "cpu" for CPU
CONF_THRESHOLD = 0.25   # Confidence threshold for validation


def main():
    # ── 0. Parse CLI args ───────────────────────
    parser = argparse.ArgumentParser(description="Train Gun Detection Model")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last.pt checkpoint (use after a crash/power failure)"
    )
    args = parser.parse_args()

    # ── 1. Diagnostics ──────────────────────────
    print("=" * 60)
    print("  GUN DETECTOR — Training Script")
    print("=" * 60)
    print(f"  CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU found. Training on CPU will be very slow!")
    print(f"  Mode           : {'RESUME from checkpoint' if args.resume else 'FRESH training'}")
    print(f"  Base Model     : {BASE_MODEL}")
    print(f"  Dataset        : {DATASET_YAML}")
    print(f"  Output         : {OUTPUT_PROJECT}/{OUTPUT_NAME}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Batch Size     : {BATCH_SIZE}")
    print(f"  Image Size     : {IMG_SIZE}")
    print("=" * 60)

    # ── 2. Handle resume mode ──────────────────
    if args.resume:
        if not os.path.exists(LAST_CHECKPOINT):
            print(f"\nERROR: No checkpoint found at: {LAST_CHECKPOINT}")
            print("Cannot resume — run without --resume to start fresh.")
            sys.exit(1)

        print(f"\n🔄 RESUMING from: {LAST_CHECKPOINT}")
        model = YOLO(LAST_CHECKPOINT)
        model.train(resume=True)

        best_weights = os.path.join(OUTPUT_PROJECT, OUTPUT_NAME, "weights", "best.pt")
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE (resumed)")
        print("=" * 60)
        if os.path.exists(best_weights):
            size_mb = os.path.getsize(best_weights) / (1024 * 1024)
            print(f"  Best weights   : {best_weights}")
            print(f"  Model size     : {size_mb:.1f} MB")
        print("=" * 60)
        return

    # ── 3. Validate paths (fresh training) ─────
    if not os.path.exists(BASE_MODEL):
        print(f"\nERROR: Base model not found: {BASE_MODEL}")
        print("Make sure yolov8m.pt exists in pycode/src/")
        sys.exit(1)

    if not os.path.exists(DATASET_YAML):
        print(f"\nERROR: Dataset config not found: {DATASET_YAML}")
        print("Download a weapon detection dataset and place it in:")
        print(f"  {os.path.dirname(DATASET_YAML)}/")
        print("\nRecommended: Roboflow WeSecure dataset (~7.8k images)")
        print("  https://universe.roboflow.com/wesecure/weapon-detection-rlvhb")
        sys.exit(1)

    # Warn if a checkpoint exists
    if os.path.exists(LAST_CHECKPOINT):
        print("\n" + "⚠" * 30)
        print(f"  WARNING: A checkpoint exists at {LAST_CHECKPOINT}")
        print("  Starting fresh will OVERWRITE it.")
        print("  To resume instead, run: python train_gun_detector.py --resume")
        print("⚠" * 30 + "\n")

    os.makedirs(OUTPUT_PROJECT, exist_ok=True)

    # ── 4. Load model ───────────────────────────
    print("\nLoading base model...")
    model = YOLO(BASE_MODEL)

    # ── 5. Train ────────────────────────────────
    print("\nStarting training...\n")
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,         # Overwrite if re-training
        pretrained=True,       # Use pretrained weights
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        save=True,
        save_period=-1,        # Save only best & last
        val=True,
        plots=True,
        verbose=True,
        # Augmentation for weapon detection
        hsv_h=0.015,           # HSV hue augmentation range
        hsv_s=0.7,             # HSV saturation augmentation range
        hsv_v=0.4,             # HSV value augmentation range
        degrees=10.0,          # Random rotation ±10°
        translate=0.1,         # Random translation ±10%
        scale=0.5,             # Random scaling ±50%
        fliplr=0.5,            # Horizontal flip probability
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.15,            # Mixup augmentation (blend images)
    )

    # ── 6. Summary ──────────────────────────────
    best_weights = os.path.join(OUTPUT_PROJECT, OUTPUT_NAME, "weights", "best.pt")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    if os.path.exists(best_weights):
        size_mb = os.path.getsize(best_weights) / (1024 * 1024)
        print(f"  Best weights   : {best_weights}")
        print(f"  Model size     : {size_mb:.1f} MB")
    else:
        print("  WARNING: best.pt not found — check training logs above")
    print("=" * 60)

    print("\n✅ Next step: Run the validation script to check accuracy:")
    print(f"   python pycode/scripts/validate_gun_detector.py")
    print("\n  Then update system_config.yaml to point to the weights:")
    print(f"   model_path: {best_weights}")


if __name__ == "__main__":
    main()

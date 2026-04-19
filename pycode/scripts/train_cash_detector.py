"""
Train Cash Detection Model
============================
Fine-tunes YOLOv8m on a cash/banknote dataset for the
Cash Handling & Reception Monitoring feature.

Usage:
    cd d:/Work/jwinfotech/Videoanalystics/video-analytics
    python pycode/scripts/train_cash_detector.py

Output:
    Trained weights → pycode/models/cash_detector/weights/best.pt
"""

import os
import sys
import torch
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIG — edit these as needed
# ──────────────────────────────────────────────

# Paths (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Base model — transfer learning from our existing YOLOv8m
BASE_MODEL = os.path.join(PROJECT_ROOT, "pycode", "src", "yolov8m.pt")

# Dataset config
DATASET_YAML = os.path.join(
    PROJECT_ROOT, "resources", "datasets", "cash.v7i.yolov8", "data.yaml"
)

# Training output directory
OUTPUT_PROJECT = os.path.join(PROJECT_ROOT, "pycode", "models")
OUTPUT_NAME = "cash_detector"

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────

EPOCHS = 50           # Number of training epochs
IMG_SIZE = 640        # Image size (dataset is already 640x640)
BATCH_SIZE = 8         # 8 is safe for 16GB RAM systems; use 4 if still OOM
PATIENCE = 10         # Early stopping — stop if no improvement for 10 epochs
WORKERS = 0           # 0 = main process only (avoids Windows multiprocessing MemoryError)
DEVICE = "0"          # "0" for GPU, "cpu" for CPU (much slower)
CONF_THRESHOLD = 0.25 # Confidence threshold for validatiosn


def main():
    # ── 1. Diagnostics ──────────────────────────
    print("=" * 60)
    print("  CASH DETECTOR — Training Script")
    print("=" * 60)
    print(f"  CUDA Available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU found. Training on CPU will be very slow!")
        DEVICE_USED = "cpu"
    print(f"  Base Model     : {BASE_MODEL}")
    print(f"  Dataset        : {DATASET_YAML}")
    print(f"  Output         : {OUTPUT_PROJECT}/{OUTPUT_NAME}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Batch Size     : {BATCH_SIZE}")
    print(f"  Image Size     : {IMG_SIZE}")
    print("=" * 60)

    # ── 2. Validate paths ───────────────────────
    if not os.path.exists(BASE_MODEL):
        print(f"\nERROR: Base model not found: {BASE_MODEL}")
        print("Make sure yolov8m.pt exists in pycode/src/")
        sys.exit(1)

    if not os.path.exists(DATASET_YAML):
        print(f"\nERROR: Dataset config not found: {DATASET_YAML}")
        print("Make sure the Roboflow dataset is in resources/datasets/cash.v7i.yolov8/")
        sys.exit(1)

    os.makedirs(OUTPUT_PROJECT, exist_ok=True)

    # ── 3. Load model ───────────────────────────
    print("\nLoading base model...")
    model = YOLO(BASE_MODEL)

    # ── 4. Train ────────────────────────────────
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
        optimizer="auto",      # Let ultralytics pick the best optimizer
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate factor
        warmup_epochs=3,       # Warmup epochs
        cos_lr=True,           # Cosine learning rate schedule
        save=True,             # Save checkpoints
        save_period=-1,        # Save only best & last (not every epoch)
        val=True,              # Run validation after each epoch
        plots=True,            # Generate training plots
        verbose=True,
    )

    # ── 5. Summary ──────────────────────────────
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
    print(f"   python pycode/scripts/validate_cash_detector.py")


if __name__ == "__main__":
    main()

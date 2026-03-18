"""
Validate Gun Detection Model
===============================
Runs validation on the test set and prints metrics.

Usage:
    cd d:/Work/jwinfotech/Videoanalystics/video-analytics
    python pycode/scripts/validate_gun_detector.py
"""

import os
import sys
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "pycode", "models", "gun_detector", "weights", "best.pt")
DATASET_YAML = os.path.join(
    PROJECT_ROOT, "resources", "datasets", "weapon-detection", "data.yaml"
)


def main():
    print("=" * 60)
    print("  GUN DETECTOR — Validation Script")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found: {MODEL_PATH}")
        print("Run train_gun_detector.py first.")
        sys.exit(1)

    if not os.path.exists(DATASET_YAML):
        print(f"\nERROR: Dataset config not found: {DATASET_YAML}")
        sys.exit(1)

    print(f"  Model  : {MODEL_PATH}")
    print(f"  Dataset: {DATASET_YAML}")
    print(f"  Device : {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # Load model
    model = YOLO(MODEL_PATH)

    # Run validation
    print("\nRunning validation...\n")
    results = model.val(
        data=DATASET_YAML,
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.5,
        device="0" if torch.cuda.is_available() else "cpu",
        verbose=True,
        plots=True,
    )

    # Print key metrics
    print("\n" + "=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)

    if hasattr(results, 'box'):
        metrics = results.box
        print(f"  mAP@50      : {metrics.map50:.4f}")
        print(f"  mAP@50-95   : {metrics.map:.4f}")

        if hasattr(metrics, 'ap_class_index'):
            print(f"\n  Per-class AP@50:")
            class_names = model.names
            for i, cls_idx in enumerate(metrics.ap_class_index):
                name = class_names.get(int(cls_idx), f"class_{cls_idx}")
                ap50 = metrics.maps[i] if i < len(metrics.maps) else 0
                print(f"    {name:20s}: {ap50:.4f}")

    print("=" * 60)
    print("\n✅ Validation complete. Check the plots in the results directory.")


if __name__ == "__main__":
    main()

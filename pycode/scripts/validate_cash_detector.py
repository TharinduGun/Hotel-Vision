"""
Validate Cash Detection Model
===============================
Runs validation on the trained cash detector and prints metrics.
Optionally runs prediction on sample images for visual inspection.

Usage:
    cd d:/Work/jwinfotech/Videoanalystics/video-analytics
    python pycode/scripts/validate_cash_detector.py
"""

import os
import sys
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Trained model path
MODEL_PATH = os.path.join(PROJECT_ROOT, "pycode", "models", "cash_detector", "weights", "best.pt")

# Dataset config
DATASET_YAML = os.path.join(
    PROJECT_ROOT, "resources", "datasets", "cash.v7i.yolov8", "data.yaml"
)

# Validation settings
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
DEVICE = "0"


def validate():
    """Run full validation on the test set and print metrics."""
    print("=" * 60)
    print("  CASH DETECTOR — Validation")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Trained model not found: {MODEL_PATH}")
        print("Run train_cash_detector.py first!")
        sys.exit(1)

    print(f"  Model   : {MODEL_PATH}")
    print(f"  Dataset : {DATASET_YAML}")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    # Run validation on the validation set
    print("\n--- Validating on validation set ---")
    val_results = model.val(
        data=DATASET_YAML,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        split="val",
        verbose=True,
    )

    print("\n--- Validating on test set ---")
    test_results = model.val(
        data=DATASET_YAML,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        split="test",
        verbose=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Val  mAP50     : {val_results.box.map50:.4f}")
    print(f"  Val  mAP50-95  : {val_results.box.map:.4f}")
    print(f"  Test mAP50     : {test_results.box.map50:.4f}")
    print(f"  Test mAP50-95  : {test_results.box.map:.4f}")
    print("=" * 60)

    if val_results.box.map50 >= 0.70:
        print("\n✅ Model looks good! mAP50 >= 0.70")
    else:
        print("\n⚠️  mAP50 is below 0.70 — consider more epochs or data augmentation")


def predict_samples(sample_dir=None):
    """Run prediction on sample images for visual inspection."""
    if sample_dir is None:
        # Use a few test images
        test_images_dir = os.path.join(
            PROJECT_ROOT, "resources", "datasets", "cash.v7i.yolov8", "test", "images"
        )
        sample_dir = test_images_dir

    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        return

    print(f"\n--- Running prediction on samples from: {sample_dir} ---")
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=sample_dir,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device=DEVICE,
        save=True,
        project=os.path.join(PROJECT_ROOT, "pycode", "models", "cash_detector"),
        name="sample_predictions",
        exist_ok=True,
        max_det=20,
    )

    output_dir = os.path.join(
        PROJECT_ROOT, "pycode", "models", "cash_detector", "sample_predictions"
    )
    print(f"\n✅ Annotated predictions saved to: {output_dir}")


if __name__ == "__main__":
    validate()

    # Uncomment to also run visual predictions on test images:
    # predict_samples()

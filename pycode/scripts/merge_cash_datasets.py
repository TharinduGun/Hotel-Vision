"""
Merge Cash Detection Datasets
==============================
Combines multiple Roboflow cash/banknote detection datasets into a single
unified dataset for training, with ONE class: 'Cash'.

What this script does:
1. Scans all 4 downloaded datasets
2. Remaps all classes (coins AND banknotes) to a single class 0 = "Cash"
3. Converts any polygon annotations to bounding boxes
4. Copies images with unique naming (avoids collisions)
5. Splits into train/valid/test
6. Generates a unified data.yaml

Usage:
  python pycode/scripts/merge_cash_datasets.py
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Configuration ──────────────────────────────────────────────────────

# Source datasets (downloaded from Roboflow)
SOURCE_DIR = PROJECT_ROOT / "resources" / "datasets" / "cash"

DATASETS = [
    {
        "name": "Cash Counter.v11-yolov8s.yolov8",
        "classes": ['Dime', 'Nickel', 'Penny', 'Quarter', 'fifty', 'five', 'hundred', 'one', 'ten', 'twenty'],
        # Classes 0-3 = coins, 4-9 = banknotes
        # For cash detection, we want ALL of them as class 0 = "Cash"
        # Set include_coins=False to only include banknotes (classes 4-9)
        "include_coins": True,
    },
    {
        "name": "cash.v1i.yolov8",
        "classes": ['Dime', 'Nickel', 'Penny', 'Quarter', 'fifty', 'five', 'one', 'ten', 'twenty'],
        "include_coins": True,
    },
    {
        "name": "coins and banknotes.v2i.yolov8",
        "classes": ['Dime', 'Nickel', 'Penny', 'Quarter', 'fifty', 'five', 'hundred', 'one', 'ten', 'twenty'],
        "include_coins": True,
    },
    {
        "name": "currency.v2-release.yolov8",
        "classes": ['Dime', 'Nickel', 'Penny', 'Quarter', 'fifty', 'five', 'hundred', 'one', 'ten', 'twenty'],
        "include_coins": True,
    },
]

# Coin class names (used when include_coins=False to filter them out)
COIN_CLASSES = {'Dime', 'Nickel', 'Penny', 'Quarter'}

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "resources" / "datasets" / "cash-detection-v3"

# Split ratios
TRAIN_RATIO = 0.80
VALID_RATIO = 0.15
TEST_RATIO = 0.05

# Random seed for reproducibility
RANDOM_SEED = 42


def polygon_to_bbox(parts):
    """
    Convert a polygon annotation to a bounding box.
    
    Polygon format: class_id x1 y1 x2 y2 x3 y3 ...
    BBox format:     class_id cx cy w h
    
    All values are normalized [0, 1].
    """
    class_id = parts[0]
    coords = [float(x) for x in parts[1:]]
    
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None
    
    xs = coords[0::2]  # Every other value starting from 0
    ys = coords[1::2]  # Every other value starting from 1
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    
    # Sanity check
    if w <= 0 or h <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
        return None
    
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_label_file(label_path, dataset_cfg):
    """
    Read a label file and remap all classes to class 0 = "Cash".
    Convert polygons to bounding boxes if needed.
    
    Returns list of remapped label lines, or empty list if no valid labels.
    """
    if not label_path.exists():
        return []
    
    lines = label_path.read_text().strip().split('\n')
    remapped = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        original_class_id = int(parts[0])
        class_name = dataset_cfg["classes"][original_class_id] if original_class_id < len(dataset_cfg["classes"]) else None
        
        # Optionally skip coins
        if not dataset_cfg.get("include_coins", True) and class_name in COIN_CLASSES:
            continue
        
        if len(parts) == 5:
            # Standard YOLO bbox: class_id cx cy w h
            # Remap class to 0
            remapped.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        else:
            # Polygon annotation — convert to bbox
            bbox_line = polygon_to_bbox(parts)
            if bbox_line:
                remapped.append(bbox_line)
    
    return remapped


def collect_all_samples(datasets):
    """
    Collect all (image_path, label_lines) pairs from all datasets.
    """
    samples = []
    stats = {
        "total_images": 0,
        "images_with_labels": 0,
        "total_annotations": 0,
        "polygons_converted": 0,
        "skipped_no_labels": 0,
    }
    
    for ds_cfg in datasets:
        ds_path = SOURCE_DIR / ds_cfg["name"]
        
        if not ds_path.exists():
            print(f"  WARNING: Dataset not found: {ds_path}")
            continue
        
        # Collect from train, valid, and test splits
        for split in ["train", "valid", "test"]:
            img_dir = ds_path / split / "images"
            label_dir = ds_path / split / "labels"
            
            if not img_dir.exists():
                continue
            
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                
                stats["total_images"] += 1
                
                # Find corresponding label
                label_file = label_dir / (img_file.stem + ".txt")
                label_lines = process_label_file(label_file, ds_cfg)
                
                if label_lines:
                    stats["images_with_labels"] += 1
                    stats["total_annotations"] += len(label_lines)
                    samples.append((img_file, label_lines, ds_cfg["name"]))
                else:
                    stats["skipped_no_labels"] += 1
    
    return samples, stats


def create_output_structure(output_dir):
    """Create the output directory structure."""
    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("CASH DATASET MERGER")
    print("=" * 70)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Check source datasets exist
    print("Scanning datasets...")
    for ds in DATASETS:
        ds_path = SOURCE_DIR / ds["name"]
        exists = "✅" if ds_path.exists() else "❌ NOT FOUND"
        coins_str = "(coins+banknotes)" if ds.get("include_coins", True) else "(banknotes only)"
        print(f"  {exists} {ds['name']} {coins_str}")
    print()
    
    # Collect all samples
    print("Collecting and remapping annotations...")
    samples, stats = collect_all_samples(DATASETS)
    
    print(f"\n  Total images found:     {stats['total_images']}")
    print(f"  Images with labels:     {stats['images_with_labels']}")
    print(f"  Total annotations:      {stats['total_annotations']}")
    print(f"  Skipped (no labels):    {stats['skipped_no_labels']}")
    print()
    
    if not samples:
        print("ERROR: No valid samples found!")
        return
    
    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    
    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    
    splits = {
        "train": samples[:n_train],
        "valid": samples[n_train:n_train + n_valid],
        "test": samples[n_train + n_valid:],
    }
    
    print(f"Split: train={len(splits['train'])}, valid={len(splits['valid'])}, test={len(splits['test'])}")
    
    # Create output directory
    if OUTPUT_DIR.exists():
        print(f"\nWARNING: Output directory exists. Removing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    create_output_structure(OUTPUT_DIR)
    
    # Copy files
    print("\nCopying files...")
    for split_name, split_samples in splits.items():
        for idx, (img_path, label_lines, ds_name) in enumerate(split_samples):
            # Create unique filename: dataset_prefix + index + original extension
            ds_prefix = ds_name.split(".")[0].replace(" ", "_").lower()[:10]
            new_name = f"{ds_prefix}_{idx:05d}{img_path.suffix}"
            
            # Copy image
            dst_img = OUTPUT_DIR / split_name / "images" / new_name
            shutil.copy2(img_path, dst_img)
            
            # Write remapped label
            dst_label = OUTPUT_DIR / split_name / "labels" / (new_name.rsplit(".", 1)[0] + ".txt")
            dst_label.write_text("\n".join(label_lines) + "\n")
        
        print(f"  {split_name}: {len(split_samples)} images copied")
    
    # Write data.yaml
    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["Cash"],
    }
    
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\ndata.yaml written to: {yaml_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Output directory:  {OUTPUT_DIR}")
    print(f"Total images:      {len(samples)}")
    print(f"Classes:           1 (Cash)")
    print(f"All annotations remapped to class 0 = 'Cash'")
    print()
    print("NEXT STEPS:")
    print("  1. Add synthetic CCTV images (Source 2) to train/images + train/labels")
    print("  2. Add hard negative images (Source 3) to train/images (no labels)")
    print("  3. Train: python pycode/scripts/train_cash_v3.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

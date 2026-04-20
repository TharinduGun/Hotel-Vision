"""
Merge ALL Weapon Detection Datasets (v1 + v2 + CCTV datasets)
================================================================
Merges four weapon detection datasets into a single unified dataset
for training a domain-adapted gun detection model.

Datasets and Class Remapping:
  v1 (weapon-detection):    ['None', 'gun', 'pistol']
    → class 1 ('gun')    → 0 ('weapon')
    → class 2 ('pistol') → 0 ('weapon')
    → class 0 ('None')   → SKIP

  v2 (weapon-detection-v2): ['Person', 'Vehicle', 'Weapon']
    → class 2 ('Weapon') → 0 ('weapon')
    → class 0,1           → SKIP

  CCTV Set 1 (weaponnewset1): ['person', 'weapon']  (Simuletic synthetic CCTV)
    → class 1 ('weapon') → 0 ('weapon')
    → class 0 ('person') → SKIP

  CCTV Set 2 (weaponnewset2): ['person', 'weapon']  (Roboflow CCTV v3)
    → class 1 ('weapon') → 0 ('weapon')
    → class 0 ('person') → SKIP

Output: resources/datasets/weapon-combined-v2/
  ├── train/images/  train/labels/
  ├── valid/images/  valid/labels/
  ├── test/images/   test/labels/
  └── data.yaml
"""

import os
import random
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DS_V1 = ROOT / "resources" / "datasets" / "weapon-detection"
DS_V2 = ROOT / "resources" / "datasets" / "weapon-detection-v2"
DS_CCTV1 = ROOT / "resources" / "datasets" / "weaponnewset1" / "Dataset"
DS_CCTV2 = ROOT / "resources" / "datasets" / "weaponnewset2"
DS_HARD_NEG = ROOT / "resources" / "datasets" / "hard_negatives"
DS_OUT = ROOT / "resources" / "datasets" / "weapon-combined-v2"

# ── Class remapping tables ─────────────────────────────────────────────
# Maps (original_class_id) → new_class_id, or None to skip
V1_REMAP = {0: None, 1: 0, 2: 0}      # None→skip, gun→weapon, pistol→weapon
V2_REMAP = {0: None, 1: None, 2: 0}   # Person→skip, Vehicle→skip, Weapon→weapon
CCTV_REMAP = {0: None, 1: 0}          # person→skip, weapon→weapon

UNIFIED_CLASSES = {0: "weapon"}

# ── Splits ─────────────────────────────────────────────────────────────
SPLITS = ["train", "valid", "test"]


def remap_label_file(src_path: Path, dst_path: Path, remap: dict) -> int:
    """
    Read a YOLO label file, remap classes, write output.
    Returns the number of weapon annotations kept.
    """
    kept = 0
    lines_out = []
    
    if not src_path.exists():
        return 0
    
    for line in src_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        orig_cls = int(parts[0])
        new_cls = remap.get(orig_cls, None)
        
        if new_cls is not None:
            parts[0] = str(new_cls)
            lines_out.append(" ".join(parts))
            kept += 1
    
    # Write output (even if empty — background image)
    dst_path.write_text("\n".join(lines_out) + "\n" if lines_out else "")
    return kept


def process_split(
    ds_name: str,
    img_dir: Path,
    lbl_dir: Path,
    remap: dict,
    split: str,
    prefix: str,
) -> dict:
    """Process one split of a dataset. Returns stats dict."""
    out_img = DS_OUT / split / "images"
    out_lbl = DS_OUT / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "with_weapons": 0, "weapon_annotations": 0}
    
    if not img_dir.exists():
        return stats
    
    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ])
    
    for img_file in tqdm(img_files, desc=f"  {ds_name}/{split}", leave=False):
        stats["total"] += 1
        
        # Unique filename with prefix to avoid collisions
        new_name = f"{prefix}_{img_file.name}"
        lbl_name = f"{prefix}_{img_file.stem}.txt"
        
        # Copy image
        shutil.copy2(img_file, out_img / new_name)
        
        # Process label
        src_lbl = lbl_dir / f"{img_file.stem}.txt"
        kept = remap_label_file(src_lbl, out_lbl / lbl_name, remap)
        
        if kept > 0:
            stats["with_weapons"] += 1
            stats["weapon_annotations"] += kept
    
    return stats


def process_flat_dataset(
    ds_name: str,
    img_dir: Path,
    lbl_dir: Path,
    remap: dict,
    prefix: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.15,
) -> dict:
    """
    Process a dataset with no train/valid/test split.
    Randomly splits into train/valid/test.
    """
    all_stats = {s: {"total": 0, "with_weapons": 0, "weapon_annotations": 0} for s in SPLITS}
    
    img_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ])
    
    random.shuffle(img_files)
    n = len(img_files)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    
    split_map = {}
    for i, f in enumerate(img_files):
        if i < n_train:
            split_map[f] = "train"
        elif i < n_train + n_valid:
            split_map[f] = "valid"
        else:
            split_map[f] = "test"
    
    for img_file in tqdm(img_files, desc=f"  {ds_name}", leave=False):
        split = split_map[img_file]
        out_img = DS_OUT / split / "images"
        out_lbl = DS_OUT / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        
        new_name = f"{prefix}_{img_file.name}"
        lbl_name = f"{prefix}_{img_file.stem}.txt"
        
        shutil.copy2(img_file, out_img / new_name)
        
        src_lbl = lbl_dir / f"{img_file.stem}.txt"
        kept = remap_label_file(src_lbl, out_lbl / lbl_name, remap)
        
        all_stats[split]["total"] += 1
        if kept > 0:
            all_stats[split]["with_weapons"] += 1
            all_stats[split]["weapon_annotations"] += kept
    
    return all_stats


def main():
    random.seed(42)  # Reproducible splits
    
    print("=" * 60)
    print("  MERGE WEAPON DATASETS (v1 + v2 + CCTV1 + CCTV2)")
    print("=" * 60)
    
    # Clean output directory
    if DS_OUT.exists():
        print(f"\n  Removing existing: {DS_OUT}")
        shutil.rmtree(DS_OUT)
    
    all_stats = {}
    
    # ── Dataset 1: weapon-detection (v1) ──────────────────────────────
    print(f"\n📦 Dataset v1: {DS_V1}")
    for split in SPLITS:
        img_dir = DS_V1 / split / "images"
        lbl_dir = DS_V1 / split / "labels"
        stats = process_split("v1", img_dir, lbl_dir, V1_REMAP, split, "v1")
        all_stats[f"v1/{split}"] = stats
    
    # ── Dataset 2: weapon-detection-v2 ────────────────────────────────
    print(f"\n📦 Dataset v2: {DS_V2}")
    for split in SPLITS:
        img_dir = DS_V2 / split / "images"
        lbl_dir = DS_V2 / split / "labels"
        stats = process_split("v2", img_dir, lbl_dir, V2_REMAP, split, "v2")
        all_stats[f"v2/{split}"] = stats
    
    # ── Dataset 3: CCTV Set 1 (flat, no split) ───────────────────────
    print(f"\n📦 CCTV Set 1: {DS_CCTV1}")
    cctv1_stats = process_flat_dataset(
        "cctv1",
        DS_CCTV1 / "images",
        DS_CCTV1 / "labels",
        CCTV_REMAP,
        "cctv1",
    )
    for split in SPLITS:
        all_stats[f"cctv1/{split}"] = cctv1_stats[split]
    
    # ── Dataset 4: CCTV Set 2 (Roboflow format) ──────────────────────
    print(f"\n📦 CCTV Set 2: {DS_CCTV2}")
    for split in SPLITS:
        img_dir = DS_CCTV2 / split / "images"
        lbl_dir = DS_CCTV2 / split / "labels"
        stats = process_split("cctv2", img_dir, lbl_dir, CCTV_REMAP, split, "cctv2")
        all_stats[f"cctv2/{split}"] = stats
    
    # ── Dataset 5: Hard Negatives ──────────────────────────────────────
    print(f"\n📦 Hard Negatives: {DS_HARD_NEG}")
    hn_stats = process_flat_dataset(
        "hard_negatives",
        DS_HARD_NEG / "images" / "train",
        DS_HARD_NEG / "labels" / "train",
        {},
        "hn",
    )
    for split in SPLITS:
        all_stats[f"hard_negatives/{split}"] = hn_stats[split]
    
    # ── Write data.yaml ──────────────────────────────────────────────
    data_yaml = {
        "names": list(UNIFIED_CLASSES.values()),
        "nc": len(UNIFIED_CLASSES),
        "path": str(DS_OUT),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    yaml_path = DS_OUT / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    
    # Count totals per split
    for split in SPLITS:
        total_imgs = sum(
            v["total"] for k, v in all_stats.items() if k.endswith(f"/{split}")
        )
        total_weapons = sum(
            v["with_weapons"] for k, v in all_stats.items() if k.endswith(f"/{split}")
        )
        total_annots = sum(
            v["weapon_annotations"] for k, v in all_stats.items() if k.endswith(f"/{split}")
        )
        print(f"  {split:6s}: {total_imgs:>6,} images | "
              f"{total_weapons:>5,} with weapons | "
              f"{total_annots:>6,} weapon annotations")
    
    grand_total = sum(v["total"] for v in all_stats.values())
    print(f"\n  TOTAL : {grand_total:>6,} images")
    print(f"  Output: {DS_OUT}")
    print(f"  Config: {yaml_path}")
    print("=" * 60)
    
    # Per-dataset breakdown
    print("\n  Per-dataset breakdown:")
    for ds in ["v1", "v2", "cctv1", "cctv2", "hard_negatives"]:
        ds_total = sum(
            v["total"] for k, v in all_stats.items() if k.startswith(f"{ds}/")
        )
        ds_weapons = sum(
            v["with_weapons"] for k, v in all_stats.items() if k.startswith(f"{ds}/")
        )
        print(f"    {ds:6s}: {ds_total:>6,} images ({ds_weapons:>5,} with weapons)")


if __name__ == "__main__":
    main()

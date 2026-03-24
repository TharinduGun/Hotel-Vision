"""
Merge Weapon Detection Datasets
=================================
Merges weapon-detection (v1) and weapon-detection-v2 (v2) into a single
unified dataset for training.

Class Remapping:
  v1 classes: ['None', 'gun', 'pistol']
    → class 0 ('None')  → SKIP (not a weapon)
    → class 1 ('gun')   → 0 ('weapon')
    → class 2 ('pistol')→ 0 ('weapon')

  v2 classes: ['Person', 'Vehicle', 'Weapon']
    → class 0 ('Person')  → SKIP
    → class 1 ('Vehicle') → SKIP
    → class 2 ('Weapon')  → 0 ('weapon')

Output: resources/datasets/weapon-combined/
  ├── train/images/  train/labels/
  ├── valid/images/  valid/labels/
  ├── test/images/   test/labels/
  └── data.yaml
"""

import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DS_V1 = ROOT / "resources" / "datasets" / "weapon-detection"
DS_V2 = ROOT / "resources" / "datasets" / "weapon-detection-v2"
DS_OUT = ROOT / "resources" / "datasets" / "weapon-combined"

# ── Class remapping tables ─────────────────────────────────────────────
# Maps (original_class_id) → new_class_id, or None to skip
V1_REMAP = {0: None, 1: 0, 2: 0}    # None→skip, gun→weapon, pistol→weapon
V2_REMAP = {0: None, 1: None, 2: 0}  # Person→skip, Vehicle→skip, Weapon→weapon

UNIFIED_CLASSES = {0: "weapon"}


def remap_label_file(src_label: Path, dst_label: Path, remap: dict) -> bool:
    """
    Read a YOLO label file, remap class IDs, and write the output.
    Returns True if any valid annotations remain, False if all were skipped.
    """
    if not src_label.exists():
        # Image with no label = negative example (keep it with empty label)
        dst_label.write_text("")
        return True

    lines = src_label.read_text().strip().splitlines()
    remapped_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        old_cls = int(parts[0])
        new_cls = remap.get(old_cls, None)

        if new_cls is None:
            continue  # Skip non-weapon classes

        parts[0] = str(new_cls)
        remapped_lines.append(" ".join(parts))

    # Write remapped file (may be empty = negative example)
    dst_label.write_text("\n".join(remapped_lines) + "\n" if remapped_lines else "")
    return True


def merge_split(split: str, ds_path: Path, remap: dict, prefix: str, stats: dict):
    """Merge one split (train/valid/test) from one dataset into the output."""
    src_img_dir = ds_path / split / "images"
    src_lbl_dir = ds_path / split / "labels"
    dst_img_dir = DS_OUT / split / "images"
    dst_lbl_dir = DS_OUT / split / "labels"

    if not src_img_dir.exists():
        print(f"  ⚠ {src_img_dir} not found, skipping.")
        return

    images = sorted(src_img_dir.iterdir())
    for img_path in tqdm(images, desc=f"  {prefix}/{split}", leave=False):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue

        # Create unique filename with prefix to avoid collisions
        new_name = f"{prefix}_{img_path.name}"
        dst_img = dst_img_dir / new_name

        # Copy image
        shutil.copy2(img_path, dst_img)

        # Remap label
        label_name = img_path.stem + ".txt"
        src_label = src_lbl_dir / label_name
        dst_label = dst_lbl_dir / f"{prefix}_{label_name}"
        remap_label_file(src_label, dst_label, remap)

        stats["images"] += 1
        if dst_label.exists() and dst_label.read_text().strip():
            stats["with_weapons"] += 1
        else:
            stats["negatives"] += 1


def main():
    print("=" * 60)
    print("  Weapon Dataset Merger")
    print("=" * 60)

    # Create output directories
    for split in ("train", "valid", "test"):
        (DS_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (DS_OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "with_weapons": 0, "negatives": 0}

    # ── Merge v1 ───────────────────────────────────────────────────────
    print(f"\n📦 Dataset v1: {DS_V1}")
    print(f"   Classes: ['None', 'gun', 'pistol'] → remap gun/pistol → 'weapon'")
    for split in ("train", "valid", "test"):
        merge_split(split, DS_V1, V1_REMAP, "v1", stats)

    # ── Merge v2 ───────────────────────────────────────────────────────
    print(f"\n📦 Dataset v2: {DS_V2}")
    print(f"   Classes: ['Person', 'Vehicle', 'Weapon'] → keep only 'Weapon'")
    for split in ("train", "valid", "test"):
        merge_split(split, DS_V2, V2_REMAP, "v2", stats)

    # ── Write unified data.yaml ────────────────────────────────────────
    data_yaml = {
        "path": str(DS_OUT),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": list(UNIFIED_CLASSES.values()),
    }

    yaml_path = DS_OUT / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Merge Complete!")
    print("=" * 60)
    print(f"  Output:         {DS_OUT}")
    print(f"  Total images:   {stats['images']}")
    print(f"  With weapons:   {stats['with_weapons']}")
    print(f"  Negatives:      {stats['negatives']}")
    print(f"  data.yaml:      {yaml_path}")
    print(f"\n  Unified classes: {UNIFIED_CLASSES}")
    print("=" * 60)


if __name__ == "__main__":
    main()

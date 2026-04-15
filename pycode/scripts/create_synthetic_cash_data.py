"""
Synthetic CCTV Cash Data Generator (Source 2)
==============================================
Creates training images that look like what a hotel CCTV camera actually sees:
tiny banknotes on counters/tables, seen from above, with CCTV-quality artifacts.

Pipeline:
  1. Extract clean banknote crops from the merged dataset (Source 1 output)
  2. Extract background frames from your hotel CCTV video
  3. Paste banknotes onto backgrounds at CCTV-realistic scale
  4. Apply post-processing (blur, noise, compression)
  5. Auto-generate YOLO labels

This is the most impactful data source because it bridges the gap between
clean stock photos of money and what the camera actually sees at a hotel.

Usage:
  # Step 1: Extract banknote crops from the merged dataset
  python pycode/scripts/create_synthetic_cash_data.py extract-crops

  # Step 2: Extract background frames from CCTV video
  python pycode/scripts/create_synthetic_cash_data.py extract-backgrounds

  # Step 3: Generate synthetic images
  python pycode/scripts/create_synthetic_cash_data.py generate --count 800

  # Or do all steps in one go:
  python pycode/scripts/create_synthetic_cash_data.py all --count 800
"""

import os
import sys
import random
import argparse
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Configuration ──────────────────────────────────────────────────────

# Source 1 merged dataset (output of merge_cash_datasets.py)
MERGED_DATASET = PROJECT_ROOT / "resources" / "datasets" / "cash-detection-v3"

# CCTV video for background extraction
CCTV_VIDEO = PROJECT_ROOT / "resources" / "videos" / "Indoor_Original_VideoStream.mp4"

# Working directories
WORK_DIR = PROJECT_ROOT / "resources" / "datasets" / "synthetic_workdir"
CROP_DIR = WORK_DIR / "banknote_crops"
BG_DIR = WORK_DIR / "backgrounds"

# Output (will be added to the merged dataset)
SYNTH_OUTPUT_IMAGES = MERGED_DATASET / "train" / "images"
SYNTH_OUTPUT_LABELS = MERGED_DATASET / "train" / "labels"

# Generation parameters
DEFAULT_COUNT = 800
MIN_NOTES_PER_IMAGE = 1
MAX_NOTES_PER_IMAGE = 3
BANKNOTE_SCALE_RANGE = (40, 120)  # Width in pixels at CCTV scale (tiny!)
ROTATION_RANGE = (-45, 45)  # Degrees
BRIGHTNESS_JITTER = 0.3  # ± brightness variation
CCTV_BLUR_RANGE = (0, 2)  # Gaussian blur kernel (0 = no blur)
JPEG_QUALITY_RANGE = (40, 80)  # CCTV compression quality
NOISE_SIGMA_RANGE = (0, 15)  # Gaussian noise std

RANDOM_SEED = 42


# ── Step 1: Extract Banknote Crops ─────────────────────────────────────

def extract_crops(args):
    """
    Extract individual banknote regions from the merged dataset.
    These crops are the "foreground" objects we'll paste onto backgrounds.
    """
    print("=" * 70)
    print("STEP 1: Extracting banknote crops from merged dataset")
    print("=" * 70)
    
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    
    img_dir = MERGED_DATASET / "train" / "images"
    label_dir = MERGED_DATASET / "train" / "labels"
    
    if not img_dir.exists():
        print(f"ERROR: Merged dataset not found at {img_dir}")
        print("Run merge_cash_datasets.py first!")
        return
    
    crop_count = 0
    max_crops = getattr(args, 'max_crops', 500)
    
    images = list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.[pP][nN][gG]"))
    random.shuffle(images)
    
    for img_path in images:
        if crop_count >= max_crops:
            break
        
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        for line in label_path.read_text().strip().split('\n'):
            if crop_count >= max_crops:
                break
            
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            _, cx, cy, bw, bh = [float(x) for x in parts]
            
            # Convert normalized to pixel coordinates
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Pad slightly
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            
            crop_path = CROP_DIR / f"crop_{crop_count:05d}.png"
            cv2.imwrite(str(crop_path), crop)
            crop_count += 1
    
    print(f"Extracted {crop_count} banknote crops to {CROP_DIR}")


# ── Step 2: Extract Background Frames ──────────────────────────────────

def extract_backgrounds(args):
    """
    Extract random frames from the CCTV video to use as backgrounds.
    These represent what the camera actually sees at the hotel.
    """
    print("=" * 70)
    print("STEP 2: Extracting background frames from CCTV video")
    print("=" * 70)
    
    BG_DIR.mkdir(parents=True, exist_ok=True)
    
    video_path = getattr(args, 'video', str(CCTV_VIDEO))
    max_backgrounds = getattr(args, 'max_backgrounds', 200)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample random frame indices (spread across the video)
    random.seed(RANDOM_SEED)
    frame_indices = sorted(random.sample(range(0, total_frames), min(max_backgrounds, total_frames)))
    
    bg_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        bg_path = BG_DIR / f"bg_{bg_count:05d}.jpg"
        cv2.imwrite(str(bg_path), frame)
        bg_count += 1
    
    cap.release()
    print(f"Extracted {bg_count} background frames to {BG_DIR}")


# ── Step 3: Generate Synthetic Images ──────────────────────────────────

def random_transform_banknote(crop, target_width):
    """
    Apply random transformations to a banknote crop:
    - Resize to CCTV scale
    - Random rotation
    - Brightness/contrast jitter
    """
    h, w = crop.shape[:2]
    
    # Resize to target width (maintaining aspect ratio)
    scale = target_width / max(w, 1)
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w < 5 or new_h < 5:
        return None, None
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Random rotation
    angle = random.uniform(*ROTATION_RANGE)
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box after rotation
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    rot_w = int(new_h * sin + new_w * cos)
    rot_h = int(new_h * cos + new_w * sin)
    M[0, 2] += (rot_w - new_w) / 2
    M[1, 2] += (rot_h - new_h) / 2
    
    rotated = cv2.warpAffine(resized, M, (rot_w, rot_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Create mask (non-black pixels after rotation)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Brightness/contrast jitter
    brightness = random.uniform(1.0 - BRIGHTNESS_JITTER, 1.0 + BRIGHTNESS_JITTER)
    rotated = np.clip(rotated.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    
    return rotated, mask


def paste_banknote_on_background(background, banknote, mask, x, y):
    """
    Paste a banknote crop onto a background image using alpha blending.
    
    Returns the bounding box of the pasted region (for label generation).
    """
    bg_h, bg_w = background.shape[:2]
    bn_h, bn_w = banknote.shape[:2]
    
    # Clip to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + bn_w)
    y2 = min(bg_h, y + bn_h)
    
    # Calculate source region
    src_x1 = x1 - x
    src_y1 = y1 - y
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Get the regions
    bg_region = background[y1:y2, x1:x2]
    bn_region = banknote[src_y1:src_y2, src_x1:src_x2]
    mask_region = mask[src_y1:src_y2, src_x1:src_x2]
    
    if bg_region.shape[:2] != bn_region.shape[:2]:
        return None
    
    # Alpha blend
    mask_3ch = cv2.merge([mask_region, mask_region, mask_region]).astype(np.float32) / 255.0
    blended = (bn_region.astype(np.float32) * mask_3ch +
               bg_region.astype(np.float32) * (1 - mask_3ch))
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    
    # Find actual bounding box of visible object (where mask is non-zero)
    visible_mask = mask_region > 0
    if not visible_mask.any():
        return None
    
    rows = np.any(visible_mask, axis=1)
    cols = np.any(visible_mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    
    actual_x1 = x1 + c_min
    actual_y1 = y1 + r_min
    actual_x2 = x1 + c_max + 1
    actual_y2 = y1 + r_max + 1
    
    return (actual_x1, actual_y1, actual_x2, actual_y2)


def apply_cctv_effects(image):
    """
    Apply effects that simulate CCTV camera quality:
    - Slight Gaussian blur
    - Gaussian noise
    - JPEG compression artifacts
    """
    # Gaussian blur
    blur_k = random.choice([0, 0, 1, 1, 3])  # Often no blur, sometimes slight
    if blur_k > 0:
        image = cv2.GaussianBlur(image, (blur_k * 2 + 1, blur_k * 2 + 1), 0)
    
    # Gaussian noise
    sigma = random.uniform(*NOISE_SIGMA_RANGE)
    if sigma > 0:
        noise = np.random.randn(*image.shape).astype(np.float32) * sigma
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # JPEG compression (encode then decode to add artifacts)
    quality = random.randint(*JPEG_QUALITY_RANGE)
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    return image


def generate_synthetic_images(args):
    """
    Generate synthetic CCTV-style images by pasting banknote crops
    onto hotel background frames.
    """
    count = getattr(args, 'count', DEFAULT_COUNT)
    
    print("=" * 70)
    print(f"STEP 3: Generating {count} synthetic CCTV images")
    print("=" * 70)
    
    # Load crops and backgrounds
    crops = list(CROP_DIR.glob("*.png"))
    backgrounds = list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.png"))
    
    if not crops:
        print(f"ERROR: No banknote crops found in {CROP_DIR}")
        print("Run 'extract-crops' first!")
        return
    
    if not backgrounds:
        print(f"ERROR: No backgrounds found in {BG_DIR}")
        print("Run 'extract-backgrounds' first!")
        return
    
    print(f"  Banknote crops:   {len(crops)}")
    print(f"  Backgrounds:      {len(backgrounds)}")
    print(f"  Scale range:      {BANKNOTE_SCALE_RANGE[0]}-{BANKNOTE_SCALE_RANGE[1]}px")
    print()
    
    # Ensure output dirs exist
    SYNTH_OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    SYNTH_OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)
    
    random.seed(RANDOM_SEED + 1)
    generated = 0
    
    for i in range(count):
        # Pick random background
        bg_path = random.choice(backgrounds)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        
        bg_h, bg_w = bg.shape[:2]
        labels = []
        
        # Paste 1-3 banknotes
        num_notes = random.randint(MIN_NOTES_PER_IMAGE, MAX_NOTES_PER_IMAGE)
        
        for _ in range(num_notes):
            # Pick random crop
            crop_path = random.choice(crops)
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            
            # Random scale
            target_width = random.randint(*BANKNOTE_SCALE_RANGE)
            
            transformed, mask = random_transform_banknote(crop, target_width)
            if transformed is None or mask is None:
                continue
            
            # Random position (try to stay within frame)
            max_x = bg_w - transformed.shape[1]
            max_y = bg_h - transformed.shape[0]
            if max_x <= 0 or max_y <= 0:
                continue
            
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            bbox = paste_banknote_on_background(bg, transformed, mask, x, y)
            if bbox:
                bx1, by1, bx2, by2 = bbox
                # Convert to YOLO format (normalized)
                cx = ((bx1 + bx2) / 2) / bg_w
                cy = ((by1 + by2) / 2) / bg_h
                bw = (bx2 - bx1) / bg_w
                bh = (by2 - by1) / bg_h
                
                # Sanity check
                if 0 < cx < 1 and 0 < cy < 1 and bw > 0.005 and bh > 0.005:
                    labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
        if not labels:
            continue
        
        # Apply CCTV effects
        bg = apply_cctv_effects(bg)
        
        # Save
        img_name = f"synth_{generated:05d}.jpg"
        label_name = f"synth_{generated:05d}.txt"
        
        cv2.imwrite(str(SYNTH_OUTPUT_IMAGES / img_name), bg)
        (SYNTH_OUTPUT_LABELS / label_name).write_text("\n".join(labels) + "\n")
        
        generated += 1
        
        if generated % 100 == 0:
            print(f"  Generated {generated}/{count} images...")
    
    print(f"\nGenerated {generated} synthetic images")
    print(f"  Images: {SYNTH_OUTPUT_IMAGES}")
    print(f"  Labels: {SYNTH_OUTPUT_LABELS}")


def run_all(args):
    """Run all three steps in sequence."""
    extract_crops(args)
    extract_backgrounds(args)
    generate_synthetic_images(args)
    
    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE")
    print("=" * 70)
    print("Your cash-detection-v3 dataset now includes:")
    print("  - Public data (merged from 4 Roboflow datasets)")
    print("  - Synthetic CCTV images (banknotes pasted on hotel backgrounds)")
    print()
    print("NEXT STEPS:")
    print("  1. Add hard negatives (Source 3) to train/images (no labels)")
    print("  2. Train: python pycode/scripts/train_cash_v3.py")
    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CCTV cash training data")
    subparsers = parser.add_subparsers(dest="command")
    
    # extract-crops
    p_crops = subparsers.add_parser("extract-crops", help="Extract banknote crops from merged dataset")
    p_crops.add_argument("--max-crops", type=int, default=500, help="Max crops to extract")
    p_crops.set_defaults(func=extract_crops)
    
    # extract-backgrounds
    p_bgs = subparsers.add_parser("extract-backgrounds", help="Extract frames from CCTV video")
    p_bgs.add_argument("--video", type=str, default=str(CCTV_VIDEO), help="Path to CCTV video")
    p_bgs.add_argument("--max-backgrounds", type=int, default=200, help="Max frames to extract")
    p_bgs.set_defaults(func=extract_backgrounds)
    
    # generate
    p_gen = subparsers.add_parser("generate", help="Generate synthetic images")
    p_gen.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of images to generate")
    p_gen.set_defaults(func=generate_synthetic_images)
    
    # all
    p_all = subparsers.add_parser("all", help="Run all steps")
    p_all.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of synthetic images")
    p_all.add_argument("--max-crops", type=int, default=500, help="Max crops to extract")
    p_all.add_argument("--max-backgrounds", type=int, default=200, help="Max backgrounds to extract")
    p_all.set_defaults(func=run_all)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()

# Performance Log — Cash Detection & Interaction System

## Model Details (v2 - 2026-03-10)

| Property | Value |
|----------|-------|
| **Base Model** | YOLOv8m |
| **Training Dataset** | `Hands in transaction.v4i.yolov8` |
| **Classes** | `Transaction` (0) |
| **Image Size** | 640×640 |
| **Early Stopping** | epoch 12 (patience=15) |

## Training Results (v2)

| Epoch | mAP50 | Notes |
|-------|-------|-------|
| **12** | **0.995** | Early stopping triggered. Fantastic model confidence for actual point-of-exchange tracking. |

### System Integration Impact

By pairing this extremely highly-confident YOLOv8v2 model with the new **Interaction Analyzer** (requires hand proximity < 90px for > 1.0s inside a designated transaction ROI), we achieved a massive reduction in false-positive "cash exchanges" (dropped from 90/hr to 7/hr on real test footage).

---## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | YOLOv8m (25.8M parameters) |
| **Training Dataset** | [Roboflow cash-74hjk v7](https://universe.roboflow.com/atm-cochs/cash-74hjk/dataset/7) |
| **Total Images** | 17,964 (15,371 train / 1,013 val / 860 test) |
| **Classes** | `Cash` (0), `person` (1) |
| **Image Size** | 640×640 |
| **Preprocessing** | Auto-orient, resize (fit within) |
| **Augmentation** | 50% horizontal flip, ±15% brightness |

---

## Training Results

| Property | Value |
|----------|-------|
| **Date** | 2026-03-05 |
| **GPU** | NVIDIA GeForce RTX 4060 Ti (16 GB) |
| **PyTorch** | 2.10.0+cu128 |
| **Ultralytics** | 8.4.6 |
| **Epochs** | 18 / 50 (early stopped, best at epoch 8) |
| **Batch Size** | 8 |
| **Training Time** | 2.528 hours |
| **Model Size** | 49.6 MB |
| **Optimizer** | MuSGD (lr=0.01, momentum=0.9) |

### Per-Epoch Snapshot (Training Output)

| Epoch | GPU Mem | Box Loss | Cls Loss | DFL Loss | mAP50 |
|-------|---------|----------|----------|----------|-------|
| 8 | 3.36G | — | — | — | **0.791** (best) |
| 16 | 3.36G | 0.4134 | 0.2432 | 0.8648 | 0.751 |
| 17 | 3.36G | 0.4059 | 0.2393 | 0.8619 | 0.633 |
| 18 | 3.36G | 0.3960 | 0.2342 | 0.8583 | 0.760 |

---

## Validation Results (best.pt — epoch 8)

### Validation Set (1,013 images)

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **All** | 1,013 | 1,836 | 0.891 | 0.749 | **0.829** | 0.602 |
| Cash | 91 | 91 | 0.791 | 0.581 | 0.694 | 0.416 |
| Person | 920 | 1,745 | 0.992 | 0.917 | 0.965 | 0.787 |

### Test Set (860 images — unseen data)

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **All** | 860 | 928 | 0.916 | 0.805 | **0.847** | 0.548 |
| Cash | 145 | 145 | 0.833 | 0.621 | 0.700 | 0.341 |
| Person | 760 | 783 | 0.999 | 0.989 | 0.994 | 0.756 |

### Inference Speed

| Stage | Time |
|-------|------|
| Preprocess | 0.5 ms |
| Inference | 6.1 ms |
| Postprocess | 0.5 ms |
| **Total per frame** | **~7 ms (~143 FPS)** |

---

## Summary

- **Overall mAP50**: 84.7% (test) — model generalizes well (test > val)
- **Cash mAP50**: 70.0% — meets target threshold (≥ 0.70)
- **Cash Precision**: 83.3% — low false alarm rate
- **Cash Recall**: 62.1% — catches ~62% of cash instances; misses from occlusion/small size
- **Person mAP50**: 99.4% — near-perfect
- **Speed**: 7ms/frame — well within real-time budget (40ms for 25 FPS)

## Notes

- Early stopping activated at epoch 18 (patience=10, best epoch=8)
- Dataset heavily imbalanced: 145 cash vs 783 person instances in test set
- Cash recall can be improved with more training data from actual hotel CCTV footage
- Model stored at: `pycode/models/cash_detector/weights/best.pt`

---

## False Positive Mitigation (v0.5.0 — 2026-03-09)

### Problem Observed

When deployed on actual hotel reception footage, the model produced many **false positive detections** on non-cash objects:

- Computer monitors and screens on counters
- Phones, receipts, and name tags
- Rectangular shapes on surfaces

At the same time, **real cash in hands** was detected intermittently — appearing for a few frames then disappearing.

### Solution: Context-Aware Post-Detection Filtering

A two-stage pipeline added to `cash_detector.py` filters raw YOLO detections before they reach the tracker:

#### Stage 1 — Geometric Sanity

| Filter | Value | Purpose |
|--------|-------|---------|
| Min area | 400 px² | Rejects tiny noise blobs |
| Max area | 10% of frame | Rejects impossibly large boxes |
| Aspect ratio | 1.0 – 8.0 | Rejects wrong proportions |

#### Stage 2 — Context Validation (must pass at least one)

| Rule | Condition | Eliminates |
|------|-----------|------------|
| Near hands | Cash overlaps lower 50% of person bbox ± 60px | Random objects far from anyone |
| On counter + near person | Cash in zone AND person within 250px | Monitors, screens on empty counters |
| Between persons | Cash in gap between two nearby people | N/A (exchange detection) |

### Detection Persistence

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `conf_threshold` | 0.25 | **0.35** | Fewer raw false positives while keeping real detections |
| `pickup_debounce` | 3 frames | **5 frames** | 0.2s confirmation before triggering CASH_PICKUP |
| `deposit_debounce` | 5 frames | **20 frames** | 0.8s tolerance for flickery detection gaps |

### Remaining Limitations

| Issue | Root Cause | Mitigation |
|-------|-----------|------------|
| Cash recall ~62% | Limited/imbalanced training data (145 cash vs 783 person instances) | Tolerant deposit_debounce; improve with hotel-specific data |
| Detection flicker | Model confidence varies frame-to-frame on same object | 20-frame deposit_debounce prevents false state resets |
| Some counter FPs remain | Objects resembling cash near active persons on counters | Rule 2 person proximity + geometric filters; further tunable |

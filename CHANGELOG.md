# Changelog

All notable changes to the Video Analytics & Surveillance System are documented here.

---

## [0.6.0] — 2026-03-11 — Advanced Cash Detection & Interaction Logic

### Problem

Even with context-aware filtering (v0.5.0), the model struggled with intermittent detections and false positives of objects that looked like cash in transaction zones. We needed a system robust enough to infer exchanges even when cash visibility flickered, while strictly eliminating false alarms from daily activities.

### What Changed

Implemented complete **6-Layer Architecture** missing structural pieces:

#### Layer 3: Hand Detection (`pycode/utils/hand_detector.py`) [NEW]
- Uses **YOLOv8-pose** to detect human wrists (COCO keypoints 9 & 10).
- Associates hands to tracked individuals via bounding box correlation.
- Computes distances between cashier and customer hands to detect physical interactions.

#### Layer 5: Multi-Signal Timeline Fusion (`pycode/utils/interaction_analyzer.py`) [NEW]
- Merges signals across rolling time windows (zones, hand interactions, cash sightings).
- **Rule Engine**: Infers a `CASH_EXCHANGE` if hands interact closely (< 90px) within transaction zones for over 1.0–1.5 seconds, *even if the YOLO cash model misses the cash*.

#### Layer 6: Fraud Rules (`pycode/utils/fraud_detector.py`) [NEW]
- Evaluates event timelines for suspicious business logic violations:
  - `UNREGISTERED_CASH`: Cashier engaged in an exchange but failed to visit the cash register within 10 seconds.
  - `POSSIBLE_POCKETING`: Cashier pockets cash immediately after an exchange.
- *Tuning Note*: Specifically ignores Customers putting cash in their pockets (a normal behavior that flooded earlier logs).

#### YOLO Model Retraining (v2)
- Re-trained the core cash detector on a highly specific **"Hands in transaction.v4"** dataset to improve performance.
- Script `train_cash_v2.py` added with Early Stopping parameters.
- **Results**: Achieved a stellar **mAP50 of 0.995** in just 12 epochs. This drastically improved raw detection capabilities underneath the interaction rules.

---

## [0.5.0] — 2026-03-09 — Context-Aware Cash Filtering

### Problem

The cash detection model (v0.4.0) produced too many **false positives** — non-cash objects (monitors, phones, receipts, tags, screen reflections) were being flagged as cash. At the same time, **real cash in hands** was detected only intermittently and not tracked persistently. Two specific issues observed:

1. **Static objects on counters** (e.g., monitors, screens) matched the "Cash" pattern because they were rectangular and located inside cashier/money-exchange zones.
2. **Real cash in hands** flickered — the model detected it for a few frames, lost it, detected again — and the tracker kept resetting the state due to short debounce windows.

### What Changed

#### Cash Detector — `pycode/utils/cash_detector.py`

Rewrote the `detect()` method with a **two-stage post-detection filter** pipeline:

**Stage 1 — Geometric Sanity Checks** (applied to all raw YOLO detections):

| Filter | Threshold | What It Rejects |
|--------|-----------|-----------------|
| Min area | 400 px² | Tiny noise blobs (< ~20×20 px) |
| Max area ratio | 10% of frame | Impossibly large detections |
| Aspect ratio | 1.0 – 8.0 | Extremely elongated or oddly proportioned shapes |

**Stage 2 — Context-Aware Validation** (cash must pass at least ONE rule):

| Rule | Logic | What It Catches |
|------|-------|-----------------|
| **Near person's hands** | Cash bbox overlaps lower 50% of any person bbox ± 60px horizontal margin | Cash being held, picked up, counted |
| **On counter zone + near person** | Cash is inside a cashier/money_exchange zone AND a person is within 250px | Cash placed on or picked from a counter |
| **Between two persons** | Cash is in the gap between two persons within 100px of each other | Cash being exchanged hand-to-hand |

Key design decision: **Rule 2 requires person proximity** (not just zone presence) — this is what eliminates monitors, screens, and static objects on counters.

New methods added:

- `_geometric_filter()` — Stage 1 area/aspect checks
- `_contextual_filter()` — Stage 2 context rule orchestrator
- `_check_near_hands()` — Rule 1 implementation
- `_check_on_counter_zone()` — Rule 2 with person proximity requirement
- `_check_between_persons()` — Rule 3 exchange detection
- `_is_near_any_person()` — Euclidean distance helper for Rule 2

The `detect()` method signature changed — it now accepts optional `person_tracks` and `roi_manager` parameters:

```python
# Before (v0.4.0)
cash_detections = cash_detector.detect(frame)

# After (v0.5.0)
cash_detections = cash_detector.detect(frame, person_tracks=curr_frame_tracks, roi_manager=roi_manager)
```

#### Main Pipeline — `pycode/src/main.py`

| Parameter | v0.4.0 | v0.5.0 | Reason |
|-----------|--------|--------|--------|
| `conf_threshold` | 0.25 | **0.35** | Balance: permissive detection + strict context filters |
| `pickup_debounce` | 3 frames | **5 frames** | ~0.2s at 25fps before confirming cash pickup |
| `deposit_debounce` | 5 frames | **20 frames** | ~0.8s tolerance for flickery detections |

Updated `detect()` call to pass `person_tracks` and `roi_manager` for context filtering.

#### Tests — `pycode/test/test_cash_detector.py`

Added **10 new unit tests** covering:

- Geometric filter: tiny/huge/square rejection, valid pass
- Context filter: near hands pass, far from everyone rejected, between persons pass, no persons rejects all, car tracks ignored, hand margin tolerance

### Current Known Issues

- **Cash recall is ~62%** — the model misses some cash instances, especially when partially occluded or small. This is a model quality limitation that can only be improved with more/better training data from actual hotel CCTV footage.
- **Detection flicker** — the model doesn't consistently detect cash every frame even when visible. Mitigated by the high `deposit_debounce` (20 frames) but not eliminated.

---

## [0.4.0] — 2026-03-05 — Cash Detection Pipeline

### Problem

The system could track people and classify roles (Cashier/Customer), but had **no way to detect cash handling** — the core use case for hotel reception monitoring. Needed to train a model, integrate detection, and build event tracking.

### What Was Built

#### Cash Detection Model — Training

Trained a **YOLOv8m** model on the [Roboflow cash-74hjk v7](https://universe.roboflow.com/atm-cochs/cash-74hjk/dataset/7) dataset:

- **15,371 training images**, 1,013 validation, 860 test
- **50 epochs** configured, early-stopped at epoch 18 (patience=10, best at epoch 8)
- **Best mAP50**: 79.1% (val), 84.7% (test) — generalizes well
- **Cash recall**: 62.1% on test set
- **Inference speed**: ~7ms/frame (143 FPS on RTX 4060 Ti)
- Model saved at `pycode/models/cash_detector/weights/best.pt`

#### Cash Detector — `pycode/utils/cash_detector.py` [NEW]

- `CashDetection` class — lightweight data structure for individual detections
- `CashDetector` class — YOLO wrapper with person-association logic
  - `detect(frame)` — runs inference, returns list of `CashDetection` objects
  - `associate_with_persons()` — assigns cash to nearest person via IOU, center-in-bbox, and hand-proximity scoring
  - `draw_detections()` — annotates frames (green for assigned, red for unassigned)

#### Cash Tracker — `pycode/utils/cash_tracker.py` [NEW]

State machine tracking per-person cash interactions:

- **States**: `NO_CASH` ↔ `HOLDING_CASH`
- **Events generated**: `CASH_PICKUP`, `CASH_DEPOSIT`, `CASH_HANDOVER`, `CASH_OUTSIDE_ZONE`, `CASH_POCKET`
- **Debouncing**: configurable frames for pickup/deposit confirmation to prevent flickering
- **Zone awareness**: safe zones (cashier, cash_register) vs. suspicious locations

#### Main Pipeline Integration — `pycode/src/main.py`

- Added `ENABLE_CASH_DETECTION` toggle and `CASH_MODEL_PATH` config
- Cash detection runs after person tracking on each frame
- Cash events exported to separate `cash_events.csv` with session metadata
- Cash holders annotated with 💰 emoji in output video
- Cash summary printed at session end

#### Backend Extensions

- `backend/models.py` — Added `CashEvent` Pydantic model
- `backend/services/aggregations.py` — Cash events contribute to alerts and security score
- `backend/services/csv_adapter.py` — Reads `cash_events.csv` alongside tracking CSV

#### Tests [NEW]

- `pycode/test/test_cash_detector.py` — 6 association logic tests
- `pycode/test/test_cash_tracker.py` — state machine transition tests

---

## [0.3.0] — 2026-02-25 — CSV-to-KPI Alignment

### Problem

The backend API was serving dashboard KPIs, but several values were **hardcoded placeholders** (security score = 94, uptime = 99.9%) because the ML pipeline's CSV didn't output enough data to compute them. Alert timestamps were fake offsets, employee `lastSeen` was relative, and there was no camera ID linking events to specific camera tiles.

### What Changed

#### ML Pipeline — `pycode/src/main.py`

Added **5 new columns** to the tracking CSV output. The original 8 columns are unchanged — these are additions only.

| New Column | Example Value | Purpose |
|---|---|---|
| `Camera_ID` | `CAM-01` | Links each tracking event to a specific camera tile on the dashboard |
| `Session_Start` | `2026-02-25T09:05:12.345678` | Real ISO datetime of when the ML pipeline session started |
| `Bbox_Start` | `306,88,430,298` | Bounding box coordinates (x1,y1,x2,y2) when the object was first detected |
| `Bbox_End` | `623,19,770,250` | Bounding box coordinates when the object was last seen |
| `Dwell_Category` | `LONG` | Pre-classified dwell duration: `SHORT` (<10s), `NORMAL` (10–45s), `LONG` (45–120s), `EXCESSIVE` (>120s) |

Added a `_dwell_category()` helper function that classifies dwell duration at ML pipeline time, so the backend doesn't need to re-derive it from raw seconds.

**CSV before (8 columns):**

```
Split,ID,Class,Role,Start_Time_Sec,End_Time_Sec,Frame_Count,Zone
0,3,person,Cashier,0.0,59.95,1501,Cashier 2
```

**CSV after (13 columns):**

```
Split,ID,Class,Role,Start_Time_Sec,End_Time_Sec,Frame_Count,Zone,Camera_ID,Session_Start,Bbox_Start,Bbox_End,Dwell_Category
0,3,person,Cashier,0.0,59.95,1501,Cashier 2,CAM-01,2026-02-25T09:05:12.345678,641,25,803,197,623,19,770,250,LONG
```

#### Backend Models — `backend/models.py`

Added 4 new **optional** fields to `TrackingEvent`:

- `sessionStart` (str, default `None`)
- `bboxStart` (str, default `None`)
- `bboxEnd` (str, default `None`)
- `dwellCategory` (str, default `"NORMAL"`)

All optional with defaults → **old CSVs without these columns still work** (backward compatible).

#### CSV Adapter — `backend/services/csv_adapter.py`

- Added 4 new entries to `COLUMN_MAP` with flexible name matching (e.g., `session_start`, `session_ts`, `session_datetime` all map to `sessionStart`)
- Added corresponding defaults to `DEFAULTS` dict
- Updated `_load()` to parse new fields gracefully — missing columns get default values, no crashes

#### Aggregation Service — `backend/services/aggregations.py`

Replaced all hardcoded/fake values with computed ones:

| KPI | Before (fake) | After (computed) |
|---|---|---|
| `securityScore` | Hardcoded `94` | Starts at 100, -3 per `LONG` dwell, -8 per `EXCESSIVE` dwell |
| `uptime` | Hardcoded `99.9` | `(cameras_online / cameras_total) × 100` |
| Alert timestamps | Fake offsets from `_now()` | Real datetime from `Session_Start + Start_Time_Sec` |
| Employee `lastSeen` | Relative offset | Real datetime from `Session_Start + Start_Time_Sec` |
| Employee `status` | `ON_DUTY` if `endTimeSec > 30` (arbitrary) | `ON_DUTY` if `dwellCategory` is NORMAL/LONG/EXCESSIVE |
| Alert rules | Raw second thresholds (>45s, >55s) | Based on `dwellCategory` (LONG → UNUSUAL_MOTION, EXCESSIVE → LOITERING) |

Added `_session_time()` helper that computes real datetimes from `sessionStart + startTimeSec`, with fallback to the old offset method for backward compatibility.

---

## [0.2.0] — 2026-02-24 — Backend API Server

### Problem

The project had only a standalone ML pipeline script (`main.py`) that processed video and wrote CSV files. There was **no API layer** — no REST endpoints, no web server — so the UI dashboard being built by the frontend team had nothing to connect to.

### What Was Built

A complete **FastAPI backend** server with 7 REST endpoints + 1 WebSocket, designed around the analytics dashboard UI contract.

#### New Files Created

```
backend/
├── app.py                  # FastAPI entry point with CORS, static files, health check
├── config.py               # Environment-based settings, deterministic CSV discovery
├── models.py               # Pydantic response schemas for all endpoints
├── requirements.txt        # Backend dependencies (fastapi, uvicorn, websockets)
├── __init__.py
├── api/
│   ├── dashboard.py        # GET /api/v1/dashboard/summary — 6 KPI cards
│   ├── cameras.py          # GET /api/v1/cameras + /snapshots
│   ├── alerts.py           # GET /api/v1/alerts — alert feed with severity/zone/evidence
│   ├── employees.py        # GET /api/v1/employees — staff tracking
│   ├── live.py             # GET /api/v1/live/state — combined payload for polling
│   ├── ws.py               # WS /ws/live — real-time alert push with dedup
│   └── __init__.py
├── services/
│   ├── csv_adapter.py      # Reads tracking CSVs with tolerant column mapping
│   ├── aggregations.py     # Derives KPIs, alerts, employees from raw events
│   └── __init__.py
└── storage/
    └── media/
        ├── snapshots/      # Placeholder for camera frame snapshots
        └── evidence/       # Placeholder for alert evidence images
```

#### Architecture

```
ML Pipeline (main.py)  →  CSV files  →  CSV Adapter  →  Aggregation Service  →  API Endpoints  →  Frontend
```

**Key design principle:** Only `csv_adapter.py` needs to change when moving from CSV to database or RTSP streams. All API contracts remain identical.

#### API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/health` | Health check — server status + latest CSV path |
| `GET` | `/api/v1/dashboard/summary` | 6 KPI cards for the top of the dashboard |
| `GET` | `/api/v1/cameras` | Camera list (4 cameras with name, location, status) |
| `GET` | `/api/v1/cameras/snapshots` | Latest snapshot URL per camera |
| `GET` | `/api/v1/alerts?status=active&limit=20` | Alert feed with severity, zone, camera, evidence |
| `GET` | `/api/v1/employees?status=ON_DUTY` | Employee tracking (role, last seen, location) |
| `GET` | `/api/v1/live/state` | Everything combined in one call (for 5s polling) |
| `WS`  | `/ws/live` | Real-time alert push via WebSocket |

#### Key Design Decisions

1. **API versioning** (`/api/v1/`) — future changes won't break the existing frontend
2. **Env-based CORS** — `DEV=true` allows `*`, production requires explicit origins via `CORS_ORIGINS`
3. **Tolerant CSV parsing** — column mapping dict handles renames/missing columns gracefully
4. **WebSocket dedup** — tracks `last_row_count` to avoid re-sending old alerts
5. **`cameraId` on every model** — all events, alerts, and employees link to camera tiles
6. **Auto-reload** — `uvicorn --reload` restarts on code changes during development
7. **Auto-discover CSV** — picks the newest session folder automatically, refreshes on file change

#### How to Run

```bash
cd d:\Work\jwinfotech\Videoanalystics\video-analytics
.\.venv\Scripts\Activate
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs at **<http://localhost:8000/docs>** (Swagger UI).

#### README Updated

- Added full backend API documentation with endpoint table, JSON response shapes, frontend integration code snippets, module architecture, and environment variables
- Updated project structure to show both `backend/` and `pycode/` directories
- Added setup instructions for both ML pipeline and backend server

---

## [0.1.0] — 2026-01-23 to 2026-02-11 — ML Tracking Pipeline

### What Was Built

A complete object tracking pipeline for CCTV surveillance, ported from Google Colab to local Windows execution.

#### Phase 1: Base Tracking Stability

- YOLOv8 + ByteTrack for person/car detection and tracking
- Configured `track_buffer=120`, `conf=0.20`, `imgsz=960` for small object detection

#### Phase 2: Occlusion Handling & ID Re-linking

- Custom Occlusion Handler (`utils/occlusion_handler.py`) to prevent ID switching
- Adaptive thresholds, edge boost, dual-box sizing, and low IoU re-linking

#### Phase 3: Event Continuity

- Event Merger (`utils/event_merger.py`) — post-processing to fuse fragmented tracks
- Merges events with gaps up to 5 seconds based on spatial proximity

#### Role Classification

- Role Classifier (`utils/role_classifier.py`) — classifies persons as Cashier or Customer
- Based on zone occupancy within defined ROI areas
- Different visual colors for Cashier (cyan) vs Customer (yellow) in output video

#### ROI Zone Mapping

- ROI Manager (`utils/roi_mapping.py`) — defines and draws named zones on video frames
- Zones configurable via `config/zones.json`

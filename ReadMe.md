# Video Analytics & Surveillance System

This project is a video analytics platform designed for surveillance applications, starting with basic object tracking and evolving into specific retail use cases like cashier monitoring.

## 🚀 Current Features

### 1. Object Tracking Pipeline (`pycode/src/main.py`)

A complete pipeline ported from Google Colab to local Windows execution.

- **Function**: Processes CCTV footage to track objects (People, Cars) using YOLOv8.
- **Features**:
  - Trims video to a specified duration using OpenCV.
  - Detects and tracks objects with high persistence.
  - **Auto-Recover**: Handles temporary obstructions and re-links lost objects.
  - **Event Merge**: Post-processes data to fuse fragmented events into cohesive timelines.
  - **Role Classification**: Classifies tracked persons as Cashier or Customer based on zone occupancy.
  - **Cash Detection**: Detects banknotes/cash in frames and associates them with persons.
  - **Cash Event Tracking**: State machine tracking for cash pickups, deposits, handovers, and suspicious pocketing.
  - **Output**: Results saved to `output/logs/session_YYYYMMDD_HHMMSS/` (tracking CSV, cash events CSV, annotated video).

## 🧠 Development Phases & Logic

Since raw YOLO tracking often fails in real-world CCTV scenarios (occlusions, exits, low confidence), we implemented a multi-layer solution.

### Phase 1: Base Tracking Stability

**Goal**: Get a good baseline tracker.

- **Implementation**: Used **ByteTrack** instead of standard BoT-SORT.
- **Why?**: ByteTrack utilizes low-confidence detections (which other trackers discard) to maintain tracks when objects are partially obscured or blurry.
- **Tuning**: Configured `track_buffer=120`, `conf=0.20`, `imgsz=960` for small object detection.

### Phase 2: Occlusion Handling & ID Re-linking

**Goal**: Prevent "ID Switching" when a person walks behind a pillar or another person.

- **Problem**: If an object disappears for >1 second, standard trackers assign a new ID (e.g., ID 5 -> ID 12).
- **Solution**: We built a custom **Occlusion Handler** (`utils/occlusion_handler.py`).
  1. **Logical IDs**: We maintain a persistent "Logical ID" separate from the tracker's raw ID.
  2. **Adaptive Thresholds**: Re-linking distance is dynamic (`2.5x` the object's size). A large person close to the camera is allowed to move further pixels than a small person in the back.
  3. **Dual-Box Sizing**: We use the *maximum* dimensions of both the old (lost) and new (found) boxes to calculate thresholds, preventing mismatches if the object reappears partially clipped.
  4. **Edge Boost**: If an object disappears near the frame edge (6% margin), we expand the search radius by **1.5x** to account for faster/unpredictable re-entries.
  5. **Lower IoU**: We allow re-linking with IoU as low as **0.10** if the spatial position is plausible.

### Phase 3: Event Continuity (Post-Processing)

**Goal**: Merge tracks that were fragmented by long gaps (up to 5 seconds).

- **Problem**: Even with Phase 2, a 4-second occlusion or a massive detector failure causes the track to break.
- **Solution**: We implemented an **Event Merger** (`utils/event_merger.py`) that runs *after* tracking but *before* CSV export.
  1. **Offline Lookahead**: It scans the entire list of events to find fragments.
  2. **Best Match Algorithm**: It finds the best candidate to merge with based on spatial proximity.
  3. **Logic**:
      - **Merge**: If Class matches, Gap < 5.0s, and Velocity < 1000px/s (plausible movement).
      - **Reject**: If events time-overlap significantly (ensures distinct people remain distinct).
- **Result**: A clean CSV report where a person walking, disappearing for 4s, and reappearing is counted as **1 Unique Event** instead of 2.

### Cash Detection & Monitoring (6-Layer Architecture)

**Goal**: Detect cash handling and fraudulent activities in hotel reception / retail CCTV footage.

The system uses a highly robust **6-Layer Architecture** to eliminate false positives in complex transaction zones:

#### Layer 1 & 2: Person Tracking & Role Classification

Standard YOLOv8 + ByteTrack pipeline tracking customers and identifying Cashiers based on ROI occupancy.

#### Layer 3: Hand Detection (`pycode/utils/hand_detector.py`)

Uses **YOLOv8-pose** to detect human wrists (keypoints 9 & 10). It constantly measures the pixel distance between cashier hands and customer hands to detect physical interactions.

#### Layer 4: Cash Detection (`pycode/utils/cash_detector.py`)

- **Architecture**: YOLOv8m carefully fine-tuned on the `Hands in transaction.v4i.yolov8` dataset.
- **Performance**: Fantastic **mAP50 = 0.995** achieved via early stopping at epoch 12.
- **Model file**: `pycode/models/cash_detector_v2/train_v2/weights/best.pt`
- Runs parallel to hand tracking, looking specifically for banknotes in transit.

#### Layer 5: Timeline Fusion (`pycode/utils/interaction_analyzer.py`)

Fuses all signals from Layers 1-4 into a rolling time window.

- **Rules Engine**: Infers a `CASH_EXCHANGE` event if hands interact closely (< 90px) inside a designated transaction zone for over 1.0 seconds (if cash is seen) or 1.5 seconds (if cash is obscured).

#### Layer 6: Fraud Rules (`pycode/utils/fraud_detector.py`)

Evaluates the timeline of exchanges and cash appearances for suspicious behavior:

- `UNREGISTERED_CASH`: A cashier conducts an exchange but fails to visit the cash register within 10 seconds.
- `CASH_POCKETED`: A cashier is seen holding cash and then pocketing it outside the safe bounds of a register zone.

#### Output

- **Video**: Hand keypoints (blue/yellow), interaction lines between hands, and bounding boxes for cash drawn directly on the footage.
- **CSVs**:
  - `exchange_events.csv`: Logs all successful inferred or confirmed transactions.
  - `fraud_alerts.csv`: Logs all specific fraud layer violations.
  - `cash_events.csv`: Raw cash state machine logs (pickups, deposits).

#### Configuration

In `main.py`:

```python
ENABLE_CASH_DETECTION = True   # Toggle on/off
conf_threshold = 0.35          # YOLO confidence threshold
pickup_debounce = 5            # Frames before confirming cash pickup
deposit_debounce = 20          # Frames before confirming cash gone
```

#### Output

- **Video**: Cash bounding boxes drawn on output video (green = assigned to person, red = unassigned)
- **CSV**: `cash_events.csv` with columns: Event_Type, Person_ID, Timestamp_Sec, Frame_Idx, Zone, Confidence, Partner_ID, Bbox_Snapshot, Camera_ID, Session_Start

## 🛠️ Setup & Usage

### Prerequisites

- Python 3.10+
- Recommended: GPU with CUDA for the ML pipeline (CPU supported but slower).

### Installation

1. Clone the repository.
2. Create and activate virtual environment:

    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate      # Windows
    # source .venv/bin/activate   # Linux/macOS
    ```

3. Install ML pipeline dependencies:

    ```bash
    pip install ultralytics opencv-python tqdm torch torchvision
    ```

4. Install backend API dependencies:

    ```bash
    pip install -r backend/requirements.txt
    ```

### Running the ML Tracker

```bash
cd pycode/src
python main.py
```

Results saved to `output/logs/session_YYYYMMDD_HHMMSS/`.

### Running the Backend API Server

```bash
# From the project root directory:
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at **<http://localhost:8000>**. Interactive docs at **<http://localhost:8000/docs>**.

---

## 🖥️ Backend API (for Frontend Integration)

The backend is a **FastAPI** server that exposes all analytics data for the dashboard UI. It currently reads from the ML pipeline's tracking CSVs, but the adapter layer is designed to be swapped for a database or RTSP stream processor later without changing any API contracts.

### Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Data source**: Tracking CSVs from `output/logs/` (auto-discovers latest session)
- **Real-time**: WebSocket at `/ws/live`
- **CORS**: Enabled for dev (`*`), configurable via `CORS_ORIGINS` env var

### API Endpoints (all versioned under `/api/v1`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns server status & latest CSV path |
| `GET` | `/api/v1/dashboard/summary` | 6 KPI cards (cameras online, alerts, staff, score, incidents, uptime) |
| `GET` | `/api/v1/cameras` | Camera list with name, location, status, stream URL |
| `GET` | `/api/v1/cameras/snapshots` | Latest snapshot URL per camera |
| `GET` | `/api/v1/alerts?status=active&limit=20` | Alert feed with severity, zone, camera, evidence |
| `GET` | `/api/v1/employees?status=ON_DUTY` | Employee tracking (role, last seen, location, zone) |
| `GET` | `/api/v1/live/state` | **Combined payload** — summary + snapshots + alerts + employees in one call |
| `WS`  | `/ws/live` | Real-time alert push (WebSocket) |

### Response Shapes

#### `GET /health`

```json
{
  "status": "ok",
  "ts": "2026-02-24T06:21:58Z",
  "dataSource": "csv",
  "latestCsv": "D:\\...\\session_20260211_111843\\tracking_summary.csv"
}
```

#### `GET /api/v1/dashboard/summary`

```json
{
  "ts": "2026-02-24T06:22:05Z",
  "cards": {
    "camerasOnline": { "value": 3, "total": 4 },
    "activeAlerts":  { "value": 5, "deltaPct": -12.0 },
    "staffOnSite":   { "value": 2 },
    "securityScore": { "value": 94, "deltaPct": 3.0 },
    "incidentsToday":{ "value": 5 },
    "uptime":        { "value": 99.9 }
  }
}
```

#### `GET /api/v1/cameras`

```json
[
  { "id": "CAM-01", "name": "Main Lobby", "location": "Ground Floor - Entrance", "status": "online", "streamUrl": "rtsp://..." },
  { "id": "CAM-02", "name": "Parking Lot A", "location": "Outdoor - North Side", "status": "online", "streamUrl": "rtsp://..." }
]
```

#### `GET /api/v1/alerts`

```json
{
  "items": [
    {
      "id": "AL-23D901",
      "type": "UNUSUAL_MOTION",
      "severity": "MEDIUM",
      "title": "Extended Presence Detected",
      "description": "Customer (Track #1) stayed in Outside for 59.9s.",
      "cameraId": "CAM-01",
      "zone": "Outside",
      "ts": "2026-02-24T06:22:15Z",
      "evidence": { "imageUrl": null, "clipUrl": null }
    }
  ]
}
```

#### `GET /api/v1/employees`

```json
{
  "items": [
    {
      "id": "E003",
      "name": "Sarah Johnson",
      "role": "Cashier",
      "status": "ON_DUTY",
      "lastSeen": "2026-02-24T06:22:20Z",
      "location": "POS Station 2",
      "zone": "Cashier 2"
    }
  ]
}
```

#### `WS /ws/live` (WebSocket messages)

```json
{ "kind": "alert_new", "payload": { "id": "AL-...", "type": "UNUSUAL_MOTION", ... } }
{ "kind": "summary_update", "payload": { "newEvents": 3 } }
```

### Frontend Integration Quick Start

**Option A — Polling** (simplest, no WebSocket needed):

```javascript
const API = "http://<backend-ip>:8000/api/v1";

// Fetch everything in one call, poll every 5s
setInterval(async () => {
  const res = await fetch(`${API}/live/state`);
  const data = await res.json();
  // data.summary, data.alerts, data.employees, data.snapshots
}, 5000);
```

**Option B — WebSocket** (real-time alerts):

```javascript
const ws = new WebSocket("ws://<backend-ip>:8000/ws/live");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.kind === "alert_new") {
    // Push to alert panel
  }
};
```

### Backend Module Architecture

| Module | File | Purpose |
|--------|------|---------|
| **Config** | `backend/config.py` | Env-based settings, CORS, CSV discovery |
| **Models** | `backend/models.py` | Pydantic schemas (all include `cameraId`) |
| **CSV Adapter** | `backend/services/csv_adapter.py` | Reads & normalizes tracking CSVs (swap this for DB later) |
| **Aggregations** | `backend/services/aggregations.py` | Derives KPIs, alerts, employee status from raw events |
| **API Routers** | `backend/api/*.py` | One router file per endpoint group |
| **App Entry** | `backend/app.py` | FastAPI app with CORS, static files, router registration |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEV` | `true` | Dev mode enables CORS `*` and debug logging |
| `CORS_ORIGINS` | `*` (dev) | Comma-separated allowed origins for production |
| `CSV_BASE_DIR` | `output/logs` | Where to find tracking session CSVs |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |

---

## ⚠️ Known Issues & Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Cash recall ~62%** | Model misses some cash, especially when partially occluded or very small | Increased deposit_debounce (20 frames) keeps tracking through gaps; can improve with more training data |
| **Detection flicker** | Model doesn't detect cash every frame even when visible in hands | High deposit_debounce (20 frames = ~0.8s) prevents tracker state from resetting during brief gaps |
| **Counter object FPs** | Static objects (monitors, signs) on counters can resemble cash | Rule 2 requires person within 250px alongside zone match; geometric filters reject wrong shapes |
| **Dataset imbalance** | Only 145 cash vs 783 person instances in test set | Model trained with class-weighted loss; additional hotel-specific cash data would improve recall |
| **Single camera** | Pipeline currently processes one camera at a time | `CAMERA_ID` field is embedded in outputs; multi-camera orchestration planned |

## 📂 Project Structure

```text
video-analytics/
├── backend/                           # Dashboard API server
│   ├── app.py                         # FastAPI entry point
│   ├── config.py                      # Environment-based settings
│   ├── models.py                      # Pydantic response schemas
│   ├── requirements.txt               # Backend Python dependencies
│   ├── api/
│   │   ├── dashboard.py               # GET /api/v1/dashboard/summary
│   │   ├── cameras.py                 # GET /api/v1/cameras, /snapshots
│   │   ├── alerts.py                  # GET /api/v1/alerts
│   │   ├── employees.py               # GET /api/v1/employees
│   │   ├── live.py                    # GET /api/v1/live/state
│   │   └── ws.py                      # WS  /ws/live
│   ├── services/
│   │   ├── csv_adapter.py             # CSV reader (swap for DB later)
│   │   └── aggregations.py            # KPI/alert/employee derivation
│   └── storage/
│       └── media/                     # Static snapshots & evidence
├── output/                            # ML pipeline output (CSVs, videos)
├── pycode/
│   ├── config/
│   │   └── zones.json                 # ROI zone definitions
│   ├── models/
│   │   └── cash_detector/             # Trained cash detection model
│   │       ├── weights/best.pt        # YOLOv8m fine-tuned weights
│   │       ├── results.csv            # Training metrics
│   │       └── confusion_matrix.png   # Validation confusion matrix
│   ├── src/
│   │   └── main.py                    # ML tracking + cash detection pipeline
│   ├── test/
│   │   ├── test_cash_detector.py      # Cash detector unit tests (16 tests)
│   │   └── test_cash_tracker.py       # Cash tracker state machine tests
│   └── utils/
│       ├── cash_detector.py           # Cash detection + context-aware filtering
│       ├── cash_tracker.py            # Cash event state machine
│       ├── occlusion_handler.py       # Phase 2: occlusion handling
│       ├── event_merger.py            # Phase 3: event continuity
│       ├── roi_mapping.py             # ROI zone manager
│       ├── role_classifier.py         # Cashier/Customer classification
│       └── tracker/
│           └── bytetrack_cctv.yaml    # ByteTrack config for CCTV
├── resources/
│   ├── videos/                        # Input raw CCTV footage
│   └── datasets/
│       └── cash.v7i.yolov8/           # Cash detection training dataset
├── CHANGELOG.md                       # Version history
├── performancelog.md                  # Model performance metrics
└── ReadMe.md                          # This file
```

## 📚 References & Tutorials

- [How to Implement ByteTrack](https://www.labellerr.com/blog/how-to-implement-bytetrack/)
- [Object Detection & Tracking using ByteTrack](https://medium.com/tech-blogs-by-nest-digital/object-tracking-object-detection-tracking-using-bytetrack-0aafe924d292)
- [What is ByteTrack? (Roboflow)](https://blog.roboflow.com/what-is-bytetrack-computer-vision/)
- [Introduction to ByteTrack (Datature)](https://datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics Detection Tasks](https://docs.ultralytics.com/tasks/detect/)

# Video Analytics & Surveillance System

This project is a video analytics platform designed for surveillance applications, starting with basic object tracking and evolving into specific retail use cases like cashier monitoring.

## 🚀 Current Features

### 1. Object Tracking Pipeline (`pycode/src/main.py`)

A complete pipeline ported from Google Colab to local Windows execution.

- **Function**: Processes CCTV footage to track objects (People, Cars) using YOLOv8.
- **Features**:
  - Trims video to a specified duration (e.g., 30s) using OpenCV.
  - Detects and tracks objects with high persistence.
  - **Auto-Recover**: Handles temporary obstructions and re-links lost objects.
  - **Event Merge**: Post-processes data to fuse fragmented events into cohesive timelines.
  - **Output**: Results are saved to `output/logs/session_YYYYMMDD_HHMMSS/`.

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

## 📅 Roadmap: Cashier Surveillance Module

We are currently planning a specialized module for **Retail POS Surveillance**.

**Objective**: Detect anomalies where a POS "Cash Sale" event occurs without the cashier physically interacting with the cash drawer.

**Architecture**:

- **Vision Engine**: Monitors a Region of Interest (ROI) for "Drawer Open" or "Hand in Drawer" events.
- **POS Listener**: Receives transaction logs from the POS system.
- **Logic Core**: Correlates Vision events with POS timestamps to flag suspicious behavior.

## 📂 Project Structure

```
video-analytics/
├── backend/                        # Dashboard API server
│   ├── app.py                      # FastAPI entry point
│   ├── config.py                   # Environment-based settings
│   ├── models.py                   # Pydantic response schemas
│   ├── requirements.txt            # Backend Python dependencies
│   ├── api/
│   │   ├── dashboard.py            # GET /api/v1/dashboard/summary
│   │   ├── cameras.py              # GET /api/v1/cameras, /snapshots
│   │   ├── alerts.py               # GET /api/v1/alerts
│   │   ├── employees.py            # GET /api/v1/employees
│   │   ├── live.py                 # GET /api/v1/live/state
│   │   └── ws.py                   # WS  /ws/live
│   ├── services/
│   │   ├── csv_adapter.py          # CSV reader (swap for DB later)
│   │   └── aggregations.py         # KPI/alert/employee derivation
│   └── storage/
│       └── media/                  # Static snapshots & evidence
├── output/                         # ML pipeline output (CSVs, videos)
├── pycode/
│   ├── src/
│   │   └── main.py                 # ML tracking pipeline
│   └── utils/
│       ├── occlusion_handler.py    # Phase 2: occlusion handling
│       └── event_merger.py         # Phase 3: event continuity
├── resources/
│   └── videos/                     # Input raw CCTV footage
└── README.md
```

## 📚 References & Tutorials

- [How to Implement ByteTrack](https://www.labellerr.com/blog/how-to-implement-bytetrack/)
- [Object Detection & Tracking using ByteTrack](https://medium.com/tech-blogs-by-nest-digital/object-tracking-object-detection-tracking-using-bytetrack-0aafe924d292)
- [What is ByteTrack? (Roboflow)](https://blog.roboflow.com/what-is-bytetrack-computer-vision/)
- [Introduction to ByteTrack (Datature)](https://datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics Detection Tasks](https://docs.ultralytics.com/tasks/detect/)

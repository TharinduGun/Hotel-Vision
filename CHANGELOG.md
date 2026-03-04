# Changelog

All notable changes to the Video Analytics & Surveillance System are documented here.

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

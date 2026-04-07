# Parking Lot Video Analysis System

A real-time CCTV-based parking lot occupancy analysis system built for hotel and cafe deployments. It processes video feeds to detect vehicles, track occupancy, measure dwell times, and log structured analytics — with a dashboard integration layer ready to connect.
---

## What It Does

- Detects vehicles in parking lot footage using YOLOv8
- Tracks each vehicle with a stable ID across frames
- Reports how many spaces are occupied, free, and what percentage full the lot is
- Tracks how long each vehicle has been parked (dwell time)
- Monitors individually defined parking bays and reports each as occupied or free
- Counts vehicles entering through a defined entry line
- Logs all analytics to a CSV file every second
- Works on local video files for testing and live RTSP camera streams for deployment
- Automatically adapts detection sensitivity for night and low-light conditions

---

## Project Structure
parking-lot-system/
│
├── main.py                         # Entry point
├── requirements.txt
│
├── configs/
│   ├── config.yaml                 # All settings — camera, capacity, thresholds
│   └── parking_spaces.py           # Parking zone polygons and entry line
│
├── core/
│   ├── pipeline.py                 # Main processing pipeline
│   └── video_loader.py             # Video file and RTSP stream handler
│
├── models/
│   └── detector.py                 # YOLOv8 vehicle detection with tiled inference
│
├── analytics/
│   ├── tracker.py                  # IoU-based vehicle tracker
│   ├── occupancy_calculator.py     # Occupied/available/percentage metrics
│   ├── dwell_tracker.py            # Per-vehicle parking duration
│   ├── parking_space_analyzer.py   # Per-bay occupied/free status
│   ├── vehicle_stats.py            # Vehicle type breakdown
│   ├── line_counter.py             # Entry gate vehicle counter
│   └── json_exporter.py            # JSON payload builder for dashboard
│
├── utils/
│   ├── csv_logger.py               # Structured CSV output
│   └── visualizer.py               # Bounding box and overlay drawing
│
├── services/
│   └── websocket_server.py         # WebSocket server (ready, not yet connected)
│
├── tools/
│   └── space_annotator.py          # Click-to-annotate parking zones tool
│
└── data/
├── videos/                     # Input video files (not tracked in git)
└── outputs/                    # CSV logs and system log

---

## Setup

**Install dependencies**

**Configure the system**

Edit `configs/config.yaml`:
```yaml
video:
  source: "data/videos/your_video.mp4"   # or rtsp://192.168.1.x/stream

parking:
  total_spaces: 20

detection:
  model: "yolov8n.pt"
  conf_day: 0.25
  conf_night: 0.20

logging:
  level: "INFO"
  file: "data/outputs/system.log"

output:
  csv_path: "data/outputs/parking_data.csv"
```

**Run**
```bash
python main.py
```
Press `ESC` to stop. Output is saved to `data/outputs/parking_data.csv`.
---

## Annotating Parking Zones

Before running on a new camera, define the parking zones using the annotator tool:
```bash
python tools/space_annotator.py --video data/videos/your_video.mp4
```

**Controls:**
- Left-click — add a corner point to the current zone
- Enter — finish the zone and name it
- L — switch to line mode (click 2 points to define the entry line)
- R — clear current points and start again
- ESC — save all zones to `configs/parking_spaces.py` and exit

Zones are optional. If none are defined, the system still counts overall occupancy across the whole frame.

---

## Output Format

Analytics are written to CSV every second:

| Column | Description |
|---|---|
| timestamp | Unix timestamp |
| occupied | Number of spaces currently occupied |
| available | Number of spaces currently free |
| capacity | Total configured capacity |
| occupancy_pct | Percentage full |
| status | `available` / `limited` / `full` |
| avg_dwell_seconds | Average time vehicles have been parked |
| cars / buses / trucks / motorcycles | Count per vehicle type |
| entry_count | Cumulative vehicles counted at entry line |

---

## Detection Approach

**Tiled inference** — each frame is split into overlapping sections and YOLO runs on each section separately before results are merged. This solves the common problem of closely parked vehicles being detected as a single object.

**CLAHE preprocessing** — contrast is enhanced before inference so vehicles on dark tarmac, particularly under shadow or at night, are more clearly visible to the model.

**Adaptive confidence** — the detection threshold is automatically lowered in low-light conditions so vehicles are not missed at night.

**Stable tracking** — each vehicle is assigned an ID that survives brief detection gaps (shadows, occlusion, a person walking past) without resetting, keeping dwell times and occupancy counts accurate.

---

## Connecting the Dashboard

The WebSocket server is written and ready. To activate it:

1. In `core/pipeline.py`, uncomment:
```python
   from services.websocket_server import update_data
```
   and:
```python
   update_data(analytics)
```

2. Run the WebSocket server alongside the main process:
```bash
   uvicorn services.websocket_server:app --host 0.0.0.0 --port 8000
```

3. Connect your dashboard to `ws://localhost:8000/ws` — it will receive a live JSON analytics payload every second.

---

## Camera Requirements

The system works best with:
- Fixed overhead or angled CCTV cameras
- Minimum 720p resolution
- Stable mounting with a clear view of all parking spaces
- For 24/7 deployment: cameras with IR night vision

For live deployment, set `video.source` in `config.yaml` to the RTSP URL of your camera. The system will automatically reconnect if the stream drops.
---

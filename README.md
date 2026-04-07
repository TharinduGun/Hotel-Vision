# AI Video Analytics – Coffee Shop Monitoring System

This project implements an **AI-based video analytics system** for monitoring operational activities in a retail or coffee shop environment using CCTV footage.

The system analyzes video streams to detect and track people, identify employees, monitor queue conditions, and detect idle staff behavior. It generates events that are streamed to a backend dashboard for real-time operational insights.

The architecture is designed to support **multiple cameras and real-time analytics**.

## Key Features

- Person detection using **YOLOv8**
- Person tracking using **ByteTrack**
- Face recognition using **InsightFace**
- Employee identification
- Zone-based activity monitoring
- Queue length estimation
- Idle staff detection
- Multi-camera processing
- Real-time event streaming via **WebSocket**
- Dashboard integration
- CSV event logging for auditing

---

## System Architecture

RTSP Cameras / Video Files
↓
Pipeline Manager
↓
Camera Pipelines
↓
YOLOv8 Person Detection
↓
ByteTrack Person Tracking
↓
Face Recognition (InsightFace)
↓
Zone Analysis (zones.json)
↓
Behavior Analytics
• Queue Monitoring
• Idle Detection
↓
Event Builder
↓
WebSocket + CSV
↓
Backend API
↓
Dashboard UI

``
## Running the Real-Time Pipeline

Configure cameras in:
configs/cameras.json
Example:

```json
{
  "cameras": [
    {"id": "CAM01", "source": "data/videos/test_video.mp4"},
    {"id": "CAM02", "source": "rtsp://192.168.1.20/live"}
  ]
}
```
## Adding Employee Data

Place employee images inside:
data/employees/

Then build the embedding database:
python build_employee_db.py
These embeddings will be used for face recognition.

## Zone Configuration
Operational zones are defined in:
configs/zones.json
Example:

```json
{
  "zones": {
    "counter": [[300,200],[600,200],[600,450],[300,450]],
    "queue": [[100,250],[280,250],[280,500],[100,500]]
  }
}

can use the provided tool:
tools/zone_selector.py
to visually select zone coordinates from video.
## Event Generation

The analytics pipeline generates structured events:

```json
{
  "timestamp": "2026-03-12T10:15:20",
  "employee_id": "E001",
  "camera_id": "CAM01",
  "zone": "counter",
  "status": "ON_DUTY"
}
```

Events are sent via:
WebSocket → ws://localhost:8000/ws/events
data/events/events.csv

## Backend Integration

The system connects to a backend API which:

* aggregates events
* provides operational insights
* powers the management dashboard

The backend consumes events through **WebSocket streaming**.

## Data Privacy
this repo doesnt include:
* employee face images
* CCTV videos
* trained model weights
* embeddings







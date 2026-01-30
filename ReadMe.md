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

- Python 3.8+
- Recommended: GPU with CUDA (though CPU is supported).

### Installation

1. Clone the repository.
2. Install dependencies:

    ```bash
    pip install ultralytics opencv-python tqdm torch torchvision
    ```

    (Note: `torch` installation may vary based on your CUDA version).

### Running the Tracker

1. Navigate to the source directory:

    ```bash
    cd pycode/src
    ```

2. Run the main script:

    ```bash
    python main.py
    ```

3. Check `output/logs/` for the results.

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
├── output/             # Generated logs, videos, and CSVs
├── pycode/
│   ├── src/
│   │   ├── main.py     # Main tracking + Phase 2/3 Integration
│   ├── utils/
│   │   ├── occlusion_handler.py # Phase 2 Logic
│   │   ├── event_merger.py      # Phase 3 Logic
├── resources/
│   ├── videos/         # Input raw footage
└── README.md
```

## 📚 References & Tutorials

Useful resources,
project links:

- [How to Implement ByteTrack](https://www.labellerr.com/blog/how-to-implement-bytetrack/)
- [Object Detection & Tracking using ByteTrack](https://medium.com/tech-blogs-by-nest-digital/object-tracking-object-detection-tracking-using-bytetrack-0aafe924d292)
- [What is ByteTrack? (Roboflow)](https://blog.roboflow.com/what-is-bytetrack-computer-vision/)
- [Introduction to ByteTrack (Datature)](https://datature.io/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box)
- [Labellerr Blog](https://www.labellerr.com/blog/untitled/)
- [Ultralytics Detection Tasks](https://docs.ultralytics.com/tasks/detect/)

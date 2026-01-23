# Video Analytics & Surveillance System

This project is a video analytics platform designed for surveillance applications, starting with basic object tracking and evolving into specific retail use cases like cashier monitoring.

## 🚀 Current Features

### 1. Object Tracking Pipeline (`pycode/src/main.py`)

A complete pipeline ported from Google Colab to local Windows execution.

- **Function**: Processes CCTV footage to track objects (People, Cars) using YOLOv8.
- **Features**:
  - Trims video to a specified duration (e.g., 30s) using OpenCV.
  - Detects and tracks objects.
  - **Phase 1: Tracking Stability**: Uses Explicit ByteTrack with tuned parameters (`track_buffer=150`, `conf=0.20`, `imgsz=960`) to handle partial visibility.
  - **Phase 2: Occlusion Handling**: Implements a custom **ID Re-linking Layer** (`utils/occlusion_handler.py`).
    - Maintains a buffer of "lost" tracks.
    - If a new ID appears, checks against lost tracks using IoU and Class ID.
    - Restores the original "Logical ID" if a match is found (e.g., person walks behind a shelf).
  - Filters recurring objects based on persistence.
  - unique "Events" logged to CSV.
  - Generates an annotated output video.
- **Output**: Results are saved to `output/logs/session_YYYYMMDD_HHMMSS/`.

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
│   │   ├── main.py     # Main detailed tracking script
│   ├── utils/
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

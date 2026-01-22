import cv2
import os
import torch
import gc
import shutil
import subprocess
import csv
from datetime import datetime
from ultralytics import YOLO
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from utils.occlusion_handler import OcclusionHandler


# ----------------------------
# CONFIG (edit these)
# ----------------------------
VIDEO_PATH = os.path.abspath(r"D:\Work\jwinfotech\Videoanalystics\video-analytics\resources\videos\Indoor_Original_VideoStream.mp4")   # <-- put your CCTV video here
MODEL_PATH = "yolov8m.pt"                             # use yolov8n.pt if CPU is slow
RUN_SECONDS = 30                                      # <-- only process first 30 sec
CLASSES = [0, 2]                                      # 0=person, 2=car
MIN_PERSISTENCE = 60                                  # in frames (same as your notebook)


def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()



def trim_video_cv2(source_path, dest_path, max_seconds):
    """Trims the first max_seconds of the video using OpenCV."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    out = cv2.VideoWriter(dest_path, fourcc, fps, (width, height))
    
    max_frames = int(fps * max_seconds)
    count = 0
    
    print(f"Trimming video: {source_path} -> {dest_path} ({max_seconds}s)")
    with tqdm(total=max_frames, desc="Trimming") as pbar:
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            count += 1
            pbar.update(1)
            
    cap.release()
    out.release()
    print("Trimming complete.")

def concat_videos_cv2(video_paths, output_path):
    """Concatenates multiple videos using OpenCV."""
    if not video_paths:
        return

    # Use parameters from the first video
    cap = cv2.VideoCapture(video_paths[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Concatenating {len(video_paths)} videos...")
    for v_path in video_paths:
        cap = cv2.VideoCapture(v_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    
    out.release()
    print(f"Concatenation complete: {output_path}")

def main():
    # --- 0. DIAGNOSTIC ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not found. This may be slow on CPU. Try yolov8n.pt")

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # --- 1. DYNAMIC SESSION SETUP ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Output to ../../output/logs relative to this script
    base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "logs"))
    session_dir = os.path.join(base_output_dir, f"session_{timestamp}")

    raw_dir = os.path.join(session_dir, "raw_splits")
    processed_dir = os.path.join(session_dir, "processed_splits")
    final_video = os.path.join(session_dir, "final_tracked_output.mp4")
    summary_csv = os.path.join(session_dir, "tracking_summary.csv")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # --- 2. STEP 1: CUT ONLY FIRST 30 SECONDS ---
    short_clip = os.path.join(raw_dir, "split_000.mp4")

    print(f"\n--- Cutting first {RUN_SECONDS} seconds ---")
    # REPLACED ffmpeg with cv2 implementation
    trim_video_cv2(VIDEO_PATH, short_clip, RUN_SECONDS)

    split_files = [short_clip]
    split_seconds = RUN_SECONDS  # for global time calculation

    # --- 3. STEP 2: TRACKING & GLOBAL LOGGING ---
    model = YOLO(MODEL_PATH).to(device)
    
    # Initialize Occlusion Handler (Phase 2)
    occlusion_handler = OcclusionHandler(max_age_seconds=3.0)

    # Structure: {(split_idx, obj_id): {...}}
    summary_data = {}
    
    # Store persistent mapping for "New ID" -> "Old ID"
    relink_map = {} 

    for s_idx, s_path in enumerate(split_files):
        print(f"\nProcessing Part {s_idx+1}/{len(split_files)}...")

        cap = cv2.VideoCapture(s_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-3:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Reset relink map per file
        relink_map = {} 

        save_path = os.path.join(processed_dir, f"tracked_{os.path.basename(s_path)}")
        out_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        with torch.no_grad():
            results = model.track(
                source=s_path,
                persist=True,
                classes=CLASSES,
                tracker=os.path.join("..", "utils", "tracker", "bytetrack_cctv.yaml"),  # Explicit tracker config
                conf=0.20,  # Lower confidence to detect partially occluded objects
                iou=0.5,    # IOU threshold for NMS
                imgsz=960,  # Increase resolution for small object detection
                stream=True,
                verbose=False
            )
            
            # Store tracks from previous frame to detect lost IDs (using LOGICAL IDs)
            prev_frame_tracks = {} 
            seen_ids_in_split = set()

            for frame_idx, r in enumerate(tqdm(results, total=total_frames, desc="Tracking")):
                # Plot returns BGR numpy array
                frame_bgr = r.plot()
                out_writer.write(frame_bgr)
                
                curr_frame_tracks = {}
                current_time = (s_idx * split_seconds) + (frame_idx / fps)

                # 1. Process current frame tracks
                if r.boxes.id is not None:
                    ids = r.boxes.id.int().cpu().tolist()
                    cls = r.boxes.cls.int().cpu().tolist()
                    boxes = r.boxes.xyxy.cpu().tolist()

                    for tid, c, box in zip(ids, cls, boxes):
                        # --- PHASE 2.2: ID RE-LINKING ---
                        logical_tid = tid
                        
                        # A. Check existing mapping
                        if tid in relink_map:
                            logical_tid = relink_map[tid]
                        elif tid not in seen_ids_in_split:
                            # B. Newly appearing ID - Try to relink
                            relinked_old_id = occlusion_handler.try_relink(
                                new_track_bbox=box,
                                new_cls_id=c,
                                current_time=current_time
                            )
                            if relinked_old_id:
                                print(f" [Link] ID {tid} -> {relinked_old_id}")
                                relink_map[tid] = relinked_old_id
                                logical_tid = relinked_old_id
                        
                        seen_ids_in_split.add(tid)
                        
                        # Store current track state using LOGICAL ID
                        curr_frame_tracks[logical_tid] = {'bbox': box, 'cls': c}
                        
                        # LOGGING
                        global_now = current_time
                        key = (s_idx, logical_tid)

                        if key not in summary_data:
                            summary_data[key] = {
                                "Split": s_idx,
                                "ID": logical_tid,
                                "Class": "person" if c == 0 else "car",
                                "Start_Time_Sec": round(global_now, 2),
                                "End_Time_Sec": round(global_now, 2),
                                "Frame_Count": 1
                            }
                        else:
                            summary_data[key]["End_Time_Sec"] = round(global_now, 2)
                            summary_data[key]["Frame_Count"] += 1
                        
                # 2. Detect Lost Tracks (Present in Prev, Missing in Curr)
                for pid, pdata in prev_frame_tracks.items():
                    if pid not in curr_frame_tracks:
                        # logical ID 'pid' was here last frame, gone now. Buffer it.
                        occlusion_handler.add_lost_track(
                            track_id=pid,
                            bbox=pdata['bbox'],
                            cls_id=pdata['cls'],
                            timestamp=current_time
                        )

                # 3. Clean up old lost tracks
                occlusion_handler.cleanup(current_time)
                
                # 4. Update previous frame memory
                prev_frame_tracks = curr_frame_tracks.copy()

        out_writer.release()
        flush_memory()

    # --- 4. STEP 3: DATA FILTERING & EXPORT ---
    print("\n--- Generating CSV Report ---")

    final_report_rows = [
        row for row in summary_data.values()
        if row["Frame_Count"] >= MIN_PERSISTENCE or row["Class"] == "car"
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Split", "ID", "Class", "Start_Time_Sec", "End_Time_Sec", "Frame_Count"]
        )
        writer.writeheader()
        writer.writerows(final_report_rows)

    # --- 5. STEP 4: VIDEO MERGE ---
    # REPLACED ffmpeg with cv2 implementation
    processed_files = sorted([os.path.join(processed_dir, p) for p in os.listdir(processed_dir)])
    concat_videos_cv2(processed_files, final_video)

    # --- 6. CLEANUP (same as notebook) ---
    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(processed_dir, ignore_errors=True)

    print("\n" + "=" * 50)
    print(f"SESSION COMPLETE: {session_dir}")
    print(f"Summary Report Created: {os.path.basename(summary_csv)}")
    print(f"Final Video Created   : {os.path.basename(final_video)}")
    print(f"Total Unique Events Logged: {len(final_report_rows)}")
    print("=" * 50)


if __name__ == "__main__":
    main()

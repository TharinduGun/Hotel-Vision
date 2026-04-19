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
from utils.roi_mapping import ROIManager
from utils.role_classifier import RoleClassifier
from utils.cash_detector import CashDetector
from utils.cash_tracker import CashTracker, CashEventType
from utils.hand_detector import HandDetector
from utils.interaction_analyzer import InteractionAnalyzer
from utils.fraud_detector import FraudDetector


def _dwell_category(duration_sec: float) -> str:
    """Classify dwell duration into a category for KPI alignment."""
    if duration_sec < 10:
        return "SHORT"
    elif duration_sec < 45:
        return "NORMAL"
    elif duration_sec < 120:
        return "LONG"
    else:
        return "EXCESSIVE"


# ----------------------------
# CONFIG (edit these)
# ----------------------------
VIDEO_PATH = os.path.abspath(r"D:\Work\jwinfotech\Videoanalystics\video-analytics\resources\videos\Indoor_Original_VideoStream.mp4")   # <-- put your CCTV video here
MODEL_PATH = "yolov8m.pt"                             # use yolov8n.pt if CPU is slow
CASH_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "models", "cash_detector_v2", "train_v2", "weights", "best.pt"
))  # Trained cash detection model
RUN_SECONDS = 1800                                      # <-- only process first 30 sec
CLASSES = [0, 2]                                      # 0=person, 2=car
MIN_PERSISTENCE = 60                                  # in frames (same as your notebook)
ENABLE_CASH_DETECTION = True                          # Toggle cash detection on/off


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
    session_start_dt = datetime.now()
    timestamp = session_start_dt.strftime("%Y%m%d_%H%M%S")
    session_start_iso = session_start_dt.isoformat()
    # Output to ../../output/logs relative to this script
    base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "logs"))
    session_dir = os.path.join(base_output_dir, f"session_{timestamp}")

    # Default camera ID — single camera for now, multi-camera will pass per-stream
    CAMERA_ID = "CAM-01"

    raw_dir = os.path.join(session_dir, "raw_splits")
    processed_dir = os.path.join(session_dir, "processed_splits")
    final_video = os.path.join(session_dir, "final_tracked_output.mp4")
    summary_csv = os.path.join(session_dir, "tracking_summary.csv")
    fraud_alerts_csv = os.path.join(session_dir, "fraud_alerts.csv")
    exchange_events_csv = os.path.join(session_dir, "exchange_events.csv")

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
    
    # ROI Manager will be initialized after we know the video frame size
    roi_manager = None
    
    # Role Classifier (Cashier vs Customer)
    role_classifier = RoleClassifier(cashier_threshold=0.60)
    
    # Cash Detection & Tracking (Phase: Cash Handling)
    cash_detector = None
    cash_tracker = None
    hand_detector = None
    interaction_analyzer = None
    fraud_detector = None
    
    all_fraud_alerts = []
    all_exchange_events = []
    
    if ENABLE_CASH_DETECTION:
        if os.path.exists(CASH_MODEL_PATH):
            cash_detector = CashDetector(
                model_path=CASH_MODEL_PATH,
                conf_threshold=0.25, # Was 0.35, lowered to catch blurry/distant cash
                device=device,
            )
            # More forgiving tracking parameters
            cash_tracker = CashTracker(
                pickup_debounce=2,    # Was 5
                deposit_debounce=15,  # Was 20
                zone_alert_cooldown=30,
            )
            
            # Initialize New Advanced Layers
            hand_detector = HandDetector(device=device)
            interaction_analyzer = InteractionAnalyzer(fps=fps if 'fps' in locals() and fps else 25.0)
            fraud_detector = FraudDetector(fps=fps if 'fps' in locals() and fps else 25.0)
            
            print("[Main] Cash detection ENABLED with Advanced Interaction Analysis")
        else:
            print(f"[Main] Cash model not found at {CASH_MODEL_PATH} — cash detection DISABLED")
            print("[Main] Run pycode/scripts/train_cash_detector.py to train the model.")

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
        
        # Initialize ROI Manager on first split (now we know the video resolution)
        if roi_manager is None:
            roi_manager = ROIManager(frame_size=(width, height))
        
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
            
            # Persistent "Latest State" for every logical ID seen in this split
            # This helps if a track flickers (dropped by detector for 1 frame, then reappears next)
            last_seen = {}

            for frame_idx, r in enumerate(tqdm(results, total=total_frames, desc="Tracking")):
                # Plot returns BGR numpy array
                # frame_bgr = r.plot()  <-- REMOVED: We do custom drawing below
                # out_writer.write(frame_bgr) <-- REMOVED
                
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
                                current_time=current_time,
                                frame_size=(width, height)
                            )
                            if relinked_old_id:
                                print(f" [Link] ID {tid} -> {relinked_old_id}")
                                relink_map[tid] = relinked_old_id
                                logical_tid = relinked_old_id
                        
                        seen_ids_in_split.add(tid)
                        
                        # Store current track state using LOGICAL ID
                        curr_frame_tracks[logical_tid] = {'bbox': box, 'cls': c}
                        
                        # Update Last Seen (Master Registry)
                        last_seen[logical_tid] = {
                            'bbox': box, 
                            'cls': c, 
                            'time': current_time
                        }
                        
                        # LOGGING
                        global_now = current_time
                        key = (s_idx, logical_tid)
                        
                        # Calculate center for post-processing merge
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        
                        # ROI Zone Detection (with type for role classification)
                        if roi_manager.has_zones:
                            zone_name, zone_type = roi_manager.get_zone_with_type(cx, cy)
                        else:
                            zone_name, zone_type = "N/A", None
                        
                        # Update Role Classifier (only for persons, class 0)
                        if c == 0:
                            role_classifier.update(logical_tid, zone_name, zone_type)

                        if key not in summary_data:
                            summary_data[key] = {
                                "Split": s_idx,
                                "ID": logical_tid,
                                "Class": "person" if c == 0 else "car",
                                "Start_Time_Sec": round(global_now, 2),
                                "End_Time_Sec": round(global_now, 2),
                                "Frame_Count": 1,
                                "Start_Center": (cx, cy),
                                "End_Center": (cx, cy),
                                "Zone": zone_name,
                                "Role": role_classifier.get_role(logical_tid) if c == 0 else "N/A",
                                "Camera_ID": CAMERA_ID,
                                "Session_Start": session_start_iso,
                                "Bbox_Start": f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}",
                                "Bbox_End": f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}",
                            }
                        else:
                            summary_data[key]["End_Time_Sec"] = round(global_now, 2)
                            summary_data[key]["Frame_Count"] += 1
                            summary_data[key]["End_Center"] = (cx, cy)
                            summary_data[key]["Zone"] = zone_name
                            summary_data[key]["Bbox_End"] = f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}"
                            if c == 0:
                                summary_data[key]["Role"] = role_classifier.get_role(logical_tid)

                # --- CASH DETECTION (runs after person tracking) ---
                cash_detections = []
                cash_associations = {"assigned": {}, "unassigned": []}
                cash_events = []
                person_hands = {}
                hand_interactions = []
                exchange_events = []
                fraud_alerts = []
                
                if cash_detector is not None:
                    # Layer 4: Cash Detection
                    cash_detections = cash_detector.detect(
                        r.orig_img,
                        person_tracks=curr_frame_tracks,
                        roi_manager=roi_manager,
                    )
                    cash_associations = cash_detector.associate_with_persons(
                        cash_detections, curr_frame_tracks
                    )
                    # Update cash state machine
                    if cash_tracker is not None:
                        cash_events = cash_tracker.update(
                            frame_idx=frame_idx,
                            current_time=current_time,
                            person_tracks=curr_frame_tracks,
                            cash_associations=cash_associations,
                            roi_manager=roi_manager,
                        )
                        
                    # Build roles dict for advanced layers
                    current_roles = {tid: role_classifier.get_role(tid) for tid in curr_frame_tracks.keys() if curr_frame_tracks[tid]['cls'] == 0}
                        
                    # Layer 3: Hand Detection & Interaction Analysis
                    if hand_detector:
                        person_hands, hand_interactions = hand_detector.detect_and_analyze(
                            frame=r.orig_img,
                            person_tracks=curr_frame_tracks,
                            roles=current_roles,
                            frame_idx=frame_idx,
                            roi_manager=roi_manager
                        )
                        
                    # Layer 5: Interaction Analyzer
                    if interaction_analyzer:
                        exchange_events = interaction_analyzer.update(
                            frame_idx=frame_idx,
                            current_time=current_time,
                            person_tracks=curr_frame_tracks,
                            roles=current_roles,
                            hand_interactions=hand_interactions,
                            cash_detections=cash_detections,
                            roi_manager=roi_manager
                        )
                        if exchange_events:
                            all_exchange_events.extend(exchange_events)
                            for evt in exchange_events:
                                print(f"  [Exchange] {evt.reason} (Conf: {evt.confidence})")
                                
                    # Layer 6: Fraud Detector
                    if fraud_detector:
                        fraud_alerts = fraud_detector.evaluate(
                            frame_idx=frame_idx,
                            current_time=current_time,
                            exchange_events=exchange_events,
                            cash_events=cash_events,
                            person_hands=person_hands,
                            roles=current_roles,
                            roi_manager=roi_manager
                        )
                        if fraud_alerts:
                            all_fraud_alerts.extend(fraud_alerts)
                            for alert in fraud_alerts:
                                print(f"  [FRAUD ALERT] {alert.alert_type}: {alert.description}")

                # --- CUSTOM ANNOTATION (Visualizing the LOGICAL ID) ---
                # We draw on the original frame using the re-linked IDs we just calculated.
                # r.orig_img is the original numpy array (BGR). We copy it to avoid mutating source if buffered.
                annotated_frame = r.orig_img.copy()
                
                # Draw ROI zones first (so they appear behind the bounding boxes)
                if roi_manager.has_zones:
                    roi_manager.draw_zones(annotated_frame)

                # Build set of active cash holders for annotation
                active_cash_holders = set()
                if cash_tracker is not None:
                    active_cash_holders = set(cash_tracker.get_active_cash_holders())

                for tid, tdata in curr_frame_tracks.items():
                    bx = tdata['bbox']
                    cls_id = tdata['cls']
                    
                    # Convert float box to int
                    x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                    
                    # --- Role-based coloring for persons ---
                    if cls_id == 0:  # Person
                        role = role_classifier.get_role(tid)
                        cash_label = " 💰" if tid in active_cash_holders else ""
                        if role == "Cashier":
                            color = (255, 255, 0)   # Cyan
                            label = f"ID: {tid} (Cashier){cash_label}"
                        else:
                            color = (0, 255, 255)   # Yellow
                            label = f"ID: {tid} (Customer){cash_label}"
                    elif cls_id == 2:  # Car
                        color = (255, 0, 0)
                        label = f"ID: {tid} (Car)"
                    else:
                        color = (255, 255, 255)
                        label = f"ID: {tid}"

                    # Draw BBox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    t_size = cv2.getTextSize(label, 0, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 2), 0, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)

                # Draw advanced interactions
                if hand_detector is not None:
                    hand_detector.draw_hands(annotated_frame, person_hands, current_roles)
                    hand_detector.draw_interactions(annotated_frame, hand_interactions)
                    
                # Draw latest active fraud alerts
                if fraud_alerts:
                    alert_text = f"ALERT: {fraud_alerts[-1].alert_type}"
                    cv2.putText(annotated_frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # Draw cash bounding boxes
                if cash_detector is not None and cash_detections:
                    cash_detector.draw_detections(annotated_frame, cash_detections, cash_associations)

                # Write the CUSTOM frame, not r.plot()
                out_writer.write(annotated_frame)
                
                # 2. Detect Lost Tracks (Present in Prev, Missing in Curr)
                for pid in prev_frame_tracks:
                    if pid not in curr_frame_tracks:
                        # logical ID 'pid' was here last frame, gone now. Buffer it.
                        # USE LAST_SEEN to get the most up-to-date info
                        if pid in last_seen:
                            ls_data = last_seen[pid]
                            occlusion_handler.add_lost_track(
                                track_id=pid,
                                bbox=ls_data['bbox'],
                                cls_id=ls_data['cls'],
                                timestamp=current_time
                            )
                        else:
                            # Fallback (shouldn't happen if logic is correct)
                             occlusion_handler.add_lost_track(
                                track_id=pid,
                                bbox=prev_frame_tracks[pid]['bbox'],
                                cls_id=prev_frame_tracks[pid]['cls'],
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
    
    # Convert dict to list
    raw_events = list(summary_data.values())
    
    # Apply Event Continuity Merging (Phase 3)
    from utils.event_merger import EventMerger
    merger = EventMerger(max_time_gap=5.0, max_speed_mps=1000.0)
    refined_events = merger.merge_events(raw_events)

    final_report_rows = []
    
    for row in refined_events:
        # Filter short events
        if row["Frame_Count"] >= MIN_PERSISTENCE or row["Class"] == "car":
            # Compute dwell category from duration
            dwell_sec = row["End_Time_Sec"] - row["Start_Time_Sec"]
            # Remove internal keys not for CSV (Start/End Center)
            csv_row = {
                "Split": row["Split"],
                "ID": row["ID"],
                "Class": row["Class"],
                "Role": row.get("Role", "N/A"),
                "Start_Time_Sec": row["Start_Time_Sec"],
                "End_Time_Sec": row["End_Time_Sec"],
                "Frame_Count": row["Frame_Count"],
                "Zone": row.get("Zone", "N/A"),
                "Camera_ID": row.get("Camera_ID", "CAM-01"),
                "Session_Start": row.get("Session_Start", session_start_iso),
                "Bbox_Start": row.get("Bbox_Start", ""),
                "Bbox_End": row.get("Bbox_End", ""),
                "Dwell_Category": _dwell_category(dwell_sec),
            }
            final_report_rows.append(csv_row)

    CSV_FIELDS = [
        "Split", "ID", "Class", "Role",
        "Start_Time_Sec", "End_Time_Sec", "Frame_Count", "Zone",
        "Camera_ID", "Session_Start", "Bbox_Start", "Bbox_End", "Dwell_Category",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(final_report_rows)

    # --- 4b. CASH EVENTS CSV ---
    if cash_tracker is not None:
        cash_events_all = cash_tracker.get_all_events()
        if cash_events_all:
            cash_csv = os.path.join(session_dir, "cash_events.csv")
            CASH_CSV_FIELDS = [
                "Event_Type", "Person_ID", "Timestamp_Sec", "Frame_Idx",
                "Zone", "Confidence", "Partner_ID", "Bbox_Snapshot",
                "Camera_ID", "Session_Start",
            ]
            with open(cash_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CASH_CSV_FIELDS)
                writer.writeheader()
                for ce in cash_events_all:
                    writer.writerow({
                        "Event_Type": ce.event_type.value,
                        "Person_ID": ce.person_id,
                        "Timestamp_Sec": round(ce.timestamp, 2),
                        "Frame_Idx": ce.frame_idx,
                        "Zone": ce.zone,
                        "Confidence": round(ce.confidence, 3),
                        "Partner_ID": ce.partner_id or "",
                        "Bbox_Snapshot": ce.bbox_snapshot or "",
                        "Camera_ID": CAMERA_ID,
                        "Session_Start": session_start_iso,
                    })
            print(f"Cash Events CSV: {os.path.basename(cash_csv)} ({len(cash_events_all)} events)")
        
        # Exchanges CSV
        if all_exchange_events:
            EXCHANGE_CSV_FIELDS = [
                "Customer_ID", "Cashier_ID", "Timestamp_Sec", "Frame_Idx",
                "Confidence", "Reason", "Camera_ID", "Session_Start"
            ]
            with open(exchange_events_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EXCHANGE_CSV_FIELDS)
                writer.writeheader()
                for evt in all_exchange_events:
                    writer.writerow({
                        "Customer_ID": evt.customer_id,
                        "Cashier_ID": evt.cashier_id,
                        "Timestamp_Sec": round(evt.timestamp, 2),
                        "Frame_Idx": evt.frame_idx,
                        "Confidence": round(evt.confidence, 3),
                        "Reason": evt.reason,
                        "Camera_ID": CAMERA_ID,
                        "Session_Start": session_start_iso,
                    })
            print(f"Exchange Events CSV: {os.path.basename(exchange_events_csv)} ({len(all_exchange_events)} events)")
            
        # Fraud Alerts CSV
        if all_fraud_alerts:
            FRAUD_CSV_FIELDS = [
                "Alert_Type", "Person_ID", "Timestamp_Sec", "Frame_Idx",
                "Confidence", "Description", "Camera_ID", "Session_Start"
            ]
            with open(fraud_alerts_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FRAUD_CSV_FIELDS)
                writer.writeheader()
                for alert in all_fraud_alerts:
                    writer.writerow({
                        "Alert_Type": alert.alert_type,
                        "Person_ID": alert.person_id,
                        "Timestamp_Sec": round(alert.timestamp, 2),
                        "Frame_Idx": alert.frame_idx,
                        "Confidence": round(alert.confidence, 3),
                        "Description": alert.description,
                        "Camera_ID": CAMERA_ID,
                        "Session_Start": session_start_iso,
                    })
            print(f"Fraud Alerts CSV: {os.path.basename(fraud_alerts_csv)} ({len(all_fraud_alerts)} alerts)")
            
        # Print cash summary
        cash_summary = cash_tracker.get_summary()
        print(f"\nCash Detection Summary:")
        print(f"  Total cash events: {cash_summary['total_events']}")
        for etype, count in cash_summary['events_by_type'].items():
            if count > 0:
                print(f"  - {etype}: {count}")
        print(f"  Total exchange events: {len(all_exchange_events)}")
        print(f"  Total fraud alerts   : {len(all_fraud_alerts)}")

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

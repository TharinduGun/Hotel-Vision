"""
Default configuration for the cash detection module.
These values are used when system_config.yaml doesn't specify them.

All tunable parameters for every sub-component are centralised here
so that adjustments don't require hunting through 5+ files.
"""

CASH_DETECTION_DEFAULTS = {
    # ── General ────────────────────────────────────────────────────────
    "enabled": True,
    "model_path": "pycode/models/cash_detector_v2/train_v2/weights/best.pt",
    "conf_threshold": 0.35,
    "fps": 25.0,
    "cameras": ["*"],

    # SAHI Slicing
    "use_sahi": False,
    "sahi_slice_width": 640,
    "sahi_slice_height": 640,
    "sahi_overlap_ratio": 0.25,

    # ── Cash Detector (Layer 1) — context-aware filtering ─────────────
    "hand_region_ratio": 0.50,       # Lower 50% of person bbox = hand region
    "hand_margin_px": 60,            # Horizontal margin beyond person bbox for hands
    "exchange_gap_px": 100,          # Max gap between two persons for exchange context
    "counter_person_radius_px": 250, # Max distance from any person for counter rule
    # Geometric sanity
    "min_area_px": 400,              # Reject tiny noise blobs
    "max_area_ratio": 0.10,          # Reject boxes covering >10% of frame
    "min_aspect_ratio": 1.0,         # Minimum aspect ratio (long/short)
    "max_aspect_ratio": 8.0,         # Maximum aspect ratio

    # Contextual Tuning
    "hand_reach_px": 30,             # Pixels added below bbox for hand reach
    "exchange_vertical_slack_px": 100, # Vertical slack for person overlap
    "near_hands_offset_px": 50,      # Additional margin around hand region
    
    # Association Heuristics
    "iou_weight": 0.5,
    "center_inside_weight": 0.4,
    "near_hands_weight": 0.3,

    # ── Cash Tracker (Layer 3) — state machine debounce ───────────────
    "pickup_debounce": 8,            # Frames cash must be seen before CASH_PICKUP
    "deposit_debounce": 15,          # Frames cash must be absent before CASH_DEPOSIT
    "zone_alert_cooldown": 30,       # Frames between repeated zone violation alerts

    # ── Hand Detector (Layer 4) — pose estimation ─────────────────────
    "pose_model_path": "yolov8m-pose.pt",
    "keypoint_conf_threshold": 0.3,  # Min confidence for wrist keypoints
    "interaction_threshold_px": 90,  # Max distance between hands to count as interaction
    "iou_threshold": 0.3,            # Minimum IOU to map pose to person

    # ── Role Classifier ───────────────────────────────────────────────
    "cashier_threshold": 0.60,       # Fraction of frames in staff zone to be "Cashier"

    # ── Interaction Analyzer (Layer 5) — signal fusion ────────────────
    "interaction_time_window_sec": 3.0,    # Rolling window for signal history
    "required_interaction_frames": 25,     # Min frames of hand proximity (1.0s at 25fps)
    "interaction_cooldown_frames": 75,     # Prevent spamming exchange events
    "inferred_exchange_sec": 1.5,          # Much longer interaction needed if no cash detected

    # ── Fraud Detector (Layer 6) — business rules ─────────────────────
    "register_wait_sec": 10.0,       # Time to visit register after exchange
    "pocketing_window_sec": 5.0,     # Window to detect post-exchange pocketing

    # ── Temporal Filter — flicker suppression ─────────────────────────
    "temporal_min_frames": 2,        # Min frames with cash in window to confirm
    "temporal_window": 5,            # Sliding window size (frames)
}

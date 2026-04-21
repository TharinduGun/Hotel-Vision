"""
Default configuration for the cash detection module.
These values are used when system_config.yaml doesn't specify them.

All tunable parameters for every sub-component are centralised here
so that adjustments don't require hunting through 5+ files.
"""

CASH_DETECTION_DEFAULTS = {
    # ── General ────────────────────────────────────────────────────────
    "enabled": True,
    "model_path": "pycode/models/cash_detector_v3/train/weights/best.pt",
    "conf_threshold": 0.55,          # ↑ Was 0.35 (Phase 0A band-aid)
    "fps": 25.0,
    "cameras": ["*"],

    # ── SAHI (Slicing Aided Hyper Inference) ───────────────────────────
    "use_sahi": False,               # Disabled by default for performance
    "sahi_slice_width": 640,
    "sahi_slice_height": 640,
    "sahi_overlap_ratio": 0.25,

    # ── Cash Detector (Layer 1) — context-aware filtering ─────────────
    "hand_region_ratio": 0.35,       # ↓ Was 0.50 — only bottom 35% (Phase 0A)
    "hand_margin_px": 30,            # ↓ Was 60 — tighter hand margin (Phase 0A)
    "exchange_gap_px": 100,          # Max gap between two persons for exchange context
    "counter_person_radius_px": 250, # Max distance from any person for counter rule
    # Geometric sanity
    "min_area_px": 800,              # ↑ Was 400 — reject small hand fragments (Phase 0A)
    "max_area_ratio": 0.05,          # ↓ Was 0.10 — cash can't cover 5% of frame (Phase 0A)
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
    "pickup_debounce": 15,           # ↑ Was 8 — 15 frames = ~0.6s at 25fps (Phase 0A)
    "deposit_debounce": 15,          # Frames cash must be absent before CASH_DEPOSIT
    "zone_alert_cooldown": 30,       # Frames between repeated zone violation alerts
    # Phase 1: Zone-free state machine params
    "occlusion_grace_frames": 20,    # Frames to wait before deciding cash is gone (0.8s)
    "suspicious_confirm_count": 3,   # Frames in SUSPICIOUS before POCKET alert
    "stale_profile_frames": 300,     # Remove profiles not seen for ~12s
    "stationary_threshold_px": 15,   # Max center movement to be "stationary"
    "proximity_threshold_px": 200,   # Max distance between persons for "near"

    # ── Hand Detector (Layer 4) — pose estimation ─────────────────────
    "pose_model_path": "yolov8m-pose.pt",
    "keypoint_conf_threshold": 0.3,  # Min confidence for wrist keypoints
    "interaction_threshold_px": 90,  # Max distance between hands to count as interaction
    "iou_threshold": 0.3,            # Minimum IOU to map pose to person

    # ── Role Classifier — behavioral signals ──────────────────────────
    "cashier_threshold": 0.50,       # ↓ Was 0.60 — multi-signal score threshold
    # Signal weights
    "role_zone_weight": 0.9,         # Weight for zone occupancy signal
    "role_stationary_weight": 0.7,   # Weight for stationarity signal
    "role_visitor_weight": 0.5,      # Weight for visitor count signal
    # Stationarity params
    "role_stationary_frames": 450,   # ~18s at 25fps before considering stationary
    "role_movement_threshold_px": 20.0,  # Max drift to be "stationary"
    # Visitor params
    "role_visitor_count_threshold": 3,   # Unique nearby IDs to be "receiving visitors"

    # ── Interaction Analyzer (Layer 5) — signal fusion ────────────────
    "interaction_time_window_sec": 3.0,    # Rolling window for signal history
    "required_interaction_frames": 25,     # Min frames of hand proximity (1.0s at 25fps)
    "interaction_cooldown_frames": 75,     # Prevent spamming exchange events
    "inferred_exchange_sec": 1.5,          # Much longer interaction needed if no cash detected

    # ── Fraud Detector (Layer 6) — business rules ─────────────────────
    "register_wait_sec": 10.0,       # Time to visit register after exchange
    "pocketing_window_sec": 5.0,     # Window to detect post-exchange pocketing

    # ── Temporal Filter — flicker suppression ─────────────────────────
    "temporal_min_frames": 4,        # ↑ Was 2 — need 4/5 frames to confirm (Phase 0A)
    "temporal_window": 5,            # Sliding window size (frames)
}

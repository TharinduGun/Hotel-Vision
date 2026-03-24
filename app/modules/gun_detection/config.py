"""
Default configuration for the gun detection module.
These values are used when the system config doesn't specify them.
"""

GUN_DETECTION_DEFAULTS = {
    "enabled": True,
    "model_path": "pycode/models/gun_detector/weights/best.pt",
    "conf_threshold": 0.55,
    "person_roi_only": True,
    "alert_cooldown_sec": 10.0,
    "save_snapshots": True,
    "save_clips": True,
    "clip_duration_sec": 5.0,
    "cameras": ["*"],  # All cameras by default
    # Hand proximity filter (YOLOv8-pose)
    "hand_proximity_filter": True,
    "pose_model_path": "pycode/src/yolov8m-pose.pt",
    "hand_radius_ratio": 0.4,      # Max distance from wrist as ratio of person height
    # Bbox size filter
    "max_weapon_area_ratio": 0.40,  # Weapon area must be < 40% of person area
}

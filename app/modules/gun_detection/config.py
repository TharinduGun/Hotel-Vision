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
}

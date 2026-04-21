"""
Default configuration for the crowd detection module.
These values are used when the system config doesn't specify them.
"""

CROWD_DETECTION_DEFAULTS = {
    "enabled": True,
    "cameras": ["*"],                    # All cameras by default

    # ── Density thresholds (persons in frame) ─────────────────────
    "density_low_max": 3,                # 0–3 = LOW
    "density_moderate_max": 8,           # 4–8 = MODERATE
    "density_high_max": 15,              # 9–15 = HIGH
    # 16+ = CRITICAL

    # ── Footfall ──────────────────────────────────────────────────
    "edge_margin_ratio": 0.05,           # 5% of frame edge = entry/exit zone
    "entry_exit_cooldown_sec": 1.0,      # Debounce: 1s between entry/exit for same track

    # ── Heat map ──────────────────────────────────────────────────
    "heatmap_resolution": 100,           # Grid cells along the longest axis
    "heatmap_decay": 0.998,              # Per-frame decay factor (0.998 = slow fade)
    "heatmap_alpha": 0.4,                # Overlay transparency on video

    # ── Trajectory ────────────────────────────────────────────────
    "trajectory_max_length": 300,        # Max centroid points stored per track
    "trajectory_draw_length": 60,        # Points to draw on video (recent path)

    # ── Alerts ────────────────────────────────────────────────────
    "alert_on_density_change": True,     # Emit event when density level changes
    "high_density_alert_sec": 10.0,      # Alert cooldown for density warnings
    "critical_density_alert_sec": 5.0,   # Alert cooldown for critical density
}

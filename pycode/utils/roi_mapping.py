"""
ROI Mapping Module
===================
Loads zone definitions from config/zones.json and provides utilities
to check which zone a point falls in and to draw zone overlays on frames.
"""

import json
import os
import cv2


class ROIZone:
    """Represents a single rectangular Region of Interest."""

    def __init__(self, name, zone_type, roi):
        """
        Args:
            name (str): Human-readable name, e.g. "Cashier 1".
            zone_type (str): Category, e.g. "cashier", "money_exchange".
            roi (list): [x, y, w, h] bounding rectangle.
        """
        self.name = name
        self.zone_type = zone_type
        self.x, self.y, self.w, self.h = roi

    def contains_point(self, cx, cy):
        """Check if a point (cx, cy) is inside this zone."""
        return (self.x <= cx <= self.x + self.w and 
                self.y <= cy <= self.y + self.h)

    def contains_bbox(self, x1, y1, x2, y2, threshold=0.5):
        """
        Check if a bounding box overlaps with this zone above a threshold.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates.
            threshold (float): Minimum fraction of bbox area that must be
                               inside the zone to count as "in zone".
        
        Returns:
            bool: True if overlap ratio >= threshold.
        """
        # Intersection
        ix1 = max(self.x, x1)
        iy1 = max(self.y, y1)
        ix2 = min(self.x + self.w, x2)
        iy2 = min(self.y + self.h, y2)

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        bbox_area = max((x2 - x1) * (y2 - y1), 1e-6)

        return (inter_area / bbox_area) >= threshold

    def __repr__(self):
        return f"ROIZone('{self.name}', type='{self.zone_type}', roi=[{self.x},{self.y},{self.w},{self.h}])"


class ROIManager:
    """Manages multiple ROI zones. Loads from JSON config."""

    # Color map for visualization
    COLOR_MAP = {
        "cashier": (0, 255, 0),          # Green
        "money_exchange": (0, 165, 255),  # Orange
        "cash_register": (255, 0, 255),   # Magenta
    }
    DEFAULT_COLOR = (255, 255, 255)

    def __init__(self, config_path=None, frame_size=None):
        """
        Args:
            config_path (str): Path to zones.json. If None, uses default location.
            frame_size (tuple): (width, height) of the actual video frames.
                                If provided and different from the reference image,
                                ROI coordinates are automatically scaled.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "zones.json"
            )
        self.config_path = os.path.abspath(config_path)
        self.zones = []
        self.image_size = None
        self.frame_size = frame_size  # Target video frame size
        self._load()

    def _load(self):
        """Load zones from the JSON config file and scale if needed."""
        if not os.path.exists(self.config_path):
            print(f"[ROIManager] WARNING: Config not found: {self.config_path}")
            print("[ROIManager] Run roi_selector.py first to define zones.")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.image_size = config.get("image_size")  # [ref_w, ref_h]
        
        # Calculate scale factors if frame_size differs from reference image
        scale_x, scale_y = 1.0, 1.0
        if self.frame_size and self.image_size:
            ref_w, ref_h = self.image_size
            vid_w, vid_h = self.frame_size
            if ref_w != vid_w or ref_h != vid_h:
                scale_x = vid_w / ref_w
                scale_y = vid_h / ref_h
                print(f"[ROIManager] Scaling ROIs from {ref_w}x{ref_h} -> {vid_w}x{vid_h} "
                      f"(scale: {scale_x:.3f}, {scale_y:.3f})")
        
        for z in config.get("zones", []):
            raw_roi = z["roi"]  # [x, y, w, h]
            scaled_roi = [
                int(raw_roi[0] * scale_x),
                int(raw_roi[1] * scale_y),
                int(raw_roi[2] * scale_x),
                int(raw_roi[3] * scale_y),
            ]
            zone = ROIZone(
                name=z["name"],
                zone_type=z["type"],
                roi=scaled_roi
            )
            self.zones.append(zone)

        print(f"[ROIManager] Loaded {len(self.zones)} zone(s) from {self.config_path}")
        for z in self.zones:
            print(f"  - {z}")

    def get_zone(self, cx, cy):
        """
        Get the zone name for a given center point.
        
        Args:
            cx, cy: Center coordinates of the tracked object.
        
        Returns:
            str: Zone name if point is inside a zone, else "Outside".
        """
        for zone in self.zones:
            if zone.contains_point(cx, cy):
                return zone.name
        return "Outside"

    def get_zone_with_type(self, cx, cy):
        """
        Get the zone name AND type for a given center point.
        
        Args:
            cx, cy: Center coordinates of the tracked object.
        
        Returns:
            tuple: (zone_name, zone_type) if inside a zone,
                   ("Outside", None) if not in any zone.
        """
        for zone in self.zones:
            if zone.contains_point(cx, cy):
                return zone.name, zone.zone_type
        return "Outside", None

    def get_zone_for_bbox(self, x1, y1, x2, y2, threshold=0.5):
        """
        Get the zone name for a bounding box using overlap ratio.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates.
            threshold: Minimum overlap fraction.
        
        Returns:
            str: Zone name if bbox overlaps enough, else "Outside".
        """
        for zone in self.zones:
            if zone.contains_bbox(x1, y1, x2, y2, threshold):
                return zone.name
        return "Outside"

    def get_all_zones_for_point(self, cx, cy):
        """
        Get ALL zone names a point falls in (for overlapping zones).
        
        Returns:
            list[str]: List of zone names. Empty if outside all zones.
        """
        return [z.name for z in self.zones if z.contains_point(cx, cy)]

    def draw_zones(self, frame, alpha=0.2, thickness=2):
        """
        Draw all zones as semi-transparent overlays on the frame.
        
        Args:
            frame: BGR numpy array (modified in-place).
            alpha: Transparency of the fill (0=invisible, 1=opaque).
            thickness: Border line thickness.
        
        Returns:
            frame: The annotated frame (same reference).
        """
        overlay = frame.copy()

        for zone in self.zones:
            color = self.COLOR_MAP.get(zone.zone_type, self.DEFAULT_COLOR)
            x, y, w, h = zone.x, zone.y, zone.w, zone.h

            # Semi-transparent fill
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

            # Border
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Label
            label = zone.name
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(frame, (x, y - t_size[1] - 8), (x + t_size[0] + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Blend the transparent fill
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    @property
    def has_zones(self):
        """Check if any zones are loaded."""
        return len(self.zones) > 0

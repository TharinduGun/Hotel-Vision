import cv2
import numpy as np


class ParkingSpaceAnalyzer:
    """
    Checks each defined parking space individually instead of just
    counting vehicles in the whole frame.

    The key difference from the old approach:
    OLD: count how many vehicle centroids fall inside each zone polygon
    NEW: for each defined space, check what fraction of the space is
         covered by a vehicle bounding box — if above a threshold,
         mark the space as occupied

    This is more robust because:
    - A vehicle parked slightly off-centre still occupies its space
    - A vehicle overlapping two spaces gets assigned to the one it
      covers more of
    - Works even when the detector slightly misplaces the bounding box
    """

    # A space is considered occupied if a vehicle bbox covers this much
    # of the space polygon area. 0.25 = 25% overlap.
    # Raise this if you get false "occupied" readings from vehicles in
    # adjacent spaces. Lower it if real vehicles are being missed.
    OCCUPANCY_OVERLAP_THRESHOLD = 0.25

    def __init__(self):
        # Import here so an empty PARKING_SPACES doesn't break the pipeline
        try:
            from configs.parking_spaces import PARKING_SPACES
            self.spaces = PARKING_SPACES
        except (ImportError, AttributeError):
            self.spaces = {}


    def _polygon_area(self, poly) -> float:
        """Shoelace formula for polygon area in pixels."""
        pts = np.array(poly, dtype=np.float32)
        x   = pts[:, 0]
        y   = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


    def _bbox_polygon_overlap_ratio(self, bbox, poly) -> float:
        """
        Returns what fraction of the space polygon is covered by the
        vehicle bounding box.

        Uses a rasterization approach — creates a mask for both shapes
        and measures pixel overlap. Accurate even for irregular polygons.
        """
        x1, y1, x2, y2 = bbox

        # Find the bounds of the space polygon to create a local mask
        pts     = np.array(poly, dtype=np.int32)
        px_min  = int(pts[:, 0].min())
        py_min  = int(pts[:, 1].min())
        px_max  = int(pts[:, 0].max())
        py_max  = int(pts[:, 1].max())

        # Add padding
        pad    = 5
        ox     = max(0, px_min - pad)
        oy     = max(0, py_min - pad)
        width  = px_max - ox + pad * 2
        height = py_max - oy + pad * 2

        if width <= 0 or height <= 0:
            return 0.0

        # Mask for the parking space polygon
        space_mask = np.zeros((height, width), dtype=np.uint8)
        shifted_poly = pts - np.array([ox, oy])
        cv2.fillPoly(space_mask, [shifted_poly], 255)

        # Mask for the vehicle bounding box
        vehicle_mask = np.zeros((height, width), dtype=np.uint8)
        vx1 = max(0, x1 - ox)
        vy1 = max(0, y1 - oy)
        vx2 = min(width, x2 - ox)
        vy2 = min(height, y2 - oy)
        if vx2 > vx1 and vy2 > vy1:
            cv2.rectangle(vehicle_mask, (vx1, vy1), (vx2, vy2), 255, -1)

        # Overlap = pixels where both masks are white
        overlap      = cv2.bitwise_and(space_mask, vehicle_mask)
        space_pixels = np.count_nonzero(space_mask)
        overlap_pix  = np.count_nonzero(overlap)

        if space_pixels == 0:
            return 0.0

        return overlap_pix / space_pixels


    def analyze(self, detections: list) -> dict:
        """
        Returns per-space status: each space is either 'occupied' or 'free',
        plus the vehicle ID that's in it (useful for dwell time per space).
        """
        if not self.spaces:
            return {}

        result = {}

        for space_name, poly in self.spaces.items():

            best_overlap  = 0.0
            occupying_tid = None

            for det in detections:
                overlap = self._bbox_polygon_overlap_ratio(det["bbox"], poly)
                if overlap > best_overlap:
                    best_overlap  = overlap
                    occupying_tid = det.get("track_id")

            is_occupied = best_overlap >= self.OCCUPANCY_OVERLAP_THRESHOLD

            result[space_name] = {
                "occupied":   is_occupied,
                "overlap":    round(best_overlap, 2),  # useful for debugging
                "vehicle_id": occupying_tid if is_occupied else None,
            }

        return result


    def count_occupied(self, space_counts: dict) -> int:
        """Helper to get total occupied count from analyze() output."""
        return sum(1 for s in space_counts.values() if s["occupied"])

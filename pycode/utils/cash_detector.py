"""
Cash Detector Module
=====================
Wrapper around a YOLOv8 model trained to detect cash/banknotes.
Provides detection, context-aware filtering, and person-association utilities.

Classes detected: Cash, person (from the Roboflow dataset)
We only use the "Cash" class (index 0) from this model.

Context-Aware Filtering:
  Cash detections are validated against real-world context rules:
  1. Near a person's hands (lower portion of person bbox)
  2. On a counter/register zone (cashier, money_exchange, cash_register)
  3. Between two people (exchange scenario)
  Detections outside all 3 contexts are rejected as false positives.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class CashDetection:
    """A single cash detection in a frame."""

    __slots__ = ("bbox", "confidence", "class_name", "class_id", "center")

    def __init__(self, bbox, confidence, class_name, class_id):
        """
        Args:
            bbox (list): [x1, y1, x2, y2] bounding box.
            confidence (float): Detection confidence score.
            class_name (str): Class name, e.g. "Cash".
            class_id (int): Class index from the model.
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        self.class_id = class_id
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def __repr__(self):
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        return f"CashDetection({self.class_name}, conf={self.confidence:.2f}, bbox=[{x1},{y1},{x2},{y2}])"


class CashDetector:
    """
    Detects cash/banknotes in video frames using a fine-tuned YOLOv8 model.
    
    The model runs independently from the person tracker — it only looks 
    for cash objects. Association with persons is done via spatial overlap.
    
    Context-Aware Filtering:
      After raw YOLO detections, each detection is validated against
      real-world contexts where cash actually appears:
        - Near a tracked person's hands
        - On a counter/register zone
        - Between two people (exchange)
      This dramatically reduces false positives from random objects.
    """

    # Only detect class 0 = "Cash" (ignore class 1 = "person" from this model)
    CASH_CLASS_ID = 0
    CASH_CLASS_NAME = "Cash"

    # Color for drawing cash bounding boxes (bright green in BGR)
    DRAW_COLOR = (0, 255, 100)
    DRAW_COLOR_UNASSIGNED = (0, 0, 255)  # Red for unassigned cash

    # Zone types considered valid for cash presence
    CASH_ZONE_TYPES = {"cashier", "money_exchange", "cash_register"}

    def __init__(
        self,
        model_path,
        conf_threshold=0.35,
        device="cuda",
        # Contextual filter parameters
        hand_region_ratio=0.50,   # Lower 50% of person bbox = hand region
        hand_margin_px=60,        # Horizontal margin beyond person bbox for hands
        exchange_gap_px=100,      # Max gap between two persons for exchange context
        counter_person_radius_px=250,  # Max distance from any person for counter rule
        # Geometric sanity checks
        min_area_px=400,          # Reject tiny noise blobs
        max_area_ratio=0.10,      # Reject boxes covering >10% of frame
        min_aspect_ratio=1.0,     # Minimum aspect ratio (cash can look square when folded)
        max_aspect_ratio=8.0,     # Maximum aspect ratio (long/short side)
    ):
        """
        Args:
            model_path (str): Path to the trained cash detection model (best.pt).
            conf_threshold (float): Minimum confidence for detections.
            device (str): Device to run inference on ("cuda" or "cpu").
            hand_region_ratio (float): Fraction of person bbox from the bottom that
                counts as the "hand region" (e.g., 0.50 = lower half).
            hand_margin_px (int): Horizontal pixel margin outside person bbox
                where cash near hands is still considered valid.
            exchange_gap_px (int): Max horizontal gap in px between two persons 
                for the "exchange" context.
            min_area_px (int): Min bounding box area in pixels (reject noise).
            max_area_ratio (float): Max fraction of frame area a cash box can cover.
            min_aspect_ratio (float): Min aspect ratio (longer side / shorter side).
            max_aspect_ratio (float): Max aspect ratio.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cash detection model not found: {model_path}\n"
                "Run pycode/scripts/train_cash_detector.py first."
            )

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self._class_names = self.model.names  # {0: "Cash", 1: "person"}

        # Contextual filter params
        self.hand_region_ratio = hand_region_ratio
        self.hand_margin_px = 100   # Was 60
        self.exchange_gap_px = 150  # Was 100
        self.counter_person_radius_px = counter_person_radius_px

        # Geometric sanity params
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        print(f"[CashDetector] Loaded model: {model_path}")
        print(f"[CashDetector] Classes: {self._class_names}")
        print(f"[CashDetector] Confidence threshold: {conf_threshold}")
        print(f"[CashDetector] Context filters: hand_region={hand_region_ratio:.0%}, "
              f"hand_margin={hand_margin_px}px, exchange_gap={exchange_gap_px}px")

    def detect(self, frame, person_tracks=None, roi_manager=None):
        """
        Detect cash objects in a single frame, applying context-aware filtering.
        
        Args:
            frame: BGR numpy array (video frame).
            person_tracks (dict|None): {logical_id: {"bbox": [x1,y1,x2,y2], "cls": int}}
                If provided, enables context-aware filtering (recommended).
            roi_manager: ROIManager instance (optional, for zone-based filtering).
            
        Returns:
            list[CashDetection]: Detected cash objects that pass contextual filters.
                Empty list if none found.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=[self.CASH_CLASS_ID],  # Only detect cash, not persons
            device=self.device,
            verbose=False,
            imgsz=640,
        )

        raw_detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = self._class_names.get(cls_id, "Cash")

                raw_detections.append(CashDetection(
                    bbox=bbox,
                    confidence=conf,
                    class_name=cls_name,
                    class_id=cls_id,
                ))

        if not raw_detections:
            return []

        # --- STAGE 1: Geometric sanity filter ---
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h
        geo_filtered = self._geometric_filter(raw_detections, frame_area)

        # --- STAGE 2: Context-aware filter ---
        if person_tracks is not None:
            context_filtered = self._contextual_filter(
                geo_filtered, person_tracks, roi_manager
            )
            return context_filtered
        else:
            # No person tracks available — return geometric-filtered only
            return geo_filtered

    def _geometric_filter(self, detections, frame_area):
        """
        Reject detections that are geometrically impossible for banknotes.
        
        Filters: minimum area, maximum area ratio, aspect ratio range.
        """
        filtered = []
        for cash in detections:
            x1, y1, x2, y2 = cash.bbox
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # Reject tiny noise blobs
            if area < self.min_area_px:
                continue

            # Reject impossibly large boxes (e.g., covers 10%+ of the frame)
            if area / frame_area > self.max_area_ratio:
                continue

            # Aspect ratio check (use longer/shorter to handle both orientations)
            long_side = max(w, h)
            short_side = max(min(w, h), 1)  # avoid div by zero
            aspect = long_side / short_side

            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue

            filtered.append(cash)

        return filtered

    def _contextual_filter(self, detections, person_tracks, roi_manager=None):
        """
        Only keep cash detections that appear in valid real-world contexts.
        
        A detection passes if ANY of these rules match:
          Rule 1: Cash is near a person's hands (lower portion of person bbox)
          Rule 2: Cash center is inside a cashier/counter/money_exchange zone
          Rule 3: Cash is in the gap between two nearby persons (exchange)
        """
        if not detections:
            return []

        # Get person-only tracks
        persons = {
            pid: data for pid, data in person_tracks.items()
            if data["cls"] == 0
        }
        person_bboxes = [(pid, data["bbox"]) for pid, data in persons.items()]

        filtered = []
        for cash in detections:
            if self._check_near_hands(cash, person_bboxes):
                filtered.append(cash)
            elif roi_manager and self._check_on_counter_zone(cash, roi_manager, person_bboxes):
                filtered.append(cash)
            elif self._check_between_persons(cash, person_bboxes):
                filtered.append(cash)
            # else: rejected — not in any valid cash context

        return filtered

    def _check_near_hands(self, cash, person_bboxes):
        """
        Rule 1: Cash is near a person's hand region.
        
        The hand region is the lower portion of the person's bounding box
        (controlled by hand_region_ratio), extended horizontally by hand_margin_px.
        """
        cx, cy = cash.center
        cash_x1, cash_y1, cash_x2, cash_y2 = cash.bbox

        for pid, pbbox in person_bboxes:
            px1, py1, px2, py2 = pbbox
            person_h = py2 - py1

            # Hand region: lower portion of person bbox
            hand_top = py1 + person_h * (1.0 - self.hand_region_ratio)
            hand_left = px1 - self.hand_margin_px
            hand_right = px2 + self.hand_margin_px
            hand_bottom = py2 + 30  # small extension below person bbox (reaching down)

            # Check if cash bbox overlaps with the hand region
            overlap_x = cash_x1 < hand_right and cash_x2 > hand_left
            overlap_y = cash_y1 < hand_bottom and cash_y2 > hand_top

            if overlap_x and overlap_y:
                return True

        return False

    def _check_on_counter_zone(self, cash, roi_manager, person_bboxes):
        """
        Rule 2: Cash is inside a cashier/counter zone AND near a person.
        
        Requires BOTH conditions:
          - Cash center is in a cashier/money_exchange/cash_register zone
          - A person is within counter_person_radius_px of the cash
        
        This prevents static objects (monitors, screens, signs) on counters
        from being flagged as cash when no person is nearby.
        """
        if not roi_manager or not roi_manager.has_zones:
            return False

        cx, cy = cash.center
        _, zone_type = roi_manager.get_zone_with_type(cx, cy)

        if zone_type is None or zone_type not in self.CASH_ZONE_TYPES:
            return False

        # Must also have a person nearby
        return self._is_near_any_person(cash, person_bboxes, self.counter_person_radius_px)

    def _is_near_any_person(self, cash, person_bboxes, radius_px):
        """
        Check if cash detection center is within radius_px of any person bbox center.
        """
        if not person_bboxes:
            return False

        cx, cy = cash.center
        for pid, pbbox in person_bboxes:
            px1, py1, px2, py2 = pbbox
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist <= radius_px:
                return True

        return False

    def _check_between_persons(self, cash, person_bboxes):
        """
        Rule 3: Cash is in the gap between two nearby persons (exchange scenario).
        
        Checks if the cash bbox center lies horizontally between two persons
        who are close enough to be exchanging items.
        """
        if len(person_bboxes) < 2:
            return False

        cx, cy = cash.center

        for i in range(len(person_bboxes)):
            for j in range(i + 1, len(person_bboxes)):
                _, bbox_a = person_bboxes[i]
                _, bbox_b = person_bboxes[j]

                # Determine the gap between the two persons (horizontal)
                a_left, a_top, a_right, a_bottom = bbox_a
                b_left, b_top, b_right, b_bottom = bbox_b

                # Ensure a is on the left
                if a_left > b_left:
                    a_left, a_top, a_right, a_bottom, b_left, b_top, b_right, b_bottom = \
                        b_left, b_top, b_right, b_bottom, a_left, a_top, a_right, a_bottom

                # Horizontal gap between persons
                gap = b_left - a_right

                # Persons must be reasonably close (could be overlapping or with a small gap)
                if gap > self.exchange_gap_px:
                    continue

                # Cash must be horizontally in the overlap/gap region
                gap_left = a_right - self.hand_margin_px
                gap_right = b_left + self.hand_margin_px

                # Cash must be vertically within the overlapping vertical range
                vert_top = max(a_top, b_top)
                vert_bottom = min(a_bottom, b_bottom)

                # Allow some vertical slack if persons don't overlap vertically
                if vert_top > vert_bottom:
                    vert_mid = (max(a_top, b_top) + min(a_bottom, b_bottom)) / 2
                    vert_top = vert_mid - 100
                    vert_bottom = vert_mid + 100

                if gap_left <= cx <= gap_right and vert_top <= cy <= vert_bottom:
                    return True

        return False

    def associate_with_persons(self, cash_detections, person_tracks):
        """
        Associate each cash detection with the nearest person using IOU overlap.
        
        Logic:
          1. For each cash bbox, compute IOU with every person bbox.
          2. If IOU > threshold, assign cash to that person.
          3. If cash bbox center is inside a person bbox, also assign it.
          4. If no person overlaps, mark as "unassigned" (cash on table/counter).
        
        Args:
            cash_detections (list[CashDetection]): Cash detections from detect().
            person_tracks (dict): {logical_id: {"bbox": [x1,y1,x2,y2], "cls": int}}
                                  (same format as curr_frame_tracks in main.py)
        
        Returns:
            dict: {
                "assigned": {person_id: [CashDetection, ...]},
                "unassigned": [CashDetection, ...]
            }
        """
        result = {
            "assigned": {},
            "unassigned": [],
        }

        if not cash_detections:
            return result

        # Filter to only person tracks (cls == 0)
        persons = {
            pid: data for pid, data in person_tracks.items()
            if data["cls"] == 0
        }

        for cash in cash_detections:
            best_person = None
            best_score = 0.0

            for pid, pdata in persons.items():
                # Method 1: IOU overlap
                iou = self._compute_iou(cash.bbox, pdata["bbox"])

                # Method 2: Check if cash center is inside person bbox
                cx, cy = cash.center
                px1, py1, px2, py2 = pdata["bbox"]
                center_inside = (px1 <= cx <= px2 and py1 <= cy <= py2)

                # Method 3: Proximity — is cash near the person's hands?
                # (lower 60% of person bbox is where hands typically are)
                hand_region_y = py1 + (py2 - py1) * 0.4  # Below 40% from top
                near_hands = (cy >= hand_region_y and px1 - 50 <= cx <= px2 + 50)

                # Score: weighted combination
                score = iou * 0.5
                if center_inside:
                    score += 0.4
                if near_hands:
                    score += 0.3

                if score > best_score:
                    best_score = score
                    best_person = pid

            # Threshold: need at least some spatial relationship
            if best_person is not None and best_score >= 0.15:
                if best_person not in result["assigned"]:
                    result["assigned"][best_person] = []
                result["assigned"][best_person].append(cash)
            else:
                result["unassigned"].append(cash)

        return result

    def draw_detections(self, frame, cash_detections, associations=None):
        """
        Draw cash bounding boxes on a frame.
        
        Args:
            frame: BGR numpy array (modified in-place).
            cash_detections (list[CashDetection]): All cash detections.
            associations (dict|None): Output from associate_with_persons().
                If provided, assigned cash is drawn green, unassigned red.
        
        Returns:
            frame: The annotated frame.
        """
        # Build a set of unassigned cash for color coding
        unassigned_set = set()
        if associations:
            for cash in associations.get("unassigned", []):
                unassigned_set.add(id(cash))

        for cash in cash_detections:
            x1, y1, x2, y2 = [int(v) for v in cash.bbox]

            # Color based on assignment status
            if id(cash) in unassigned_set:
                color = self.DRAW_COLOR_UNASSIGNED
            else:
                color = self.DRAW_COLOR

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"Cash {cash.confidence:.0%}"
            t_size = cv2.getTextSize(label, 0, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y2), (x1 + t_size[0], y2 + t_size[1] + 4), color, -1)
            cv2.putText(frame, label, (x1, y2 + t_size[1] + 2), 0, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        return frame

    @staticmethod
    def _compute_iou(box1, box2):
        """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 1e-6)
        area2 = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 1e-6)
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

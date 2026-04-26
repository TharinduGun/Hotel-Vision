"""
Hand Detector Module (Layer 3)
==============================
Uses YOLOv8-pose to detect wrist keypoints and track hand interactions.
Python 3.13 compatible since it relies on ultralytics (already installed)
instead of MediaPipe.

Tracks:
 - Wrist positions for each person
 - Distance between customer and cashier hands
 - Hand presence in transaction zones
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from app.shared.model_manager import model_manager
import logging

from .utils import compute_iou

logger = logging.getLogger(__name__)


@dataclass
class HandInteraction:
    """Represents a potential interaction between a customer and a cashier."""
    customer_id: int
    cashier_id: int
    distance_px: float
    customer_hand_pos: Tuple[int, int]
    cashier_hand_pos: Tuple[int, int]
    in_transaction_roi: bool
    frame_idx: int


class HandDetector:
    """
    Detects hands (wrists) using YOLOv8-pose and analyzes interactions.
    
    COCO Pose Keypoints used:
      9: left_wrist
     10: right_wrist
    """

    # COCO Keypoint indices for wrists
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10

    def __init__(
        self,
        model_path="yolov8m-pose.pt",
        device="cuda",
        keypoint_conf_threshold=0.3,
        interaction_threshold_px=90, # Strict distance (was 150)
        iou_threshold: float = 0.3,
    ):
        """
        Args:
            model_path: Path to YOLO pose model (will auto-download if not present)
            device: 'cuda' or 'cpu'
            keypoint_conf_threshold: Minimum confidence for a wrist keypoint to be valid
            interaction_threshold_px: Max distance between hands to count as an interaction
            iou_threshold: Minimum IOU to match pose detection to a person track
        """
        # self.model = model_manager.get_model(model_path, device) # Model removed, using shared pose
        self.device = device
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.interaction_threshold_px = interaction_threshold_px
        self.iou_threshold = iou_threshold
        logger.info(f"HandDetector now uses shared pose keypoints from PersonDetector")

    def detect_and_analyze(
        self,
        frame: np.ndarray,
        person_tracks: Dict[int, dict],
        roles: Dict[int, str],
        frame_idx: int,
        roi_manager=None
    ) -> Tuple[Dict[int, List[Tuple[int, int]]], List[HandInteraction]]:
        """
        Runs pose detection, maps wrists to tracked persons, and finds interactions.
        
        Args:
            frame: BGR video frame
            person_tracks: {logical_id: {"bbox": [x1,y1,x2,y2], "cls": 0}}
            roles: {logical_id: "Cashier" or "Customer"}
            frame_idx: Current frame number
            roi_manager: Optional ROIManager for zone checking
            
        Returns:
            person_hands: {logical_id: [(x, y), ...]} (up to 2 wrists per person)
            interactions: List of HandInteraction events
        """
        # { person_id: [ (x, y), (x, y) ] }
        person_hands = {}
        for pid in person_tracks.keys():
            person_hands[pid] = []

        # 1. Extract wrists from shared person keypoints
        for pid, pdata in person_tracks.items():
            kpts = pdata.get("keypoints")
            if not kpts:
                continue
                
            hands = []
            # Left wrist
            if len(kpts) > self.LEFT_WRIST_IDX:
                lw_x, lw_y, lw_conf = kpts[self.LEFT_WRIST_IDX]
                if lw_conf >= self.keypoint_conf_threshold:
                    hands.append((int(lw_x), int(lw_y)))
            
            # Right wrist
            if len(kpts) > self.RIGHT_WRIST_IDX:
                rw_x, rw_y, rw_conf = kpts[self.RIGHT_WRIST_IDX]
                if rw_conf >= self.keypoint_conf_threshold:
                    hands.append((int(rw_x), int(rw_y)))
                    
            person_hands[pid].extend(hands)

        # 2. Analyze interactions between Customers and Cashiers
        interactions = []
        
        customers = [pid for pid, role in roles.items() if role == "Customer" and pid in person_hands]
        cashiers = [pid for pid, role in roles.items() if role == "Cashier" and pid in person_hands]

        for cust_id in customers:
            cust_hands = person_hands[cust_id]
            if not cust_hands:
                continue
                
            for cash_id in cashiers:
                cash_hands = person_hands[cash_id]
                if not cash_hands:
                    continue
                    
                # Find minimum distance between any customer hand and any cashier hand
                min_dist = float('inf')
                best_cust_hand = None
                best_cash_hand = None
                
                for pt_cust in cust_hands:
                    for pt_cash in cash_hands:
                        d = math.dist(pt_cust, pt_cash)
                        if d < min_dist:
                            min_dist = d
                            best_cust_hand = pt_cust
                            best_cash_hand = pt_cash
                            
                # Check if interaction threshold met
                if min_dist <= self.interaction_threshold_px:
                    # Optional: Check if the interaction happens inside the transaction ROI
                    in_transaction_zone = False
                    if roi_manager and roi_manager.has_zones:
                        cx = (best_cust_hand[0] + best_cash_hand[0]) / 2
                        cy = (best_cust_hand[1] + best_cash_hand[1]) / 2
                        
                        _, zone_type = roi_manager.get_zone_with_type(cx, cy)
                        # We consider it a transaction zone if it's explicitly marked or if it's a cashier zone
                        if zone_type in {"cashier", "cash_register", "money_exchange"}:
                            in_transaction_zone = True

                    interactions.append(
                        HandInteraction(
                            customer_id=cust_id,
                            cashier_id=cash_id,
                            distance_px=min_dist,
                            customer_hand_pos=best_cust_hand,
                            cashier_hand_pos=best_cash_hand,
                            in_transaction_roi=in_transaction_zone,
                            frame_idx=frame_idx
                        )
                    )

        return person_hands, interactions

    def draw_hands(self, frame: np.ndarray, person_hands: Dict[int, List[Tuple[int, int]]], roles: Dict[int, str]):
        """
        Draws hand landmarks on the frame.
        Customer hands = Blue dots
        Cashier hands = Yellow dots
        """
        for pid, hands in person_hands.items():
            role = roles.get(pid, "Unknown")
            # Color coding
            if role == "Cashier":
                color = (0, 255, 255)  # Yellow
            elif role == "Customer":
                color = (255, 100, 0)  # Blue
            else:
                color = (255, 255, 255)  # White

            for (x, y) in hands:
                # Draw outer circle
                cv2.circle(frame, (x, y), 6, color, 2)
                # Draw inner dot
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    def draw_interactions(self, frame: np.ndarray, interactions: List[HandInteraction]):
        """Draws lines connecting interacting hands."""
        for ix in interactions:
            color = (0, 0, 255) if ix.in_transaction_roi else (0, 165, 255) # Red if inside transaction ROI, else Orange
            pt1 = ix.customer_hand_pos
            pt2 = ix.cashier_hand_pos
            
            # Draw line between hands
            cv2.line(frame, pt1, pt2, color, 3)
            
            # Midpoint for label
            mx = int((pt1[0] + pt2[0]) / 2)
            my = int((pt1[1] + pt2[1]) / 2)
            
            label = f"Interacting ({int(ix.distance_px)}px)"
            cv2.putText(frame, label, (mx - 40, my - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


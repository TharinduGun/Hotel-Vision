"""
Interaction Analyzer Module (Layer 5)
=====================================
Fuses signals from tracked persons, hands, zones, and cash detection
to infer complex events like CASH_EXCHANGE.

Follows a multi-signal timeline approach, making it robust against missing cash detections
by relying on hand interactions and zone presence.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set


@dataclass
class ExchangeEvent:
    customer_id: int
    cashier_id: int
    timestamp: float
    frame_idx: int
    confidence: float
    reason: str


class InteractionAnalyzer:
    """
    Maintains a rolling window of signals per customer-cashier pair
    to trigger meaningful interactions like CASH_EXCHANGE.
    """
    
    def __init__(
        self,
        time_window_sec: float = 3.0,
        fps: float = 25.0,
        required_interaction_frames: int = 25, # Was 5 (0.2s), now 25 (1.0s)
        cooldown_frames: int = 75, # Prevent spamming events
    ):
        self.time_window_sec = time_window_sec
        self.fps = fps
        self.window_frames = int(time_window_sec * fps)
        self.required_interaction_frames = required_interaction_frames
        self.cooldown_frames = cooldown_frames
        
        # { (customer_id, cashier_id): deque([ {frame: idx, signals: set()} ]) }
        self.history = {}
        # { (customer_id, cashier_id): last_event_frame_idx }
        self.last_event_frame = {}
        
    def update(
        self,
        frame_idx: int,
        current_time: float,
        person_tracks: Dict[int, dict],
        roles: Dict[int, str],
        hand_interactions: List['HandInteraction'],  # From hand_detector.py
        cash_detections: List['CashDetection'],      # From cash_detector.py
        roi_manager=None
    ) -> List[ExchangeEvent]:
        """
        Process current frame signals and return detected events.
        """
        events = []
        current_signals = {}  # { (cust_id, cash_id): set("signal_name") }
        
        # 1. Gather Customer Near Counter signals
        cust_near_counter = set()
        cashiers = [pid for pid, r in roles.items() if r == "Cashier"]
        customers = [pid for pid, r in roles.items() if r == "Customer"]
        
        if roi_manager and roi_manager.has_zones:
            for pid in customers:
                bbox = person_tracks.get(pid, {}).get("bbox")
                if bbox:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    _, zone_type = roi_manager.get_zone_with_type(cx, cy)
                    if zone_type in {"cashier", "cash_register", "money_exchange"}:
                        cust_near_counter.add(pid)
                        
        # 2. Gather Hand Interactions
        interacting_pairs = set()
        hands_in_roi = set()
        for hi in hand_interactions:
            # ONLY count interactions that occur IN the transaction zones (cashier/money_exchange)
            if not hi.in_transaction_roi:
                continue
                
            pair = (hi.customer_id, hi.cashier_id)
            interacting_pairs.add(pair)
            
            if pair not in current_signals:
                current_signals[pair] = set()
            
            current_signals[pair].add("HANDS_CLOSE")
            current_signals[pair].add("HANDS_IN_ROI")
                
        # Initialize default pairs if not interacting but present
        for cust_id in customers:
            for cash_id in cashiers:
                pair = (cust_id, cash_id)
                if pair not in current_signals:
                    current_signals[pair] = set()
                    
                if cust_id in cust_near_counter:
                    current_signals[pair].add("CUSTOMER_NEAR_COUNTER")
                    
                # Add cash detection signal
                if len(cash_detections) > 0:
                    current_signals[pair].add("CASH_DETECTED")
                    
        # 3. Update timeline and evaluate rules
        for pair, signals in current_signals.items():
            if pair not in self.history:
                self.history[pair] = deque(maxlen=self.window_frames)
                
            self.history[pair].append({
                "frame": frame_idx,
                "signals": signals,
                "time": current_time
            })
            
            # Evaluate CASH EXCHANGE
            # Rule: Hands must be close for required_interaction_frames within the window
            last_event = self.last_event_frame.get(pair, -999)
            if frame_idx - last_event > self.cooldown_frames:
                
                interaction_count = sum(1 for entry in self.history[pair] if "HANDS_CLOSE" in entry["signals"])
                cash_count = sum(1 for entry in self.history[pair] if "CASH_DETECTED" in entry["signals"])
                
                if interaction_count >= self.required_interaction_frames:
                    # High confidence exchange
                    if cash_count > 0:
                        events.append(ExchangeEvent(
                            customer_id=pair[0],
                            cashier_id=pair[1],
                            timestamp=current_time,
                            frame_idx=frame_idx,
                            confidence=0.9,
                            reason=f"Hands interacted ({interaction_count} frames) + Cash detected"
                        ))
                    elif interaction_count >= int(self.fps * 1.5): # e.g. 37 frames
                        # Interaction without explicit cash detection -> inferred exchange
                        # We require a MUCH longer interaction (1.5s) if we don't see cash
                        events.append(ExchangeEvent(
                            customer_id=pair[0],
                            cashier_id=pair[1],
                            timestamp=current_time,
                            frame_idx=frame_idx,
                            confidence=0.7,
                            reason=f"Hands interacted ({interaction_count} frames), but no cash detected"
                        ))
                    if events:
                        self.last_event_frame[pair] = frame_idx
                    
        return events

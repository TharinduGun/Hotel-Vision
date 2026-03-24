"""
Fraud Detector Module (Layer 6)
===============================
Analyzes combined cash events and interaction timelines to flag
suspicious behaviors like pocketing or unregistered cash.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .interaction_analyzer import ExchangeEvent
from .cash_tracker import CashEvent, CashEventType


@dataclass
class FraudAlert:
    alert_type: str
    person_id: int
    timestamp: float
    frame_idx: int
    confidence: float
    description: str


class FraudDetector:
    """
    Evaluates business rules over the timeline of interactions and cash events
    to identify potentially fraudulent scenarios.
    """
    
    def __init__(
        self,
        register_wait_sec: float = 10.0,
        pocketing_window_sec: float = 5.0,
        fps: float = 25.0
    ):
        self.register_wait_frames = int(register_wait_sec * fps)
        self.pocketing_window_frames = int(pocketing_window_sec * fps)
        
        # State tracking
        self.pending_exchanges = {} # { event_id: ExchangeEvent }
        self.resolved_exchanges = set()
        
        # Track hand movements to register
        # { cashier_id: last_frame_registered }
        self.last_register_visit = {}
        
        self.event_counter = 0

    def evaluate(
        self,
        frame_idx: int,
        current_time: float,
        exchange_events: List[ExchangeEvent],
        cash_events: List[CashEvent],
        person_hands: Dict[int, List[Tuple[int, int]]],
        roles: Dict[int, str],
        roi_manager=None
    ) -> List[FraudAlert]:
        alerts = []
        
        # 1. Update hand positions to check for register visits
        if roi_manager and roi_manager.has_zones:
            for pid, hands in person_hands.items():
                if roles.get(pid) == "Cashier":
                    for pt in hands:
                        _, zone_type = roi_manager.get_zone_with_type(pt[0], pt[1])
                        if zone_type == "cash_register":
                            self.last_register_visit[pid] = frame_idx
                            break
                            
        # 2. Register new exchange events to watch
        for evt in exchange_events:
            self.event_counter += 1
            self.pending_exchanges[self.event_counter] = evt
            
        # 3. Process cash events (specifically pocketing)
        pocket_events = [e for e in cash_events if e.event_type == CashEventType.CASH_POCKET]
        for pkt in pocket_events:
            # Rule: Cash returned to pocket after exchange
            # Look for recent exchanges involving this person
            recent_exchanges = [
                (eid, evt) for eid, evt in self.pending_exchanges.items()
                if (evt.customer_id == pkt.person_id or evt.cashier_id == pkt.person_id)
                and (frame_idx - evt.frame_idx) <= self.pocketing_window_frames
            ]
            
            # We only care if a *Cashier* pockets cash. Customers naturally pocket change.
            role_str = roles.get(pkt.person_id, "Unknown")
            if role_str != "Cashier":
                # If they are tied to a recent exchange, we still want to resolve the exchange
                # so it doesn't trigger UNREGISTERED_CASH on the cashier
                if recent_exchanges:
                    for eid, _ in recent_exchanges:
                        self.resolved_exchanges.add(eid)
                continue
                
            if recent_exchanges:
                for eid, evt in recent_exchanges:
                    alerts.append(FraudAlert(
                        alert_type="POSSIBLE_POCKETING",
                        person_id=pkt.person_id,
                        timestamp=current_time,
                        frame_idx=frame_idx,
                        confidence=0.85, # High confidence if directly tied to recent exchange
                        description=f"{role_str} {pkt.person_id} pocketed cash shortly after an exchange."
                    ))
                    self.resolved_exchanges.add(eid)
            else:
                # Regular pocketing (no recent exchange tied directly)
                 alerts.append(FraudAlert(
                    alert_type="CASH_POCKETED",
                    person_id=pkt.person_id,
                    timestamp=current_time,
                    frame_idx=frame_idx,
                    confidence=0.6,
                    description=f"Cashier {pkt.person_id} pocketed cash."
                ))

        # 4. Check for unresolved exchanges (Rule: Cash Taken but Not Registered)
        expired_exchanges = []
        for eid, evt in self.pending_exchanges.items():
            if eid in self.resolved_exchanges:
                expired_exchanges.append(eid)
                continue
                
            elapsed_frames = frame_idx - evt.frame_idx
            if elapsed_frames > self.register_wait_frames:
                # Time's up. Did the cashier visit the register?
                last_visit = self.last_register_visit.get(evt.cashier_id, -999)
                
                if last_visit < evt.frame_idx:
                    # Cashier never went to the register since the exchange
                    alerts.append(FraudAlert(
                        alert_type="UNREGISTERED_CASH",
                        person_id=evt.cashier_id,
                        timestamp=current_time,
                        frame_idx=frame_idx,
                        confidence=0.75,
                        description=(f"Cashier {evt.cashier_id} engaged in an exchange "
                                     f"but hand did not visit register within {self.register_wait_frames / 25.0}s.")
                    ))
                
                expired_exchanges.append(eid)
                
        # Cleanup
        for eid in expired_exchanges:
            del self.pending_exchanges[eid]
            if eid in self.resolved_exchanges:
                self.resolved_exchanges.remove(eid)
                
        # 5. Rule: Hidden Cash Exchange
        # Handled in interaction_analyzer.py by inferring exchanges without cash detection.
        # If an inferred exchange triggers UNREGISTERED_CASH, it naturally captures this.
                
        return alerts

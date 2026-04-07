"""
Writes structured events to CSV and optionally sends via WebSocket.

Events are deduplicated: the same employee+camera+zone+status combination
is only written once until the state changes.
"""

import csv
import os
from datetime import datetime
from services.websocket_client import WebSocketClient

EVENT_FILE = "data/events/events.csv"
CSV_HEADERS = ["timestamp", "employee_id", "camera_id", "zone", "status"]


class EventBuilder:

    def __init__(self, enable_websocket=False, ws_url="ws://localhost:8000/ws/events"):
        """
        Args:
            enable_websocket: Set True to connect and stream events via WebSocket.
            ws_url: WebSocket endpoint URL.
        """
        os.makedirs(os.path.dirname(EVENT_FILE), exist_ok=True)
        self._ensure_csv_headers()

        self.last_events = {}
        self.ws = None

        if enable_websocket:
            try:
                self.ws = WebSocketClient(url=ws_url)
                print(f"WebSocket connected: {ws_url}")
            except Exception as e:
                print(f"WebSocket connection failed: {e}. Events will only log to CSV.")

    def _ensure_csv_headers(self):
        """Write CSV headers if the file is new or empty."""
        if not os.path.exists(EVENT_FILE) or os.path.getsize(EVENT_FILE) == 0:
            with open(EVENT_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def write_event(self, employee_id, camera_id, zone, status):
        """
        Log an event if it represents a state change.

        Args:
            employee_id: Employee ID or label (e.g. 'E001', 'SYSTEM').
            camera_id: Camera identifier string.
            zone: Zone name where the event occurred.
            status: Status string (ON_DUTY, IDLE, ON_BREAK, OFFLINE, LONG_QUEUE).
        """
        dedup_key = f"{employee_id}_{camera_id}"
        current_state = f"{zone}_{status}"

        if self.last_events.get(dedup_key) == current_state:
            return

        self.last_events[dedup_key] = current_state
        timestamp = datetime.now().isoformat()

        event = {
            "timestamp": timestamp,
            "employee_id": employee_id,
            "camera_id": camera_id,
            "zone": zone,
            "status": status,
        }

        with open(EVENT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, employee_id, camera_id, zone, status])

        if self.ws is not None:
            try:
                self.ws.send_event(event)
            except Exception:
                pass
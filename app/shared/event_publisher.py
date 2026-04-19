"""
Event Publisher (Shared Service)
=================================
Collects AnalyticsEvents from all modules and writes them to
the output layer (CSV files for now, DB/WebSocket later).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TextIO

from app.contracts.event_schema import AnalyticsEvent


# CSV columns (superset of all modules)
EVENT_CSV_FIELDS = [
    "module", "camera_id", "timestamp", "event_type", "confidence",
    "bbox", "severity", "frame_idx", "person_id",
    "snapshot_path", "clip_path", "iso_timestamp",
]

# Backend-compatible tracking summary columns (read by backend/services/csv_adapter.py)
TRACKING_SUMMARY_FIELDS = [
    "Split", "ID", "Class", "Role",
    "Start_Time_Sec", "End_Time_Sec", "Frame_Count",
    "Zone", "Camera_ID", "Session_Start",
    "Bbox_Start", "Bbox_End", "Dwell_Category",
    "Employee_ID",
]

# Map staff event_type → Dwell_Category for the backend alert rules
_STAFF_DWELL_MAP = {
    "employee_idle":     "LONG",
    "employee_on_break": "NORMAL",
    "employee_offline":  "NORMAL",
}


class EventPublisher:
    """
    Collects events from modules and persists them.

    Current backends:
      - CSV file (one per session)

    Future backends:
      - Database (SQLite / PostgreSQL)
      - WebSocket push
      - Message queue (Redis / RabbitMQ)
    """

    def __init__(self, output_dir: str, session_name: str | None = None):
        """
        Args:
            output_dir: Base directory for output logs.
            session_name: Session folder name (auto-generated if None).
        """
        from datetime import datetime

        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session_dir = Path(output_dir) / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._events: list[AnalyticsEvent] = []
        
        # Unified analytics events (new modular format)
        self._csv_path = self.session_dir / "analytics_events.csv"
        self._csv_file: TextIO | None = None
        self._csv_writer: csv.DictWriter | None = None

        # Backend-readable tracking summary (discovered by backend/config.py:discover_latest_csv)
        self._staff_csv_path = self.session_dir / "tracking_summary.csv"
        self._staff_file: TextIO | None = None
        self._staff_writer: csv.DictWriter | None = None

        # Legacy split CSV paths (to maintain backward compatibility with dashboard)
        self._cash_events_path = self.session_dir / "cash_events.csv"
        self._cash_file: TextIO | None = None
        self._cash_writer: csv.DictWriter | None = None

        self._exchange_path = self.session_dir / "exchange_events.csv"
        self._exchange_file: TextIO | None = None
        self._exchange_writer: csv.DictWriter | None = None

        self._fraud_path = self.session_dir / "fraud_alerts.csv"
        self._fraud_file: TextIO | None = None
        self._fraud_writer: csv.DictWriter | None = None

        # Open all CSVs for streaming writes
        self._open_csvs()

        print(f"[EventPublisher] Session: {self.session_dir}")
        print(f"[EventPublisher] Unified Events: {self._csv_path}")
        print(f"[EventPublisher] Tracking Summary (backend): {self._staff_csv_path}")
        print(f"[EventPublisher] Legacy CSVs: cash_events.csv, exchange_events.csv, fraud_alerts.csv")

    def _open_csvs(self):
        """Open all CSV files and write their headers."""
        # 1. Unified modular format
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=EVENT_CSV_FIELDS, extrasaction="ignore"
        )
        self._csv_writer.writeheader()

        # 2. Backend-compatible tracking summary (staff events in TrackingEvent schema)
        self._staff_file = open(self._staff_csv_path, "w", newline="", encoding="utf-8")
        self._staff_writer = csv.DictWriter(
            self._staff_file, fieldnames=TRACKING_SUMMARY_FIELDS
        )
        self._staff_writer.writeheader()

        # 3. Legacy Cash Events
        self._cash_file = open(self._cash_events_path, "w", newline="", encoding="utf-8")
        self._cash_writer = csv.DictWriter(
            self._cash_file, 
            fieldnames=["event_id", "timestamp", "frame", "person_id", "zone", "event_type", "confidence", "bbox_snapshot", "partner_id"]
        )
        self._cash_writer.writeheader()

        # 3. Legacy Exchange Events
        self._exchange_file = open(self._exchange_path, "w", newline="", encoding="utf-8")
        self._exchange_writer = csv.DictWriter(
            self._exchange_file,
            fieldnames=["event_id", "timestamp", "frame", "customer_id", "cashier_id", "confidence", "reason"]
        )
        self._exchange_writer.writeheader()

        # 4. Legacy Fraud Alerts
        self._fraud_file = open(self._fraud_path, "w", newline="", encoding="utf-8")
        self._fraud_writer = csv.DictWriter(
            self._fraud_file,
            fieldnames=["alert_id", "timestamp", "frame", "alert_type", "person_id", "confidence", "description"]
        )
        self._fraud_writer.writeheader()

    def publish(self, event: AnalyticsEvent) -> None:
        """
        Publish a single event: buffer it and write to CSV.
        """
        self._events.append(event)

        # 1. Write to unified CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow(event.to_dict())
            self._csv_file.flush()

        # 2. Write to backend-compatible tracking_summary.csv for staff events
        if (
            self._staff_writer is not None
            and event.module == "staff_tracking"
            and event.event_type in _STAFF_DWELL_MAP
        ):
            bbox = event.bbox or [0, 0, 0, 0]
            self._staff_writer.writerow({
                "Split":          0,
                "ID":             event.person_id if event.person_id is not None else -1,
                "Class":          "person",
                "Role":           "Cashier",
                "Start_Time_Sec": round(event.timestamp, 3),
                "End_Time_Sec":   round(event.timestamp, 3),
                "Frame_Count":    event.frame_idx,
                "Zone":           event.metadata.get("zone", "Unknown"),
                "Camera_ID":      event.camera_id,
                "Session_Start":  event.iso_timestamp or "",
                "Bbox_Start":     f"{int(bbox[0])},{int(bbox[1])}",
                "Bbox_End":       f"{int(bbox[2])},{int(bbox[3])}",
                "Dwell_Category": _STAFF_DWELL_MAP[event.event_type],
                "Employee_ID":    event.metadata.get("employee_id", ""),
            })
            self._staff_file.flush()

        # 3. Write to Legacy CSVs based on event type
        if self._cash_writer is not None and event.event_type in [
            "cash_pickup", "cash_deposit", "cash_handover", "cash_pocket", "cash_outside_zone"
        ]:
            self._cash_writer.writerow({
                "event_id": f"evt_{int(event.timestamp*1000)}",
                "timestamp": round(event.timestamp, 2),
                "frame": event.frame_idx,
                "person_id": event.person_id,
                "zone": event.metadata.get("zone", ""),
                "event_type": event.event_type.replace("cash_", "").upper(), # Match old format
                "confidence": round(event.confidence, 3),
                "bbox_snapshot": ",".join(map(str, event.bbox)) if event.bbox else "",
                "partner_id": event.metadata.get("partner_id", "")
            })
            self._cash_file.flush()

        elif self._exchange_writer is not None and event.event_type == "cash_exchange":
            persons = event.metadata.get("persons", [])
            customer_id = persons[0] if len(persons) > 0 else ""
            cashier_id = persons[1] if len(persons) > 1 else ""
            self._exchange_writer.writerow({
                "event_id": f"exc_{int(event.timestamp*1000)}",
                "timestamp": round(event.timestamp, 2),
                "frame": event.frame_idx,
                "customer_id": customer_id,
                "cashier_id": cashier_id,
                "confidence": round(event.confidence, 3),
                "reason": event.metadata.get("reason", "")
            })
            self._exchange_file.flush()

        elif self._fraud_writer is not None and event.event_type.startswith("fraud_"):
            self._fraud_writer.writerow({
                "alert_id": f"alt_{int(event.timestamp*1000)}",
                "timestamp": round(event.timestamp, 2),
                "frame": event.frame_idx,
                "alert_type": event.metadata.get("alert_type", event.event_type).upper(),
                "person_id": event.person_id or "",
                "confidence": round(event.confidence, 3),
                "description": event.metadata.get("description", "")
            })
            self._fraud_file.flush()

        # Log high-severity events
        if event.severity.value in ("high", "critical"):
            print(f"  🚨 [{event.module}] {event.event_type} "
                  f"(conf={event.confidence:.2f}, cam={event.camera_id})")

    def publish_batch(self, events: list[AnalyticsEvent]) -> None:
        """Publish multiple events at once."""
        for event in events:
            self.publish(event)

    def get_all_events(self) -> list[AnalyticsEvent]:
        """Return all accumulated events."""
        return list(self._events)

    def get_events_by_module(self, module_name: str) -> list[AnalyticsEvent]:
        """Filter events by module name."""
        return [e for e in self._events if e.module == module_name]

    def get_summary(self) -> dict:
        """Return event counts by module and severity."""
        by_module: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for e in self._events:
            by_module[e.module] = by_module.get(e.module, 0) + 1
            sev = e.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        return {
            "total_events": len(self._events),
            "by_module": by_module,
            "by_severity": by_severity,
            "session_dir": str(self.session_dir),
        }

    def shutdown(self):
        """Close the CSV files and print summary."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        if self._staff_file is not None:
            self._staff_file.close()
            self._staff_file = None
            self._staff_writer = None

        if self._cash_file is not None:
            self._cash_file.close()
            self._cash_file = None

        if self._exchange_file is not None:
            self._exchange_file.close()
            self._exchange_file = None

        if self._fraud_file is not None:
            self._fraud_file.close()
            self._fraud_file = None

        summary = self.get_summary()
        print(f"\n[EventPublisher] Session complete:")
        print(f"  Total events: {summary['total_events']}")
        for mod, count in summary["by_module"].items():
            print(f"    {mod}: {count}")
        print(f"  Events CSV: {self._csv_path}")

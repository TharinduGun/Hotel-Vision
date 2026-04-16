"""
CSV Adapter — reads analytics_events.csv and normalizes them into TrackingEvent objects.

Design:
  - Reads the new AnalyticsEvent CSV format (module, camera_id, timestamp, event_type, ...)
  - Maps events into the existing TrackingEvent model that the frontend expects
  - Backward compatible: also reads old tracking_summary.csv if found
  - This is the ONLY file that changes when you swap CSV → DB / RTSP stream

New CSV columns (from app/contracts/event_schema.py):
  module, camera_id, timestamp, event_type, confidence, bbox,
  severity, frame_idx, person_id, snapshot_path, clip_path, iso_timestamp,
  meta_zone, meta_role, meta_partner_id, meta_description, ...
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from backend.config import discover_latest_csv
from backend.models import TrackingEvent

logger = logging.getLogger(__name__)


# ── Old-format column mapping (backward compatibility) ─────────────────
OLD_COLUMN_MAP: dict[str, list[str]] = {
    "split":        ["split", "split_idx", "part"],
    "trackId":      ["id", "track_id", "tid", "object_id"],
    "objectClass":  ["class", "object_class", "cls", "type"],
    "role":         ["role", "person_role", "classification"],
    "startTimeSec": ["start_time_sec", "start_time", "start_sec", "start"],
    "endTimeSec":   ["end_time_sec", "end_time", "end_sec", "end"],
    "frameCount":   ["frame_count", "frame_cnt", "frames", "count"],
    "zone":         ["zone", "roi", "region", "area"],
    "cameraId":     ["camera_id", "cameraid", "cam_id", "camera"],
    "sessionStart": ["session_start", "session_ts", "session_datetime"],
    "bboxStart":    ["bbox_start", "start_bbox", "box_start"],
    "bboxEnd":      ["bbox_end", "end_bbox", "box_end"],
    "dwellCategory":["dwell_category", "dwell_cat", "dwell_class"],
    "cashEventType": ["cash_event_type", "cash_event", "cash_type"],
    "cashConfidence": ["cash_confidence", "cash_conf"],
    "cashPartnerId": ["cash_partner_id", "partner_id", "cash_partner"],
}

DEFAULTS: dict[str, object] = {
    "split": 0,
    "trackId": -1,
    "objectClass": "unknown",
    "role": "Unknown",
    "startTimeSec": 0.0,
    "endTimeSec": 0.0,
    "frameCount": 0,
    "zone": "Unknown",
    "cameraId": "CAM-01",
    "sessionStart": None,
    "bboxStart": None,
    "bboxEnd": None,
    "dwellCategory": "NORMAL",
    "cashEventType": None,
    "cashConfidence": None,
    "cashPartnerId": None,
}


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _is_new_format(header: list[str]) -> bool:
    """Check if CSV uses the new AnalyticsEvent format."""
    header_lower = {h.strip().lower() for h in header}
    # New format has 'module' and 'event_type' columns
    return "module" in header_lower and "event_type" in header_lower


def _resolve_columns_old(header: list[str]) -> dict[str, str | None]:
    """Given an old-format CSV header, map internal fields to column names."""
    header_lower = {h.strip().lower(): h.strip() for h in header}
    resolved: dict[str, str | None] = {}

    for field_name, candidates in OLD_COLUMN_MAP.items():
        matched = None
        for candidate in candidates:
            if candidate.lower() in header_lower:
                matched = header_lower[candidate.lower()]
                break
        if matched is None:
            logger.warning("CSV column for '%s' not found — using default: %s", field_name, DEFAULTS.get(field_name))
        resolved[field_name] = matched

    return resolved


# ── Event type to class mapping for new format ─────────────────────────

# Maps event_type → objectClass for the TrackingEvent model
_EVENT_TYPE_TO_CLASS = {
    # Cash detection events
    "CASH_PICKUP": "person",
    "CASH_DEPOSIT": "person",
    "CASH_HANDOVER": "person",
    "CASH_POCKET": "person",
    "CASH_OUTSIDE_ZONE": "person",
    "cash_exchange": "person",
    # Fraud alerts
    "fraud_cash_pocketed": "person",
    "fraud_possible_pocketing": "person",
    "fraud_unregistered_cash": "person",
    # Crowd detection events
    "density_change": "crowd",
    "person_entry": "person",
    "person_exit": "person",
    "occupancy_warning": "crowd",
    # Gun detection events
    "weapon_detected": "weapon",
    "weapon_alert": "weapon",
}

# Event types that are cash-related (used to populate cashEventType)
_CASH_EVENT_TYPES = {
    "CASH_PICKUP", "CASH_DEPOSIT", "CASH_HANDOVER",
    "CASH_POCKET", "CASH_OUTSIDE_ZONE",
}

# Severity mapping
_SEVERITY_MAP = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "critical": "CRITICAL",
}


def _parse_new_format_row(row: dict) -> Optional[TrackingEvent]:
    """Parse a single row from the new AnalyticsEvent CSV format."""
    try:
        event_type = row.get("event_type", "").strip()
        module = row.get("module", "").strip()
        camera_id = row.get("camera_id", "CAM-01").strip()
        timestamp = _safe_float(row.get("timestamp", "0"))
        confidence = _safe_float(row.get("confidence", "0"))
        severity = row.get("severity", "low").strip().lower()
        frame_idx = _safe_int(row.get("frame_idx", "0"))
        person_id = _safe_int(row.get("person_id", ""), -1)
        iso_timestamp = row.get("iso_timestamp", "").strip() or None
        bbox_str = row.get("bbox", "").strip()
        snapshot_path = row.get("snapshot_path", "").strip() or None
        
        # Determine object class from event type
        obj_class = _EVENT_TYPE_TO_CLASS.get(event_type, "event")
        
        # Extract metadata fields (prefixed with meta_)
        zone = row.get("meta_zone", "Unknown").strip() or "Unknown"
        role = row.get("meta_role", "Unknown").strip() or "Unknown"
        partner_id = row.get("meta_partner_id", "").strip()
        description = row.get("meta_description", "").strip()
        
        # Determine dwell category from severity
        dwell_category = "NORMAL"
        if severity in ("high", "critical"):
            dwell_category = "EXCESSIVE"
        elif severity == "medium":
            dwell_category = "LONG"
        
        # Cash event type mapping
        cash_event_type = None
        cash_confidence = None
        cash_partner_id = None
        
        if event_type in _CASH_EVENT_TYPES:
            cash_event_type = event_type
            cash_confidence = confidence
            if partner_id:
                cash_partner_id = _safe_int(partner_id, None)
        elif event_type.startswith("fraud_"):
            # Map fraud alerts to their cash event equivalent
            cash_event_type = event_type.upper().replace("FRAUD_", "")
            cash_confidence = confidence

        return TrackingEvent(
            split=0,
            trackId=person_id if person_id >= 0 else frame_idx,
            objectClass=obj_class,
            role=role,
            startTimeSec=timestamp,
            endTimeSec=timestamp + 0.04,  # ~1 frame at 25fps
            frameCount=1,
            zone=zone,
            cameraId=camera_id,
            sessionStart=iso_timestamp,
            bboxStart=bbox_str if bbox_str else None,
            bboxEnd=bbox_str if bbox_str else None,
            dwellCategory=dwell_category,
            cashEventType=cash_event_type,
            cashConfidence=cash_confidence,
            cashPartnerId=cash_partner_id,
        )
    except Exception as e:
        logger.warning("Failed to parse new-format row: %s", e)
        return None


def _parse_old_format_row(row: dict, col_map: dict) -> Optional[TrackingEvent]:
    """Parse a single row from the old tracking_summary.csv format."""
    try:
        session_start_raw = row.get(col_map.get("sessionStart") or "", "").strip() or None
        bbox_start_raw = row.get(col_map.get("bboxStart") or "", "").strip() or None
        bbox_end_raw = row.get(col_map.get("bboxEnd") or "", "").strip() or None
        dwell_cat_raw = row.get(col_map.get("dwellCategory") or "", "").strip() or DEFAULTS["dwellCategory"]

        return TrackingEvent(
            split=_safe_int(row.get(col_map["split"] or "", ""), DEFAULTS["split"]),
            trackId=_safe_int(row.get(col_map["trackId"] or "", ""), DEFAULTS["trackId"]),
            objectClass=row.get(col_map["objectClass"] or "", DEFAULTS["objectClass"]).strip().lower(),
            role=row.get(col_map["role"] or "", DEFAULTS["role"]).strip(),
            startTimeSec=_safe_float(row.get(col_map["startTimeSec"] or "", ""), DEFAULTS["startTimeSec"]),
            endTimeSec=_safe_float(row.get(col_map["endTimeSec"] or "", ""), DEFAULTS["endTimeSec"]),
            frameCount=_safe_int(row.get(col_map["frameCount"] or "", ""), DEFAULTS["frameCount"]),
            zone=row.get(col_map["zone"] or "", DEFAULTS["zone"]).strip(),
            cameraId=row.get(col_map["cameraId"] or "", DEFAULTS["cameraId"]).strip(),
            sessionStart=session_start_raw,
            bboxStart=bbox_start_raw,
            bboxEnd=bbox_end_raw,
            dwellCategory=dwell_cat_raw,
            cashEventType=row.get(col_map.get("cashEventType") or "", "").strip() or None,
            cashConfidence=_safe_float(
                row.get(col_map.get("cashConfidence") or "", ""), None
            ) if row.get(col_map.get("cashConfidence") or "", "").strip() else None,
            cashPartnerId=_safe_int(
                row.get(col_map.get("cashPartnerId") or "", ""), None
            ) if row.get(col_map.get("cashPartnerId") or "", "").strip() else None,
        )
    except Exception as e:
        logger.warning("Failed to parse old-format row: %s", e)
        return None


class CSVDataSource:
    """
    Reads the latest analytics CSV and provides normalized TrackingEvent lists.
    
    Supports both:
      - New format: analytics_events.csv (from app/ pipeline)
      - Old format: tracking_summary.csv (from pycode/ pipeline)

    Usage:
        source = CSVDataSource()
        events = source.get_events()        # all events
        employees = source.get_employees()  # role == Cashier / staff
        customers = source.get_customers()  # role == Customer
    """

    def __init__(self, csv_path: Path | None = None):
        self._csv_path = csv_path or discover_latest_csv()
        self._events: list[TrackingEvent] | None = None
        self._last_mtime: float = 0.0
        self._last_row_count: int = 0

    @property
    def csv_path(self) -> Path | None:
        return self._csv_path

    @property
    def last_row_count(self) -> int:
        return self._last_row_count

    def refresh_if_needed(self) -> bool:
        """Re-read the CSV only if file has been modified since last read."""
        if self._csv_path is None:
            return False
        try:
            current_mtime = self._csv_path.stat().st_mtime
        except OSError:
            return False

        if current_mtime != self._last_mtime:
            self._events = None
            return True
        return False

    def _load(self) -> list[TrackingEvent]:
        """Parse the CSV file into TrackingEvent objects."""
        if self._csv_path is None or not self._csv_path.exists():
            logger.warning("No CSV file found at %s", self._csv_path)
            return []

        events: list[TrackingEvent] = []

        try:
            with open(self._csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    logger.warning("CSV file is empty or has no header: %s", self._csv_path)
                    return []

                is_new = _is_new_format(list(reader.fieldnames))
                
                if is_new:
                    logger.info("Detected new AnalyticsEvent CSV format")
                    for row_idx, row in enumerate(reader):
                        event = _parse_new_format_row(row)
                        if event:
                            events.append(event)
                else:
                    logger.info("Detected old tracking_summary CSV format")
                    col_map = _resolve_columns_old(list(reader.fieldnames))
                    for row_idx, row in enumerate(reader):
                        event = _parse_old_format_row(row, col_map)
                        if event:
                            events.append(event)

            self._last_mtime = self._csv_path.stat().st_mtime
            self._last_row_count = len(events)
            logger.info("Loaded %d events from %s", len(events), self._csv_path.name)

        except Exception as e:
            logger.error("Failed to read CSV %s: %s", self._csv_path, e)

        return events

    def get_events(self) -> list[TrackingEvent]:
        """All tracking events (cached, auto-refreshes on file change)."""
        self.refresh_if_needed()
        if self._events is None:
            self._events = self._load()
        return self._events

    def get_people(self) -> list[TrackingEvent]:
        """Only person class events."""
        return [e for e in self.get_events() if e.objectClass == "person"]

    def get_employees(self) -> list[TrackingEvent]:
        """People classified as Cashier / staff roles."""
        return [e for e in self.get_people() if e.role.lower() in ("cashier",)]

    def get_customers(self) -> list[TrackingEvent]:
        """People classified as Customer."""
        return [e for e in self.get_people() if e.role.lower() == "customer"]

    def get_new_events_since(self, last_count: int) -> list[TrackingEvent]:
        """Return only events added since last_count (for WebSocket)."""
        all_events = self.get_events()
        if len(all_events) > last_count:
            return all_events[last_count:]
        return []

"""
CSV Adapter — reads tracking CSVs and normalizes them into TrackingEvent objects.

Design:
  - Column mapping dictionary for tolerance (columns can be renamed/missing)
  - Graceful defaults for missing columns (warnings, not crashes)
  - discover_latest_csv() picks the newest valid session
  - This is the ONLY file that changes when you swap CSV → DB / RTSP stream

CSV columns expected (current):
  Split, ID, Class, Role, Start_Time_Sec, End_Time_Sec, Frame_Count, Zone,
  Camera_ID, Session_Start, Bbox_Start, Bbox_End, Dwell_Category
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from backend.config import discover_latest_csv
from backend.models import TrackingEvent

logger = logging.getLogger(__name__)

# ── Column mapping ─────────────────────────────────────────────────────
# Maps our internal field names → possible CSV header names (case-insensitive).
# First match wins. This makes the adapter tolerant to header renames.
COLUMN_MAP: dict[str, list[str]] = {
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
    # Cash handling fields
    "cashEventType": ["cash_event_type", "cash_event", "cash_type"],
    "cashConfidence": ["cash_confidence", "cash_conf"],
    "cashPartnerId": ["cash_partner_id", "partner_id", "cash_partner"],
    # Employee identity (stable ID from face recognition via employee_id metadata)
    "employeeId": ["employee_id", "emp_id", "staff_id"],
}

# Default values when a column is missing entirely
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
    # Cash handling defaults
    "cashEventType": None,
    "cashConfidence": None,
    "cashPartnerId": None,
    "employeeId": None,
}


def _resolve_columns(header: list[str]) -> dict[str, str | None]:
    """
    Given a CSV header row, return a mapping:
        internal_field_name → actual_csv_column_name (or None if missing)
    """
    header_lower = {h.strip().lower(): h.strip() for h in header}
    resolved: dict[str, str | None] = {}

    for field, candidates in COLUMN_MAP.items():
        matched = None
        for candidate in candidates:
            if candidate.lower() in header_lower:
                matched = header_lower[candidate.lower()]
                break
        if matched is None:
            logger.warning("CSV column for '%s' not found — using default: %s", field, DEFAULTS.get(field))
        resolved[field] = matched

    return resolved


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))   # handles "1.0" strings
    except (ValueError, TypeError):
        return default


class CSVDataSource:
    """
    Reads the latest tracking CSV and provides normalized TrackingEvent lists.

    Usage:
        source = CSVDataSource()
        events = source.get_events()        # all events
        employees = source.get_employees()  # role == Cashier / staff
        customers = source.get_customers()  # role == Customer
    """

    def __init__(self, csv_path: Path | None = None):
        self._csv_path = csv_path or discover_latest_csv()
        self._events: list[TrackingEvent] | None = None    # lazy cache
        self._last_mtime: float = 0.0
        self._last_row_count: int = 0

    @property
    def csv_path(self) -> Path | None:
        return self._csv_path

    @property
    def last_row_count(self) -> int:
        return self._last_row_count

    def refresh_if_needed(self) -> bool:
        """
        Re-read the CSV only if file has been modified since last read.
        Returns True if data was refreshed, False otherwise.
        """
        if self._csv_path is None:
            return False
        try:
            current_mtime = self._csv_path.stat().st_mtime
        except OSError:
            return False

        if current_mtime != self._last_mtime:
            self._events = None    # invalidate cache
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

                col_map = _resolve_columns(list(reader.fieldnames))

                for row_idx, row in enumerate(reader):
                    try:
                        # Read new optional columns (graceful if missing)
                        session_start_raw = row.get(col_map.get("sessionStart") or "", "").strip() or None
                        bbox_start_raw = row.get(col_map.get("bboxStart") or "", "").strip() or None
                        bbox_end_raw = row.get(col_map.get("bboxEnd") or "", "").strip() or None
                        dwell_cat_raw = row.get(col_map.get("dwellCategory") or "", "").strip() or DEFAULTS["dwellCategory"]

                        event = TrackingEvent(
                            split=_safe_int(
                                row.get(col_map["split"] or "", ""),
                                DEFAULTS["split"],
                            ),
                            trackId=_safe_int(
                                row.get(col_map["trackId"] or "", ""),
                                DEFAULTS["trackId"],
                            ),
                            objectClass=row.get(col_map["objectClass"] or "", DEFAULTS["objectClass"]).strip().lower(),
                            role=row.get(col_map["role"] or "", DEFAULTS["role"]).strip(),
                            startTimeSec=_safe_float(
                                row.get(col_map["startTimeSec"] or "", ""),
                                DEFAULTS["startTimeSec"],
                            ),
                            endTimeSec=_safe_float(
                                row.get(col_map["endTimeSec"] or "", ""),
                                DEFAULTS["endTimeSec"],
                            ),
                            frameCount=_safe_int(
                                row.get(col_map["frameCount"] or "", ""),
                                DEFAULTS["frameCount"],
                            ),
                            zone=row.get(col_map["zone"] or "", DEFAULTS["zone"]).strip(),
                            cameraId=row.get(col_map["cameraId"] or "", DEFAULTS["cameraId"]).strip(),
                            sessionStart=session_start_raw,
                            bboxStart=bbox_start_raw,
                            bboxEnd=bbox_end_raw,
                            dwellCategory=dwell_cat_raw,
                            # Cash handling fields (graceful if missing)
                            cashEventType=row.get(col_map.get("cashEventType") or "", "").strip() or None,
                            cashConfidence=_safe_float(
                                row.get(col_map.get("cashConfidence") or "", ""), None
                            ) if row.get(col_map.get("cashConfidence") or "", "").strip() else None,
                            cashPartnerId=_safe_int(
                                row.get(col_map.get("cashPartnerId") or "", ""), None
                            ) if row.get(col_map.get("cashPartnerId") or "", "").strip() else None,
                            employeeId=row.get(col_map.get("employeeId") or "", "").strip() or None,
                        )
                        events.append(event)
                    except Exception as e:
                        logger.warning("Skipping row %d: %s", row_idx, e)

            self._last_mtime = self._csv_path.stat().st_mtime
            self._last_row_count = len(events)

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
        """
        Return only events added since last_count.
        Used by WebSocket to avoid re-sending old data.
        """
        all_events = self.get_events()
        if len(all_events) > last_count:
            return all_events[last_count:]
        return []

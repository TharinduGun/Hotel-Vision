"""
GET /api/v1/staff — Staff tracking analytics endpoint.

Reads staff events from analytics_events.csv (module == 'staff_tracking')
and returns employee statuses, idle alerts, and queue summaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.csv_adapter import CSVDataSource

router = APIRouter(prefix="/staff", tags=["Staff Tracking"])


# ── Response models ────────────────────────────────────────────────────

class StaffEvent(BaseModel):
    employeeId: str
    cameraId: str
    eventType: str          # employee_idle | employee_on_break | employee_offline | long_queue
    severity: str
    zone: str
    ts: datetime
    queueCount: Optional[int] = None


class StaffSummary(BaseModel):
    activeEmployees: int
    idleAlerts: int
    onBreakCount: int
    offlineCount: int
    longQueueAlerts: int


class StaffResponse(BaseModel):
    ts: datetime
    summary: StaffSummary
    recentEvents: list[StaffEvent]


# ── Route ──────────────────────────────────────────────────────────────

@router.get("", response_model=StaffResponse)
async def get_staff_analytics(limit: int = 50):
    """
    Returns the latest staff tracking events and a summary.

    Args:
        limit: Max number of recent events to return (default 50).
    """
    source = CSVDataSource()
    all_events = source.get_events()

    # Filter to staff_tracking module events only
    staff_events = [
        e for e in all_events
        if getattr(e, "module", None) == "staff_tracking"
           or getattr(e, "objectClass", "") == "staff"
    ]

    # If the CSV adapter uses the unified analytics_events.csv format, the
    # fields come through differently — handle both formats gracefully.
    recent: list[StaffEvent] = []
    counts = {"idle": 0, "on_break": 0, "offline": 0, "queue": 0, "active_ids": set()}

    for ev in staff_events[-limit:]:
        event_type = (
            getattr(ev, "event_type", None)
            or getattr(ev, "status", "unknown")
        )
        employee_id = (
            getattr(ev, "employee_id", None)
            or getattr(ev, "trackId", "UNKNOWN")
        )
        zone = getattr(ev, "zone", "unknown")
        severity = getattr(ev, "severity", "low")
        ts_raw = getattr(ev, "iso_timestamp", None) or getattr(ev, "sessionStart", None)

        try:
            ts = datetime.fromisoformat(str(ts_raw)) if ts_raw else datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        queue_count = None
        if event_type == "long_queue":
            counts["queue"] += 1
            queue_count = getattr(ev, "meta_queue_count", None)
        elif event_type == "employee_idle":
            counts["idle"] += 1
            counts["active_ids"].add(str(employee_id))
        elif event_type == "employee_on_break":
            counts["on_break"] += 1
            counts["active_ids"].add(str(employee_id))
        elif event_type == "employee_offline":
            counts["offline"] += 1
        else:
            counts["active_ids"].add(str(employee_id))

        recent.append(StaffEvent(
            employeeId=str(employee_id),
            cameraId=getattr(ev, "cameraId", "CAM-01"),
            eventType=event_type,
            severity=str(severity).lower(),
            zone=str(zone),
            ts=ts,
            queueCount=queue_count,
        ))

    summary = StaffSummary(
        activeEmployees=len(counts["active_ids"]),
        idleAlerts=counts["idle"],
        onBreakCount=counts["on_break"],
        offlineCount=counts["offline"],
        longQueueAlerts=counts["queue"],
    )

    return StaffResponse(
        ts=datetime.now(timezone.utc),
        summary=summary,
        recentEvents=recent,
    )

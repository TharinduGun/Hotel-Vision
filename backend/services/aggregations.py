"""
Aggregation service — derives dashboard KPIs, alerts, and employee status
from the raw TrackingEvent list provided by the CSV adapter.

All business logic for turning raw events into API responses lives here.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from backend.config import CAMERAS
from backend.models import (
    Alert,
    AlertEvidence,
    AlertsResponse,
    Camera,
    CameraSnapshot,
    DashboardSummary,
    Employee,
    EmployeesResponse,
    KPICard,
    TrackingEvent,
)

# Placeholder employee name pool (when CSV doesn't have names)
_EMPLOYEE_NAMES = [
    "Sarah Johnson",
    "Michael Chen",
    "Priya Patel",
    "James Wilson",
    "Ana Rodriguez",
    "David Kim",
    "Emma Thompson",
    "Robert Singh",
]

_ROLE_DISPLAY = {
    "cashier": "Cashier",
    "customer": "Customer",
}

_LOCATION_MAP = {
    "Cashier 1": "POS Station 1",
    "Cashier 2": "POS Station 2",
    "Outside": "Main Lobby",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _event_id(event: TrackingEvent) -> str:
    """Deterministic hash-based ID for an event."""
    raw = f"{event.cameraId}-{event.trackId}-{event.startTimeSec}"
    return "AL-" + hashlib.md5(raw.encode()).hexdigest()[:6].upper()


# ── Dashboard Summary ──────────────────────────────────────────────────

def build_summary(events: list[TrackingEvent]) -> DashboardSummary:
    """Build the 6 KPI cards from current event data."""
    people = [e for e in events if e.objectClass == "person"]
    workers = [e for e in people if e.role.lower() in ("cashier",)]
    customers = [e for e in people if e.role.lower() == "customer"]

    cameras_online = sum(1 for c in CAMERAS if c["status"] == "online")
    cameras_total = len(CAMERAS)

    # Derive alerts count (events with long dwell or unusual zone)
    alerts = _derive_alerts(events)
    incidents_today = len(alerts)

    return DashboardSummary(
        ts=_now(),
        cards={
            "camerasOnline": KPICard(value=cameras_online, total=cameras_total),
            "activeAlerts": KPICard(value=incidents_today, deltaPct=-12.0),
            "staffOnSite": KPICard(value=len(workers)),
            "securityScore": KPICard(value=94, deltaPct=3.0),
            "incidentsToday": KPICard(value=incidents_today),
            "uptime": KPICard(value=99.9),
        },
    )


# ── Alerts ─────────────────────────────────────────────────────────────

def _derive_alerts(events: list[TrackingEvent]) -> list[Alert]:
    """
    Derive alerts from tracking events using simple rules:
    - Person in a cashier zone with high dwell time → UNUSUAL_MOTION
    - Customer in staff-only zones → ZONE_INTRUSION
    - Very long presence → LOITERING
    """
    alerts: list[Alert] = []

    for event in events:
        if event.objectClass != "person":
            continue

        dwell_sec = event.endTimeSec - event.startTimeSec

        # Rule 1: customer in cashier zone
        if event.role.lower() == "customer" and "cashier" in event.zone.lower():
            alerts.append(Alert(
                id=_event_id(event),
                type="ZONE_INTRUSION",
                severity="HIGH",
                title="Unauthorized Zone Access",
                description=f"Customer detected in {event.zone} for {dwell_sec:.1f}s.",
                cameraId=event.cameraId,
                zone=event.zone,
                ts=_now() - timedelta(seconds=max(0, 60 - dwell_sec)),
                evidence=AlertEvidence(),
            ))

        # Rule 2: long dwell (> 45 sec) anywhere
        elif dwell_sec > 45:
            alerts.append(Alert(
                id=_event_id(event),
                type="UNUSUAL_MOTION",
                severity="MEDIUM",
                title="Extended Presence Detected",
                description=f"{event.role} (Track #{event.trackId}) stayed in {event.zone} for {dwell_sec:.1f}s.",
                cameraId=event.cameraId,
                zone=event.zone,
                ts=_now() - timedelta(seconds=max(0, 60 - dwell_sec)),
                evidence=AlertEvidence(),
            ))

        # Rule 3: loitering (> 55 sec, many frames)
        elif dwell_sec > 55 and event.frameCount > 1000:
            alerts.append(Alert(
                id=_event_id(event),
                type="LOITERING",
                severity="LOW",
                title="Loitering Detected",
                description=f"Track #{event.trackId} loitering in {event.zone} for {dwell_sec:.1f}s ({event.frameCount} frames).",
                cameraId=event.cameraId,
                zone=event.zone,
                ts=_now() - timedelta(seconds=30),
                evidence=AlertEvidence(),
            ))

    return alerts


def build_alerts(
    events: list[TrackingEvent],
    status: str = "active",
    limit: int = 20,
) -> AlertsResponse:
    """Build the alerts response."""
    alerts = _derive_alerts(events)
    # For MVP, all derived alerts are "active"
    return AlertsResponse(items=alerts[:limit])


# ── Cameras ────────────────────────────────────────────────────────────

def build_cameras() -> list[Camera]:
    return [Camera(**c) for c in CAMERAS]


def build_snapshots() -> list[CameraSnapshot]:
    now = _now()
    return [
        CameraSnapshot(
            cameraId=c["id"],
            snapshotUrl=f"/media/snapshots/{c['id']}.jpg",
            ts=now,
        )
        for c in CAMERAS
    ]


# ── Employees ──────────────────────────────────────────────────────────

def build_employees(
    events: list[TrackingEvent],
    status_filter: str | None = None,
) -> EmployeesResponse:
    """
    Build employee list from tracking events.
    Groups by track ID for Cashier-role people.
    """
    workers = [e for e in events if e.objectClass == "person" and e.role.lower() in ("cashier",)]

    employees: list[Employee] = []
    for idx, w in enumerate(workers):
        name = _EMPLOYEE_NAMES[idx % len(_EMPLOYEE_NAMES)]
        location = _LOCATION_MAP.get(w.zone, w.zone)
        emp_status = "ON_DUTY" if w.endTimeSec > 30 else "BREAK"

        emp = Employee(
            id=f"E{w.trackId:03d}",
            name=name,
            role=_ROLE_DISPLAY.get(w.role.lower(), w.role),
            status=emp_status,
            lastSeen=_now() - timedelta(seconds=max(0, 60 - w.endTimeSec)),
            location=location,
            zone=w.zone,
        )
        employees.append(emp)

    if status_filter:
        employees = [e for e in employees if e.status == status_filter.upper()]

    return EmployeesResponse(items=employees)

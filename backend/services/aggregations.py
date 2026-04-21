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

# Face-recognition employee ID → display name
_ID_TO_NAME = {
    "E001": "Anabel",
    "E002": "Irving",
    "E003": "Rosa",
    "E004": "Yareht",
    "E005": "Aurora",
    "E006": "Alejandra",
    "E007": "Lilia",
    "E008": "Sandra",
    "E009": "Maria",
    "E010": "Lizeth",
    "E011": "Karlelis",
    "E012": "Geraldine",
    "E013": "Paulo",
    "E014": "Adela",
    "E015": "Maribel",
    "E016": "Lilianan",
    "E017": "Aditya",
    "E018": "Angela",
    "E019": "Manish",
    "E020": "Manidhar",
    "E021": "Supreeth",
    "E022": "Irma",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _session_time(event: TrackingEvent) -> datetime:
    """Get the real datetime for an event based on sessionStart + offset."""
    if event.sessionStart:
        try:
            base = datetime.fromisoformat(event.sessionStart)
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
            return base + timedelta(seconds=event.startTimeSec)
        except (ValueError, TypeError):
            pass
    # Fallback: offset from now
    return _now() - timedelta(seconds=max(0, 60 - event.startTimeSec))


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

    # Count cash transactions
    cash_events = [e for e in events if e.cashEventType is not None]
    cash_transactions = len(cash_events)

    # Compute security score from dwell categories + cash violations
    # Start at 100, penalize for LONG (-3), EXCESSIVE (-8), cash violations (-5)
    score = 100
    for e in people:
        if e.dwellCategory == "LONG":
            score -= 3
        elif e.dwellCategory == "EXCESSIVE":
            score -= 8

    # Penalize security score for cash violations
    cash_violations = [e for e in cash_events if e.cashEventType in (
        "CASH_OUTSIDE_ZONE", "CASH_POCKET", "CASH_HANDOVER"
    )]
    for _ in cash_violations:
        score -= 5
    score = max(0, min(100, score))  # clamp to 0-100

    # Compute uptime from cameras
    uptime = round((cameras_online / cameras_total) * 100, 1) if cameras_total > 0 else 0.0

    # Count unique person tracks as footfall
    unique_person_ids = set(e.trackId for e in people)

    return DashboardSummary(
        ts=_now(),
        cards={
            "camerasOnline": KPICard(value=cameras_online, total=cameras_total),
            "activeAlerts": KPICard(value=incidents_today, deltaPct=-12.0),
            "staffOnSite": KPICard(value=len(workers)),
            "securityScore": KPICard(value=score, deltaPct=3.0),
            "incidentsToday": KPICard(value=incidents_today),
            "uptime": KPICard(value=uptime),
            "cashTransactions": KPICard(value=cash_transactions),
            "peopleCounted": KPICard(value=len(unique_person_ids)),
        },
    )


# ── Alerts ─────────────────────────────────────────────────────────────

def _derive_alerts(events: list[TrackingEvent]) -> list[Alert]:
    """
    Derive alerts from tracking events using dwellCategory + zone rules:
    - Customer in cashier zone → ZONE_INTRUSION (HIGH)
    - LONG dwell → UNUSUAL_MOTION (MEDIUM)
    - EXCESSIVE dwell → LOITERING (HIGH)
    """
    alerts: list[Alert] = []

    for event in events:
        dwell_sec = event.endTimeSec - event.startTimeSec
        event_ts = _session_time(event)
        
        evidence = AlertEvidence(imageUrl=event.snapshotPath, clipUrl=event.clipPath)

        # ── Weapon / Crowd Rules ──────────────────────────────────────────
        if event.objectClass == "weapon":
            alerts.append(Alert(
                id=_event_id(event),
                type="WEAPON_DETECTED",
                severity="CRITICAL",
                title="Weapon Detected",
                description=f"Weapon detected on camera {event.cameraId}.",
                cameraId=event.cameraId,
                zone=event.zone,
                ts=event_ts,
                evidence=evidence,
            ))
            continue
            
        elif event.objectClass == "crowd":
            if event.dwellCategory in ("EXCESSIVE", "LONG"):
                sev = "CRITICAL" if event.dwellCategory == "EXCESSIVE" else "HIGH"
                alerts.append(Alert(
                    id=_event_id(event) + "-CROWD",
                    type="CROWD_ALERT",
                    severity=sev,
                    title="High Crowd Density",
                    description=f"High crowd density level detected on camera {event.cameraId}.",
                    cameraId=event.cameraId,
                    zone=event.zone,
                    ts=event_ts,
                    evidence=evidence,
                ))
            continue

        if event.objectClass != "person":
            continue

        # Rule 1: customer in cashier zone (any dwell)
        if event.role.lower() == "customer" and "cashier" in event.zone.lower():
            alerts.append(Alert(
                id=_event_id(event),
                type="ZONE_INTRUSION",
                severity="HIGH",
                title="Unauthorized Zone Access",
                description=f"Customer detected in {event.zone} for {dwell_sec:.1f}s.",
                cameraId=event.cameraId,
                zone=event.zone,
                ts=event_ts,
                evidence=evidence,
            ))

        # Rule 2: EXCESSIVE dwell → LOITERING
        elif event.dwellCategory == "EXCESSIVE":
            alerts.append(Alert(
                id=_event_id(event),
                type="LOITERING",
                severity="HIGH",
                title="Loitering Detected",
                description=f"Track #{event.trackId} loitering in {event.zone} for {dwell_sec:.1f}s ({event.frameCount} frames).",
                cameraId=event.cameraId,
                zone=event.zone,
                ts=event_ts,
                evidence=evidence,
            ))

        # Rule 3: LONG dwell → UNUSUAL_MOTION
        elif event.dwellCategory == "LONG":
            alerts.append(Alert(
                id=_event_id(event),
                type="UNUSUAL_MOTION",
                severity="MEDIUM",
                title="Extended Presence Detected",
                description=f"{event.role} (Track #{event.trackId}) stayed in {event.zone} for {dwell_sec:.1f}s.",
                cameraId=event.cameraId,
                zone=event.zone,
                ts=event_ts,
                evidence=evidence,
            ))

        # --- Cash-specific alert rules ---
        if event.cashEventType:
            if event.cashEventType == "CASH_OUTSIDE_ZONE":
                alerts.append(Alert(
                    id=_event_id(event) + "-CASH",
                    type="CASH_OUTSIDE_ZONE",
                    severity="HIGH",
                    title="Cash Outside Designated Zone",
                    description=f"Cash detected on {event.role} (Track #{event.trackId}) in {event.zone} — outside register/exchange zones.",
                    cameraId=event.cameraId,
                    zone=event.zone,
                    ts=event_ts,
                    evidence=evidence,
                ))
            elif event.cashEventType == "CASH_HANDOVER":
                partner_info = f" to Track #{event.cashPartnerId}" if event.cashPartnerId else ""
                sev = "CRITICAL" if event.role.lower() != "cashier" else "MEDIUM"
                alerts.append(Alert(
                    id=_event_id(event) + "-CASH",
                    type="CASH_HANDOVER_SUSPICIOUS",
                    severity=sev,
                    title="Cash Handover Detected",
                    description=f"{event.role} (Track #{event.trackId}) handed cash{partner_info} in {event.zone}.",
                    cameraId=event.cameraId,
                    zone=event.zone,
                    ts=event_ts,
                    evidence=evidence,
                ))
            elif event.cashEventType == "CASH_POCKET":
                alerts.append(Alert(
                    id=_event_id(event) + "-CASH",
                    type="CASH_POCKET",
                    severity="CRITICAL",
                    title="Possible Cash Pocketing",
                    description=f"Cash disappeared from {event.role} (Track #{event.trackId}) outside register zone in {event.zone}.",
                    cameraId=event.cameraId,
                    zone=event.zone,
                    ts=event_ts,
                    evidence=evidence,
                ))
            elif event.cashEventType in ("UNREGISTERED_CASH", "POSSIBLE_POCKETING"):
                alerts.append(Alert(
                    id=_event_id(event) + "-FRAUD",
                    type="UNREGISTERED_CASH_HANDLING" if event.cashEventType == "UNREGISTERED_CASH" else "CASH_POCKET",
                    severity="CRITICAL",
                    title="Unregistered Cash" if event.cashEventType == "UNREGISTERED_CASH" else "Possible Cash Pocketing",
                    description=f"Suspicious cash handling ({event.cashEventType.replace('_', ' ').lower()}) on Track #{event.trackId}.",
                    cameraId=event.cameraId,
                    zone=event.zone,
                    ts=event_ts,
                    evidence=evidence,
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
        # Use stable employee ID from face recognition when available;
        # fall back to tracker-generated ID for legacy/unrecognised tracks.
        if w.employeeId and w.employeeId not in ("unknown", ""):
            emp_id = w.employeeId
            name = _ID_TO_NAME.get(emp_id, emp_id)   # ID → real name
        else:
            emp_id = f"E{w.trackId:03d}"
            name = _EMPLOYEE_NAMES[idx % len(_EMPLOYEE_NAMES)] if _EMPLOYEE_NAMES else emp_id

        location = _LOCATION_MAP.get(w.zone, w.zone)
        emp_status = "ON_DUTY" if w.dwellCategory in ("NORMAL", "LONG", "EXCESSIVE") else "BREAK"

        # Use real session time if available
        last_seen_ts = _session_time(w)

        emp = Employee(
            id=emp_id,
            name=name,
            role=_ROLE_DISPLAY.get(w.role.lower(), w.role),
            status=emp_status,
            lastSeen=last_seen_ts,
            location=location,
            zone=w.zone,
        )
        employees.append(emp)

    if status_filter:
        employees = [e for e in employees if e.status == status_filter.upper()]

    return EmployeesResponse(items=employees)

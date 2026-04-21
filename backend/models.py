"""
Pydantic response schemas matching the UI dashboard API contract.

Every event/alert model includes cameraId (defaults to "CAM-01"
when the CSV source doesn't provide it).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Core event (internal, normalized from CSV) ─────────────────────────

class TrackingEvent(BaseModel):
    """One row from the tracking CSV, normalized."""
    split: int = 0
    trackId: int
    objectClass: str            # "person" | "car"
    role: str = "Unknown"       # "Cashier" | "Customer" | "N/A"
    startTimeSec: float
    endTimeSec: float
    frameCount: int
    zone: str = "Unknown"
    cameraId: str = "CAM-01"    # default when CSV lacks it
    # New columns for KPI alignment (optional for backward compat)
    sessionStart: Optional[str] = None      # ISO datetime of session start
    bboxStart: Optional[str] = None         # "x1,y1,x2,y2"
    bboxEnd: Optional[str] = None           # "x1,y1,x2,y2"
    dwellCategory: str = "NORMAL"           # SHORT | NORMAL | LONG | EXCESSIVE
    # Cash handling fields (Phase: Cash Handling & Reception Monitoring)
    cashEventType: Optional[str] = None      # CASH_PICKUP | CASH_DEPOSIT | CASH_HANDOVER | None
    cashConfidence: Optional[float] = None   # Detection confidence for cash
    cashPartnerId: Optional[int] = None      # Partner person ID for handover events
    # Stable employee identity from face recognition (e.g. "E001")
    employeeId: Optional[str] = None


# ── Dashboard Summary ──────────────────────────────────────────────────

class KPICard(BaseModel):
    value: float | int
    total: Optional[int] = None
    deltaPct: Optional[float] = None


class DashboardSummary(BaseModel):
    ts: datetime
    cards: dict[str, KPICard]


# ── Cameras ────────────────────────────────────────────────────────────

class Camera(BaseModel):
    id: str
    name: str
    location: str
    status: str                 # "online" | "offline"
    streamUrl: str


class CameraSnapshot(BaseModel):
    cameraId: str
    snapshotUrl: str
    ts: datetime


# ── Alerts ─────────────────────────────────────────────────────────────

class AlertEvidence(BaseModel):
    imageUrl: Optional[str] = None
    clipUrl: Optional[str] = None


class Alert(BaseModel):
    id: str
    type: str                   # "UNUSUAL_MOTION" | "ZONE_INTRUSION" | "LOITERING" |
                                # "CASH_OUTSIDE_ZONE" | "CASH_HANDOVER_SUSPICIOUS" |
                                # "UNREGISTERED_CASH_HANDLING" | "CASH_POCKET"
    severity: str               # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    title: str
    description: str
    cameraId: str = "CAM-01"
    zone: str = "Unknown"
    ts: datetime
    evidence: AlertEvidence = Field(default_factory=AlertEvidence)


class AlertsResponse(BaseModel):
    items: list[Alert]


# ── Employees ──────────────────────────────────────────────────────────

class Employee(BaseModel):
    id: str
    name: str
    role: str
    status: str                 # "ON_DUTY" | "BREAK" | "OFF_DUTY"
    lastSeen: datetime
    location: str
    zone: str


class EmployeesResponse(BaseModel):
    items: list[Employee]


# ── Crowd Detection ───────────────────────────────────────────────────

class CrowdFootfallEvent(BaseModel):
    trackId: int
    direction: str              # "entry" | "exit"
    timestampSec: float
    frameIdx: int
    edge: str                   # "top" | "bottom" | "left" | "right"
    positionX: float
    positionY: float


class CrowdDwellRecord(BaseModel):
    trackId: int
    entryTimeSec: float
    exitTimeSec: float
    durationSec: float
    entryX: float
    entryY: float
    exitX: float
    exitY: float


class CrowdSummary(BaseModel):
    totalEntries: int = 0
    totalExits: int = 0
    peakOccupancy: int = 0
    avgOccupancy: float = 0.0
    currentDensity: str = "low"         # low | moderate | high | critical
    avgDwellSec: float = 0.0
    maxDwellSec: float = 0.0
    totalUniquePersons: int = 0


class CrowdInsightsResponse(BaseModel):
    ts: datetime
    summary: CrowdSummary
    recentFootfall: list[CrowdFootfallEvent] = []
    recentDwells: list[CrowdDwellRecord] = []
    heatmapUrl: Optional[str] = None


# ── Combined live state ────────────────────────────────────────────────

class LiveState(BaseModel):
    summary: DashboardSummary
    snapshots: list[CameraSnapshot]
    alerts: AlertsResponse
    employees: EmployeesResponse
    crowd: Optional[CrowdInsightsResponse] = None



# ── WebSocket message envelope ─────────────────────────────────────────

class WSMessage(BaseModel):
    kind: str                   # "alert_new" | "summary_update" | ...
    payload: dict


# ── Health ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    ts: datetime
    dataSource: str = "csv"
    latestCsv: Optional[str] = None

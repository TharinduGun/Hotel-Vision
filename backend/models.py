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
    type: str                   # "UNUSUAL_MOTION" | "ZONE_INTRUSION" | ...
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


# ── Combined live state ────────────────────────────────────────────────

class LiveState(BaseModel):
    summary: DashboardSummary
    snapshots: list[CameraSnapshot]
    alerts: AlertsResponse
    employees: EmployeesResponse


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

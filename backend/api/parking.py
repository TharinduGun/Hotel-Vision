"""
GET /api/v1/parking — Parking lot analytics endpoint.

Reads parking events from analytics_events.csv (module == 'parking')
and returns current occupancy status, history, and alerts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.csv_adapter import CSVDataSource

router = APIRouter(prefix="/parking", tags=["Parking"])


# ── Response models ────────────────────────────────────────────────────

class ParkingStatus(BaseModel):
    occupied: int = 0
    available: int = 0
    capacity: int = 0
    occupancyPct: float = 0.0
    status: str = "available"          # available | limited | full
    avgDwellSeconds: float = 0.0
    totalVehiclesSeen: int = 0
    vehicleTypes: dict = {}


class ParkingAlert(BaseModel):
    eventType: str                     # parking_full | parking_limited | long_dwell
    severity: str
    cameraId: str
    ts: datetime
    occupancyPct: Optional[float] = None
    vehicleId: Optional[int] = None
    dwellMinutes: Optional[float] = None


class ParkingResponse(BaseModel):
    ts: datetime
    current: ParkingStatus
    recentAlerts: list[ParkingAlert]
    totalAlerts: int


# ── Route ──────────────────────────────────────────────────────────────

@router.get("", response_model=ParkingResponse)
async def get_parking_analytics(limit: int = 50):
    """
    Returns the latest parking occupancy state and recent alerts.
    """
    source = CSVDataSource()
    all_events = source.get_events()

    parking_events = [
        e for e in all_events
        if getattr(e, "module", None) == "parking"
    ]

    current = ParkingStatus()
    alerts: list[ParkingAlert] = []

    # Walk events newest-first — take the first status_update as current state
    for ev in reversed(parking_events):
        event_type = getattr(ev, "event_type", getattr(ev, "status", ""))
        ts_raw = getattr(ev, "iso_timestamp", None) or getattr(ev, "sessionStart", None)

        try:
            ts = datetime.fromisoformat(str(ts_raw)) if ts_raw else datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        camera_id = getattr(ev, "cameraId", "CAM-02")

        # Populate current state from the most recent status_update
        if event_type == "parking_status_update" and current.capacity == 0:
            current = ParkingStatus(
                occupied=int(getattr(ev, "meta_occupied", 0) or 0),
                available=int(getattr(ev, "meta_available", 0) or 0),
                capacity=int(getattr(ev, "meta_capacity", 0) or 0),
                occupancyPct=float(getattr(ev, "meta_occupancy_pct", 0.0) or 0.0),
                status=str(getattr(ev, "meta_status", "available") or "available"),
                avgDwellSeconds=float(getattr(ev, "meta_avg_dwell_seconds", 0.0) or 0.0),
                totalVehiclesSeen=int(getattr(ev, "meta_total_vehicles_seen", 0) or 0),
            )

        # Collect alerts (non-update events)
        if event_type in ("parking_full", "parking_limited", "long_dwell"):
            alerts.append(ParkingAlert(
                eventType=event_type,
                severity=str(getattr(ev, "severity", "medium")).lower(),
                cameraId=camera_id,
                ts=ts,
                occupancyPct=float(getattr(ev, "meta_occupancy_pct", 0.0) or 0.0),
                vehicleId=getattr(ev, "meta_vehicle_id", None),
                dwellMinutes=float(getattr(ev, "meta_dwell_minutes", 0.0) or 0.0)
                    if event_type == "long_dwell" else None,
            ))

    return ParkingResponse(
        ts=datetime.now(timezone.utc),
        current=current,
        recentAlerts=alerts[-limit:],
        totalAlerts=len(alerts),
    )

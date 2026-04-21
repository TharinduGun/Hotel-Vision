"""
GET /api/v1/crowd/insights — crowd analytics data from latest session.

Reads the crowd CSVs exported by the CrowdDetectionModule and
returns footfall events, dwell records, and summary statistics.
"""

import json
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from backend.config import discover_latest_csv
from backend.models import (
    CrowdFootfallEvent,
    CrowdDwellRecord,
    CrowdInsightsResponse,
    CrowdSummary,
)

router = APIRouter(prefix="/crowd", tags=["Crowd Insights"])
logger = logging.getLogger(__name__)


def _find_session_dir() -> Path | None:
    """Find the session directory that contains crowd data."""
    csv_path = discover_latest_csv()
    if csv_path is None:
        return None
    # Session dir is the parent of the CSV
    return csv_path.parent


def _read_footfall(session_dir: Path, limit: int = 50) -> list[CrowdFootfallEvent]:
    """Read footfall events from crowd_footfall.csv."""
    ff_path = session_dir / "crowd_footfall.csv"
    if not ff_path.exists():
        return []

    events = []
    try:
        with open(ff_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(CrowdFootfallEvent(
                    trackId=int(float(row.get("track_id", 0))),
                    direction=row.get("direction", "entry"),
                    timestampSec=float(row.get("timestamp", 0)),
                    frameIdx=int(float(row.get("frame_idx", 0))),
                    edge=row.get("edge", "unknown"),
                    positionX=float(row.get("position_x", 0)),
                    positionY=float(row.get("position_y", 0)),
                ))
    except Exception as e:
        logger.warning("Error reading footfall CSV: %s", e)

    # Return most recent events
    return events[-limit:]


def _read_dwells(session_dir: Path, limit: int = 50) -> list[CrowdDwellRecord]:
    """Read dwell records from crowd_dwell.csv."""
    dwell_path = session_dir / "crowd_dwell.csv"
    if not dwell_path.exists():
        return []

    records = []
    try:
        with open(dwell_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(CrowdDwellRecord(
                    trackId=int(float(row.get("track_id", 0))),
                    entryTimeSec=float(row.get("entry_time", 0)),
                    exitTimeSec=float(row.get("exit_time", 0)),
                    durationSec=float(row.get("duration_sec", 0)),
                    entryX=float(row.get("entry_x", 0)),
                    entryY=float(row.get("entry_y", 0)),
                    exitX=float(row.get("exit_x", 0)),
                    exitY=float(row.get("exit_y", 0)),
                ))
    except Exception as e:
        logger.warning("Error reading dwell CSV: %s", e)

    return records[-limit:]


def _read_summary(session_dir: Path) -> CrowdSummary:
    """Read crowd summary from crowd_summary.json."""
    summary_path = session_dir / "crowd_summary.json"
    if not summary_path.exists():
        return CrowdSummary()

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CrowdSummary(
            totalEntries=data.get("total_entries", 0),
            totalExits=data.get("total_exits", 0),
            peakOccupancy=data.get("peak_occupancy", 0),
            avgOccupancy=data.get("avg_occupancy", 0.0),
            currentDensity=data.get("current_density", "low"),
            avgDwellSec=data.get("avg_dwell_sec", 0.0),
            maxDwellSec=data.get("max_dwell_sec", 0.0),
            totalUniquePersons=data.get("total_unique_tracks", 0),
        )
    except Exception as e:
        logger.warning("Error reading crowd summary: %s", e)
        return CrowdSummary()


@router.get("/insights", response_model=CrowdInsightsResponse)
async def crowd_insights():
    """
    Return crowd analytics data from the latest processing session.
    Includes footfall events, dwell records, and summary statistics.
    """
    session_dir = _find_session_dir()

    if session_dir is None:
        return CrowdInsightsResponse(
            ts=datetime.now(timezone.utc),
            summary=CrowdSummary(),
        )

    # Check for heatmap
    heatmap_path = session_dir / "crowd_heatmap.png"
    heatmap_url = f"/evidence/{session_dir.name}/crowd_heatmap.png" if heatmap_path.exists() else None

    return CrowdInsightsResponse(
        ts=datetime.now(timezone.utc),
        summary=_read_summary(session_dir),
        recentFootfall=_read_footfall(session_dir),
        recentDwells=_read_dwells(session_dir),
        heatmapUrl=heatmap_url,
    )

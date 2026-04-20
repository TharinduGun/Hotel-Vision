"""GET /api/v1/live/state — combined payload for easy frontend polling."""

from fastapi import APIRouter

from backend.models import LiveState
from backend.services.csv_adapter import CSVDataSource
from backend.services.aggregations import (
    build_alerts,
    build_employees,
    build_snapshots,
    build_summary,
)

router = APIRouter(prefix="/live", tags=["Live"])

_source = CSVDataSource()


@router.get("/state", response_model=LiveState)
async def live_state():
    """
    Return the entire dashboard state in one call.
    Frontend can poll this every 5 seconds if WebSocket isn't ready.
    """
    events = _source.get_events()
    return LiveState(
        summary=build_summary(events),
        snapshots=build_snapshots(),
        alerts=build_alerts(events),
        employees=build_employees(events),
    )

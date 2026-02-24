"""GET /api/v1/alerts — alert feed with filtering."""

from fastapi import APIRouter, Query

from backend.models import AlertsResponse
from backend.services.csv_adapter import CSVDataSource
from backend.services.aggregations import build_alerts

router = APIRouter(prefix="/alerts", tags=["Alerts"])

_source = CSVDataSource()


@router.get("", response_model=AlertsResponse)
async def list_alerts(
    status: str = Query("active", description="Filter by status: active | resolved | all"),
    limit: int = Query(20, ge=1, le=100, description="Max alerts to return"),
):
    """Return derived alerts from tracking events."""
    events = _source.get_events()
    return build_alerts(events, status=status, limit=limit)

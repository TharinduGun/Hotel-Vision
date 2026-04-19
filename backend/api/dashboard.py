"""GET /api/v1/dashboard/summary — KPI cards for the top of the dashboard."""

from fastapi import APIRouter

from backend.models import DashboardSummary
from backend.services.csv_adapter import CSVDataSource
from backend.services.aggregations import build_summary

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

_source = CSVDataSource()


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary():
    """Return all 6 KPI cards in a single call."""
    events = _source.get_events()
    return build_summary(events)

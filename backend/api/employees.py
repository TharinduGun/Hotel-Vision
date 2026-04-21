"""GET /api/v1/employees — employee tracking panel."""

from fastapi import APIRouter, Query

from backend.models import EmployeesResponse
from backend.services.csv_adapter import CSVDataSource
from backend.services.aggregations import build_employees

router = APIRouter(prefix="/employees", tags=["Employees"])

_source = CSVDataSource()


@router.get("", response_model=EmployeesResponse)
async def list_employees(
    status: str = Query(None, description="Filter: ON_DUTY | BREAK | OFF_DUTY"),
):
    """Return employee (staff) tracking data."""
    events = _source.get_events()
    return build_employees(events, status_filter=status)

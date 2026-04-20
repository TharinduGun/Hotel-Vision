"""
GET /api/v1/cameras          — camera list
GET /api/v1/cameras/snapshots — latest snapshot URLs per camera
"""

from fastapi import APIRouter

from backend.models import Camera, CameraSnapshot
from backend.services.aggregations import build_cameras, build_snapshots

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("", response_model=list[Camera])
async def list_cameras():
    """Return all configured cameras and their status."""
    return build_cameras()


@router.get("/snapshots", response_model=list[CameraSnapshot])
async def camera_snapshots():
    """Return the latest snapshot URL for each camera."""
    return build_snapshots()

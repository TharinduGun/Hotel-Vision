"""
GET /api/v1/cameras              — camera list
GET /api/v1/cameras/snapshots    — latest snapshot URLs per camera
GET /api/v1/cameras/{cam_id}/video — stream the latest output video for a camera
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.models import Camera, CameraSnapshot
from backend.services.aggregations import build_cameras, build_snapshots
from backend import config

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("", response_model=list[Camera])
async def list_cameras():
    """Return all configured cameras and their status."""
    return build_cameras()


@router.get("/snapshots", response_model=list[CameraSnapshot])
async def camera_snapshots():
    """Return the latest snapshot URL for each camera."""
    return build_snapshots()


def _find_latest_video(camera_id: str) -> Path | None:
    """Find the latest output video for a given camera ID."""
    log_dir = config.CSV_BASE_DIR
    if not log_dir.exists():
        return None

    # Search session directories newest-first
    sessions = sorted(log_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
    for session_dir in sessions:
        if not session_dir.is_dir():
            continue
        video_path = session_dir / f"output_{camera_id}.mp4"
        if video_path.exists():
            return video_path

    return None


@router.get("/{camera_id}/video")
async def stream_camera_video(camera_id: str, request: Request):
    """
    Stream the latest processed output video for a camera.
    Supports HTTP Range requests for seeking in the browser video player.
    """
    video_path = _find_latest_video(camera_id)
    if not video_path:
        raise HTTPException(status_code=404, detail=f"No output video found for {camera_id}")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range header: "bytes=start-end"
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1

        # Clamp end to file size
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_range():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 1024 * 1024  # 1MB chunks
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
            },
        )
    else:
        # Full file download
        def iter_file():
            with open(video_path, "rb") as f:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data

        return StreamingResponse(
            iter_file(),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Type": "video/mp4",
            },
        )

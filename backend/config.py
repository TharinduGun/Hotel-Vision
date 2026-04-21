"""
Centralized configuration for the analytics backend.

Reads from environment variables with sensible defaults.
Set DEV=true for development mode (permissive CORS).
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
# Root of the video-analytics project (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the ML pipeline writes session output
CSV_BASE_DIR = Path(
    os.getenv("CSV_BASE_DIR", str(PROJECT_ROOT / "output" / "logs"))
)

# Static media served at /media/...
MEDIA_DIR = Path(__file__).resolve().parent / "storage" / "media"


# ── Environment ────────────────────────────────────────────────────────
DEV_MODE: bool = os.getenv("DEV", "true").lower() in ("true", "1", "yes")


# ── CORS ───────────────────────────────────────────────────────────────
def _parse_cors_origins() -> list[str]:
    """
    Read CORS_ORIGINS env var (comma-separated).
    Falls back to ["*"] ONLY when DEV=true.
    """
    raw = os.getenv("CORS_ORIGINS", "")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if DEV_MODE:
        return ["*"]
    # Production with no explicit origins → locked down to localhost only
    return ["http://localhost:3000", "http://localhost:5173"]


CORS_ORIGINS: list[str] = _parse_cors_origins()


# ── Camera config (static for MVP) ────────────────────────────────────
CAMERAS = [
    {
        "id": "CAM-01",
        "name": "Main Lobby",
        "location": "Ground Floor - Entrance",
        "status": "online",
        "streamUrl": "rtsp://placeholder/cam01",
    },
    {
        "id": "CAM-02",
        "name": "Parking Lot A",
        "location": "Outdoor - North Side",
        "status": "online",
        "streamUrl": "rtsp://placeholder/cam02",
    },
    {
        "id": "CAM-03",
        "name": "Cashier Area",
        "location": "Ground Floor - POS",
        "status": "online",
        "streamUrl": "rtsp://placeholder/cam03",
    },
    {
        "id": "CAM-04",
        "name": "Back Office",
        "location": "First Floor - Staff",
        "status": "offline",
        "streamUrl": "rtsp://placeholder/cam04",
    },
]


# ── CSV discovery ─────────────────────────────────────────────────────
CSV_FILENAME = "analytics_events.csv"


def discover_latest_csv() -> Path | None:
    """
    Deterministic latest-session picker.
    Returns the newest directory under CSV_BASE_DIR that contains
    a tracking_summary.csv file, or None if nothing is found.
    """
    if not CSV_BASE_DIR.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for entry in CSV_BASE_DIR.iterdir():
        if not entry.is_dir():
            continue
        csv_path = entry / CSV_FILENAME
        if csv_path.is_file():
            # Use the directory modification time for sorting
            candidates.append((entry.stat().st_mtime, csv_path))

    if not candidates:
        return None

    # Sort descending by mtime → newest first
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ── Server ─────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

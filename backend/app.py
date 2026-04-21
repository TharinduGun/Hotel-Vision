"""
FastAPI entry point for the Analytics Dashboard backend.

Run with:
    cd d:\\Work\\jwinfotech\\Videoanalystics\\video-analytics
    python -m backend.app

Features:
  - Versioned API routes under /api/v1
  - Health endpoint at /health
  - Environment-based CORS
  - Static file serving for /media
  - WebSocket at /ws/live
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.models import HealthResponse

# Routers
from backend.api.dashboard import router as dashboard_router
from backend.api.cameras import router as cameras_router
from backend.api.alerts import router as alerts_router
from backend.api.employees import router as employees_router
from backend.api.live import router as live_router
from backend.api.crowd import router as crowd_router
from backend.api.ws import router as ws_router
from backend.api.staff import router as staff_router
from backend.api.parking import router as parking_router

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if config.DEV_MODE else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend")


# ── Lifespan ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    csv_path = config.discover_latest_csv()
    logger.info("=== Analytics Backend starting ===")
    logger.info("DEV mode: %s", config.DEV_MODE)
    logger.info("CORS origins: %s", config.CORS_ORIGINS)
    logger.info("CSV base dir: %s", config.CSV_BASE_DIR)
    logger.info("Latest CSV: %s", csv_path or "(none found)")
    logger.info("Media dir: %s", config.MEDIA_DIR)
    yield
    logger.info("=== Analytics Backend shutting down ===")


# ── App creation ───────────────────────────────────────────────────────
app = FastAPI(
    title="Video Analytics Dashboard API",
    description="Backend API for the hotel video analytics dashboard. "
                "Reads from tracking CSVs today, swappable for RTSP/DB later.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (snapshots, evidence images) ──────────────────────────
config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
(config.MEDIA_DIR / "snapshots").mkdir(exist_ok=True)
(config.MEDIA_DIR / "evidence").mkdir(exist_ok=True)

app.mount("/media", StaticFiles(directory=str(config.MEDIA_DIR)), name="media")
app.mount("/evidence", StaticFiles(directory=str(config.CSV_BASE_DIR)), name="evidence")

# ── Versioned API routes (/api/v1) ─────────────────────────────────────
app.include_router(dashboard_router, prefix="/api/v1")
app.include_router(cameras_router,   prefix="/api/v1")
app.include_router(alerts_router,    prefix="/api/v1")
app.include_router(employees_router, prefix="/api/v1")
app.include_router(live_router,      prefix="/api/v1")
app.include_router(crowd_router,     prefix="/api/v1")
app.include_router(staff_router,     prefix="/api/v1")
app.include_router(parking_router,   prefix="/api/v1")

# ── WebSocket (not versioned) ──────────────────────────────────────────
app.include_router(ws_router)


# ── Health endpoint (not versioned) ────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Deployment health check — is the server running and can it find data?"""
    csv_path = config.discover_latest_csv()
    return HealthResponse(
        status="ok",
        ts=datetime.now(timezone.utc),
        dataSource="csv",
        latestCsv=str(csv_path) if csv_path else None,
    )


# ── Run with: python -m backend.app ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEV_MODE,
    )

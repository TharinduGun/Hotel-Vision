"""
WS /ws/live — real-time alert push via WebSocket.

MVP strategy:
  - Poll the CSV file every 2 seconds
  - Track last_modified timestamp + last_row_index to avoid duplicates
  - Only push genuinely new events as alert_new messages
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.csv_adapter import CSVDataSource
from backend.services.aggregations import _derive_alerts

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])

POLL_INTERVAL_SEC = 2.0


@router.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Keeps a WebSocket open and pushes new alerts as they appear.
    Tracks last_row_count to deduplicate.
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    source = CSVDataSource()
    # Seed: mark everything currently in CSV as "already sent"
    _ = source.get_events()
    last_row_count = source.last_row_count

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL_SEC)

            # Check for new rows
            source.refresh_if_needed()
            new_events = source.get_new_events_since(last_row_count)

            if new_events:
                last_row_count = source.last_row_count
                alerts = _derive_alerts(new_events)

                for alert in alerts:
                    msg = {
                        "kind": "alert_new",
                        "payload": json.loads(alert.model_dump_json()),
                    }
                    await ws.send_json(msg)
                    logger.debug("Pushed alert %s", alert.id)

                # Also send a summary_update heartbeat
                await ws.send_json({
                    "kind": "summary_update",
                    "payload": {"newEvents": len(new_events)},
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await ws.close(code=1011, reason=str(e))

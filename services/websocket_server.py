import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

latest_data: dict = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    FIX H-03:
    1. Added asyncio.sleep(1.0) — original was a tight infinite loop with no
       sleep, sending thousands of messages/sec and maxing out CPU.
    2. Added try/except WebSocketDisconnect — original crashed the server
       whenever a client closed the browser tab.
    """

    await websocket.accept()

    try:
        while True:
            await websocket.send_text(json.dumps(latest_data))
            await asyncio.sleep(1.0)   # ← throttle to 1 update/sec

    except WebSocketDisconnect:
        pass   # clean exit, not an error


def update_data(data: dict):
    global latest_data
    latest_data = data
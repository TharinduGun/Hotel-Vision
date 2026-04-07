"""
WebSocket client for streaming events to the backend dashboard.

Stubbed out by default. To enable:
    Set enable_websocket=True in EventBuilder constructor.
"""

import json
import websocket


class WebSocketClient:

    def __init__(self, url="ws://localhost:8000/ws/events"):
        self.url = url
        self.ws = websocket.WebSocket()
        self.ws.connect(url)

    def send_event(self, event):
        """Send a structured event dict as JSON."""
        self.ws.send(json.dumps(event))

    def close(self):
        """Close the WebSocket connection."""
        self.ws.close()
"""WebSocket connection manager for pushing real-time data to the dashboard."""

import asyncio
import json
from typing import Any

from fastapi import WebSocket

from app.core.logging import get_logger

logger = get_logger(__name__)


class WebSocketHub:
    """Manages WebSocket connections and broadcasts messages to clients."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str = "default", accept: bool = True) -> None:
        if accept:
            await websocket.accept()
        if channel not in self._connections:
            self._connections[channel] = []
        self._connections[channel].append(websocket)
        logger.info("ws_client_connected", channel=channel, total=len(self._connections[channel]))

    def disconnect(self, websocket: WebSocket, channel: str = "default") -> None:
        if channel in self._connections:
            self._connections[channel] = [
                ws for ws in self._connections[channel] if ws != websocket
            ]
            logger.info("ws_client_disconnected", channel=channel)

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Send data to all connected clients on a channel."""
        if channel not in self._connections:
            return

        message = json.dumps(data)
        disconnected = []

        for ws in self._connections[channel]:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        # Cleanup disconnected clients
        for ws in disconnected:
            self.disconnect(ws, channel)

    @property
    def connection_count(self) -> int:
        return sum(len(clients) for clients in self._connections.values())


ws_hub = WebSocketHub()

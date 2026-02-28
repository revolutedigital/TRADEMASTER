"""Bridge between Redis Streams events and WebSocket hub.

Consumes events from Redis and broadcasts them to connected dashboard clients.
"""

import asyncio
import json

from app.api.websocket.hub import ws_hub
from app.core.events import EventType, event_bus
from app.core.logging import get_logger
from app.core.metrics import metrics

logger = get_logger(__name__)


class WebSocketBroadcaster:
    """Consumes events from Redis Streams and pushes to WebSocket clients."""

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="ws_broadcaster")
        logger.info("ws_broadcaster_started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ws_broadcaster_stopped")

    async def _loop(self) -> None:
        """Main loop: consume events and broadcast to WebSocket clients."""
        while self._running:
            try:
                events = await event_bus.subscribe(
                    event_types=[
                        EventType.KLINE_UPDATE,
                        EventType.TRADE_UPDATE,
                        EventType.SIGNAL_GENERATED,
                        EventType.ORDER_FILLED,
                        EventType.POSITION_OPENED,
                        EventType.POSITION_CLOSED,
                        EventType.RISK_ALERT,
                        EventType.CIRCUIT_BREAKER_TRIGGERED,
                    ],
                    group="ws_broadcaster",
                    consumer="broadcaster_1",
                    count=50,
                    block_ms=2000,
                )

                for event in events:
                    await self._dispatch(event.type, event.data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ws_broadcaster_error", error=str(e))
                await asyncio.sleep(2)

    async def _dispatch(self, event_type: EventType, data: dict) -> None:
        """Route event to the correct WebSocket channel."""
        metrics.ws_connections.set(ws_hub.connection_count)

        if event_type == EventType.KLINE_UPDATE:
            symbol = data.get("symbol", "")
            await ws_hub.broadcast(f"market:{symbol}", {
                "type": "kline",
                "data": data,
            })

        elif event_type == EventType.TRADE_UPDATE:
            symbol = data.get("symbol", "")
            await ws_hub.broadcast(f"market:{symbol}", {
                "type": "trade",
                "data": data,
            })

        elif event_type == EventType.SIGNAL_GENERATED:
            await ws_hub.broadcast("signals", {
                "type": "signal",
                "data": data,
            })

        elif event_type in (EventType.ORDER_FILLED, EventType.POSITION_OPENED, EventType.POSITION_CLOSED):
            await ws_hub.broadcast("portfolio", {
                "type": event_type.value,
                "data": data,
            })

        elif event_type in (EventType.RISK_ALERT, EventType.CIRCUIT_BREAKER_TRIGGERED):
            await ws_hub.broadcast("portfolio", {
                "type": "risk_alert",
                "data": data,
            })


ws_broadcaster = WebSocketBroadcaster()

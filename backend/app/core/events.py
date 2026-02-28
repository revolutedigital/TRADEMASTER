"""Redis Streams event bus for inter-service communication."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import redis.asyncio as redis

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EventType(StrEnum):
    # Market data events
    KLINE_UPDATE = "kline.update"
    TRADE_UPDATE = "trade.update"
    ORDERBOOK_UPDATE = "orderbook.update"

    # Trading events
    SIGNAL_GENERATED = "signal.generated"
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"

    # Risk events
    RISK_ALERT = "risk.alert"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"

    # System events
    SYSTEM_HEALTH = "system.health"
    MODEL_RETRAINED = "ml.model_retrained"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "trademaster"


class EventBus:
    """Redis Streams-based event bus for publishing and consuming events."""

    def __init__(self) -> None:
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("event_bus_connected")

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("event_bus_disconnected")

    async def publish(self, event: Event) -> str | None:
        """Publish an event to a Redis Stream. Returns the message ID."""
        if not self._redis:
            logger.warning("event_bus_not_connected", event_type=event.type)
            return None

        stream_key = f"stream:{event.type}"
        message = {
            "type": event.type,
            "data": json.dumps(event.data),
            "timestamp": event.timestamp,
            "source": event.source,
        }

        msg_id = await self._redis.xadd(stream_key, message, maxlen=10000)
        logger.debug("event_published", event_type=event.type, msg_id=msg_id)
        return msg_id

    async def subscribe(
        self,
        event_types: list[EventType],
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> list[Event]:
        """Consume events from Redis Streams using consumer groups."""
        if not self._redis:
            return []

        streams = {}
        for event_type in event_types:
            stream_key = f"stream:{event_type}"
            # Create consumer group if it doesn't exist
            try:
                await self._redis.xgroup_create(stream_key, group, id="0", mkstream=True)
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

            streams[stream_key] = ">"

        results = await self._redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams=streams,
            count=count,
            block=block_ms,
        )

        events = []
        for _stream_key, messages in results:
            for msg_id, msg_data in messages:
                try:
                    event = Event(
                        type=EventType(msg_data["type"]),
                        data=json.loads(msg_data["data"]),
                        timestamp=msg_data["timestamp"],
                        source=msg_data.get("source", "unknown"),
                    )
                    events.append(event)
                    # Acknowledge the message
                    await self._redis.xack(
                        _stream_key, group, msg_id
                    )
                except (KeyError, ValueError) as e:
                    logger.error("event_parse_error", msg_id=msg_id, error=str(e))

        return events


# Global event bus instance
event_bus = EventBus()

"""Event sourcing: store all state changes as immutable events.

Provides complete audit trail and ability to rebuild state from events.
Every state change in the system is captured as an event.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    SIGNAL_GENERATED = "signal.generated"
    CONFIG_CHANGED = "config.changed"
    CIRCUIT_BREAKER_ACTIVATED = "circuit_breaker.activated"
    RISK_THRESHOLD_BREACHED = "risk.threshold_breached"
    MODEL_RETRAINED = "model.retrained"
    ALERT_TRIGGERED = "alert.triggered"


@dataclass
class Event:
    event_id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: str
    data: dict
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }


class EventStore:
    """In-memory event store (production: use EventStoreDB or PostgreSQL)."""

    def __init__(self):
        self._events: list[Event] = []
        self._handlers: dict[EventType, list] = {}
        self._snapshots: dict[str, dict] = {}

    async def append(self, event: Event) -> None:
        """Append an event to the store."""
        self._events.append(event)
        logger.debug("event_appended", type=event.event_type.value, aggregate=event.aggregate_id)

        # Dispatch to handlers
        for handler in self._handlers.get(event.event_type, []):
            try:
                await handler(event)
            except Exception as e:
                logger.warning("event_handler_failed", type=event.event_type.value, error=str(e))

    def subscribe(self, event_type: EventType, handler) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def get_events(self, aggregate_id: str | None = None, event_type: EventType | None = None, since: datetime | None = None, limit: int = 100) -> list[Event]:
        """Query events with optional filters."""
        events = self._events
        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events[-limit:]

    async def rebuild_state(self, aggregate_id: str) -> dict:
        """Rebuild current state from events."""
        events = await self.get_events(aggregate_id=aggregate_id, limit=10000)
        state: dict[str, Any] = {}
        for event in events:
            state.update(event.data)
            state["_last_event"] = event.event_type.value
            state["_version"] = event.version
        return state

    async def create_snapshot(self, aggregate_id: str) -> dict:
        """Create a snapshot of current state for faster rebuilds."""
        state = await self.rebuild_state(aggregate_id)
        self._snapshots[aggregate_id] = state
        logger.info("snapshot_created", aggregate=aggregate_id)
        return state

    def get_stats(self) -> dict:
        return {
            "total_events": len(self._events),
            "event_types": list(set(e.event_type.value for e in self._events)),
            "handlers_registered": sum(len(h) for h in self._handlers.values()),
            "snapshots": len(self._snapshots),
        }


# Helper to create events
def create_event(event_type: EventType, aggregate_id: str, aggregate_type: str, data: dict, **metadata) -> Event:
    return Event(
        event_id=str(uuid4()),
        event_type=event_type,
        aggregate_id=aggregate_id,
        aggregate_type=aggregate_type,
        data=data,
        metadata=metadata,
    )


event_store = EventStore()

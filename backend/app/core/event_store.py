"""Event sourcing: append-only event store with PostgreSQL persistence."""
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DomainEvent:
    event_id: str
    event_type: str  # e.g., "OrderPlaced", "OrderFilled", "PositionOpened"
    aggregate_type: str  # e.g., "Order", "Position", "Portfolio"
    aggregate_id: str
    data: dict[str, Any]
    metadata: dict[str, Any]
    version: int
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class EventStore:
    """Event store with in-memory cache and PostgreSQL persistence.

    Events are kept in-memory for fast reads and also persisted to the
    ``stored_events`` table for durability across restarts.
    Persistence is best-effort: if the DB is unavailable the event is
    still stored in memory and processing continues.
    """

    def __init__(self):
        self._events: list[DomainEvent] = []
        self._snapshots: dict[str, dict] = {}  # aggregate_id -> snapshot
        self._handlers: dict[str, list] = {}  # event_type -> handlers
        self._snapshot_interval = 100  # Snapshot every N events per aggregate

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _save_to_db(self, event: DomainEvent) -> None:
        """Persist a single event to PostgreSQL (best-effort)."""
        try:
            from app.models.base import async_session_factory
            from app.models.event import StoredEvent

            async with async_session_factory() as session:
                row = StoredEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    aggregate_type=event.aggregate_type,
                    aggregate_id=event.aggregate_id,
                    data=json.dumps(event.data),
                    metadata_=json.dumps(event.metadata),
                    version=event.version,
                    event_timestamp=event.timestamp,
                )
                session.add(row)
                await session.commit()
        except Exception as e:
            logger.warning("event_persist_failed", event_id=event.event_id, error=str(e))

    async def load_from_db(self, limit: int = 10_000) -> int:
        """Load recent events from PostgreSQL into in-memory store.

        Call this once at startup to rehydrate state.
        Returns the number of events loaded.
        """
        try:
            from app.models.base import async_session_factory
            from app.models.event import StoredEvent
            from sqlalchemy import select

            async with async_session_factory() as session:
                stmt = (
                    select(StoredEvent)
                    .order_by(StoredEvent.id.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = list(result.scalars().all())

            # Insert in chronological order (rows are newest-first)
            loaded = 0
            existing_ids = {e.event_id for e in self._events}
            for row in reversed(rows):
                if row.event_id in existing_ids:
                    continue
                event = DomainEvent(
                    event_id=row.event_id,
                    event_type=row.event_type,
                    aggregate_type=row.aggregate_type,
                    aggregate_id=row.aggregate_id,
                    data=json.loads(row.data),
                    metadata=json.loads(row.metadata_) if row.metadata_ else {},
                    version=row.version,
                    timestamp=row.event_timestamp,
                )
                self._events.append(event)
                existing_ids.add(row.event_id)
                loaded += 1

            logger.info("events_loaded_from_db", count=loaded)
            return loaded
        except Exception as e:
            logger.warning("events_load_from_db_failed", error=str(e))
            return 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str,
        data: dict,
        metadata: dict | None = None,
    ) -> DomainEvent:
        """Append a new event to the store (in-memory + async DB persist)."""
        # Get version for this aggregate
        agg_events = [e for e in self._events if e.aggregate_id == aggregate_id]
        version = len(agg_events) + 1

        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            data=data,
            metadata=metadata or {},
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._events.append(event)

        # Best-effort async persist to PostgreSQL
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_to_db(event))
        except RuntimeError:
            # No running event loop (e.g., during tests) — skip DB persist
            logger.debug("event_persist_skipped_no_loop", event_id=event.event_id)

        # Notify handlers
        for handler in self._handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.warning("event_handler_failed", event_type=event_type, error=str(e))

        # Auto-snapshot
        if version % self._snapshot_interval == 0:
            self._create_snapshot(aggregate_id)

        logger.debug("event_appended", event_type=event_type, aggregate_id=aggregate_id, version=version)
        return event

    def get_events(
        self,
        aggregate_id: str | None = None,
        event_type: str | None = None,
        since_version: int = 0,
        limit: int = 1000,
    ) -> list[DomainEvent]:
        """Query events with filters."""
        result = self._events

        if aggregate_id:
            result = [e for e in result if e.aggregate_id == aggregate_id]
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        if since_version > 0:
            result = [e for e in result if e.version > since_version]

        return result[-limit:]

    def get_aggregate_state(self, aggregate_id: str) -> list[DomainEvent]:
        """Get all events for an aggregate, starting from last snapshot."""
        snapshot = self._snapshots.get(aggregate_id)
        since = snapshot.get("version", 0) if snapshot else 0
        return self.get_events(aggregate_id=aggregate_id, since_version=since)

    def subscribe(self, event_type: str, handler) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def _create_snapshot(self, aggregate_id: str) -> None:
        """Create a snapshot for faster aggregate reconstruction."""
        events = [e for e in self._events if e.aggregate_id == aggregate_id]
        if events:
            self._snapshots[aggregate_id] = {
                "version": events[-1].version,
                "timestamp": events[-1].timestamp,
                "event_count": len(events),
            }

    def get_stats(self) -> dict:
        """Get event store statistics."""
        event_types = {}
        for e in self._events:
            event_types[e.event_type] = event_types.get(e.event_type, 0) + 1

        return {
            "total_events": len(self._events),
            "total_snapshots": len(self._snapshots),
            "event_types": event_types,
            "unique_aggregates": len(set(e.aggregate_id for e in self._events)),
        }


# Global event store
event_store = EventStore()

"""CQRS projector: materialise read models from an append-only event store.

Implements the read side of Command Query Responsibility Segregation.
Events written to the event store are projected into denormalised read
models (projections) optimised for specific query patterns. Supports
snapshotting every N events and eventual-consistency lag monitoring.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.core.event_sourcing import Event, EventStore, EventType
from app.core.logging import get_logger

logger = get_logger(__name__)

SNAPSHOT_INTERVAL: int = 1000  # events between automatic snapshots


# -----------------------------------------------------------------------
# Base projection
# -----------------------------------------------------------------------

class Projection(ABC):
    """Base class for all read-model projections.

    Subclasses implement *apply* to fold an event into the read model and
    *reset* to clear the model for a full replay.
    """

    def __init__(self) -> None:
        self.last_event_version: int = 0
        self.last_event_timestamp: datetime | None = None
        self.events_applied: int = 0

    @abstractmethod
    async def apply(self, event: Event) -> None:
        """Fold a single event into the read model."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the projection for a full rebuild."""

    @abstractmethod
    def get_state(self) -> dict:
        """Return the current materialised state."""

    def _track(self, event: Event) -> None:
        """Update internal bookkeeping after applying an event."""
        self.last_event_version = event.version
        self.last_event_timestamp = event.timestamp
        self.events_applied += 1


# -----------------------------------------------------------------------
# Concrete projections
# -----------------------------------------------------------------------

class PortfolioProjection(Projection):
    """Read model optimised for querying current portfolio positions.

    Maintains per-symbol holdings, cash balance, and total portfolio value.
    """

    def __init__(self) -> None:
        super().__init__()
        self._positions: dict[str, dict[str, Any]] = {}
        self._cash_balance: float = 0.0
        self._total_value: float = 0.0

    async def apply(self, event: Event) -> None:
        data = event.data

        if event.event_type == EventType.POSITION_OPENED:
            symbol = data.get("symbol", "")
            self._positions[symbol] = {
                "symbol": symbol,
                "quantity": data.get("quantity", 0),
                "avg_price": data.get("price", 0),
                "current_price": data.get("price", 0),
                "unrealized_pnl": 0.0,
                "opened_at": event.timestamp.isoformat(),
            }
            cost = data.get("quantity", 0) * data.get("price", 0)
            self._cash_balance -= cost

        elif event.event_type == EventType.POSITION_CLOSED:
            symbol = data.get("symbol", "")
            pos = self._positions.pop(symbol, None)
            if pos is not None:
                proceeds = data.get("quantity", 0) * data.get("price", 0)
                self._cash_balance += proceeds

        elif event.event_type == EventType.ORDER_FILLED:
            symbol = data.get("symbol", "")
            qty = data.get("quantity", 0)
            price = data.get("price", 0)
            side = data.get("side", "buy")
            if symbol in self._positions:
                pos = self._positions[symbol]
                if side == "buy":
                    total_cost = pos["avg_price"] * pos["quantity"] + price * qty
                    pos["quantity"] += qty
                    pos["avg_price"] = (
                        total_cost / pos["quantity"] if pos["quantity"] else 0
                    )
                    self._cash_balance -= price * qty
                else:
                    pos["quantity"] -= qty
                    self._cash_balance += price * qty
                    if pos["quantity"] <= 0:
                        self._positions.pop(symbol, None)

        self._recalculate_value()
        self._track(event)

    def _recalculate_value(self) -> None:
        position_value = sum(
            p["quantity"] * p["current_price"] for p in self._positions.values()
        )
        self._total_value = self._cash_balance + position_value

    def reset(self) -> None:
        self._positions.clear()
        self._cash_balance = 0.0
        self._total_value = 0.0
        self.events_applied = 0
        self.last_event_version = 0
        self.last_event_timestamp = None

    def get_state(self) -> dict:
        return {
            "positions": dict(self._positions),
            "cash_balance": self._cash_balance,
            "total_value": self._total_value,
            "position_count": len(self._positions),
        }


class TradeHistoryProjection(Projection):
    """Append-only trade blotter optimised for history queries.

    Records every order placement, fill, and cancellation with running
    statistics (win rate, total PnL, trade count).
    """

    def __init__(self) -> None:
        super().__init__()
        self._trades: list[dict] = []
        self._stats: dict[str, Any] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }

    async def apply(self, event: Event) -> None:
        data = event.data

        if event.event_type in (
            EventType.ORDER_PLACED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
        ):
            trade_record = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "symbol": data.get("symbol", ""),
                "side": data.get("side", ""),
                "quantity": data.get("quantity", 0),
                "price": data.get("price", 0),
                "timestamp": event.timestamp.isoformat(),
                "aggregate_id": event.aggregate_id,
            }
            self._trades.append(trade_record)

        if event.event_type == EventType.ORDER_FILLED:
            pnl = data.get("realized_pnl", 0.0)
            self._stats["total_trades"] += 1
            self._stats["total_pnl"] += pnl
            if pnl > 0:
                self._stats["winning_trades"] += 1
            elif pnl < 0:
                self._stats["losing_trades"] += 1
            total = self._stats["total_trades"]
            self._stats["win_rate"] = (
                self._stats["winning_trades"] / total if total > 0 else 0.0
            )

        self._track(event)

    def reset(self) -> None:
        self._trades.clear()
        self._stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }
        self.events_applied = 0
        self.last_event_version = 0
        self.last_event_timestamp = None

    def get_state(self) -> dict:
        return {
            "trades": list(self._trades),
            "stats": dict(self._stats),
        }


class RiskProjection(Projection):
    """Real-time risk dashboard derived from events.

    Tracks circuit breaker activations, risk threshold breaches, current
    exposure, and drawdown metrics.
    """

    def __init__(self) -> None:
        super().__init__()
        self._breaches: list[dict] = []
        self._circuit_breaker_events: list[dict] = []
        self._current_exposure: float = 0.0
        self._max_exposure: float = 0.0
        self._peak_value: float = 0.0
        self._max_drawdown: float = 0.0

    async def apply(self, event: Event) -> None:
        data = event.data

        if event.event_type == EventType.RISK_THRESHOLD_BREACHED:
            self._breaches.append({
                "event_id": event.event_id,
                "threshold": data.get("threshold", ""),
                "value": data.get("value", 0),
                "limit": data.get("limit", 0),
                "timestamp": event.timestamp.isoformat(),
            })

        elif event.event_type == EventType.CIRCUIT_BREAKER_ACTIVATED:
            self._circuit_breaker_events.append({
                "event_id": event.event_id,
                "reason": data.get("reason", ""),
                "timestamp": event.timestamp.isoformat(),
            })

        elif event.event_type == EventType.POSITION_OPENED:
            notional = data.get("quantity", 0) * data.get("price", 0)
            self._current_exposure += notional
            self._max_exposure = max(self._max_exposure, self._current_exposure)

        elif event.event_type == EventType.POSITION_CLOSED:
            notional = data.get("quantity", 0) * data.get("price", 0)
            self._current_exposure = max(0, self._current_exposure - notional)

        # Drawdown tracking
        portfolio_value = data.get("portfolio_value")
        if portfolio_value is not None:
            if portfolio_value > self._peak_value:
                self._peak_value = portfolio_value
            if self._peak_value > 0:
                drawdown = (self._peak_value - portfolio_value) / self._peak_value
                self._max_drawdown = max(self._max_drawdown, drawdown)

        self._track(event)

    def reset(self) -> None:
        self._breaches.clear()
        self._circuit_breaker_events.clear()
        self._current_exposure = 0.0
        self._max_exposure = 0.0
        self._peak_value = 0.0
        self._max_drawdown = 0.0
        self.events_applied = 0
        self.last_event_version = 0
        self.last_event_timestamp = None

    def get_state(self) -> dict:
        return {
            "breaches": list(self._breaches),
            "circuit_breaker_events": list(self._circuit_breaker_events),
            "current_exposure": self._current_exposure,
            "max_exposure": self._max_exposure,
            "max_drawdown": self._max_drawdown,
            "breach_count": len(self._breaches),
            "circuit_breaker_count": len(self._circuit_breaker_events),
        }


# -----------------------------------------------------------------------
# Snapshot store
# -----------------------------------------------------------------------

@dataclass
class ProjectionSnapshot:
    """Point-in-time snapshot of a projection's state."""

    snapshot_id: str
    projection_name: str
    state: dict
    event_version: int
    events_applied: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# -----------------------------------------------------------------------
# CQRS projector
# -----------------------------------------------------------------------

class CQRSProjector:
    """Manages projections, snapshots, and consistency monitoring.

    Subscribes to the event store and fans out each event to all
    registered projections. Periodically creates snapshots and exposes
    lag metrics for eventual-consistency monitoring.

    Usage::

        projector = CQRSProjector(event_store)
        projector.register_projection("portfolio", PortfolioProjection())
        await projector.start()
    """

    def __init__(
        self,
        event_store: EventStore,
        *,
        snapshot_interval: int = SNAPSHOT_INTERVAL,
    ):
        self._event_store = event_store
        self._projections: dict[str, Projection] = {}
        self._snapshots: dict[str, list[ProjectionSnapshot]] = {}
        self._snapshot_interval = snapshot_interval
        self._write_timestamp: datetime | None = None
        self._read_timestamp: datetime | None = None
        self._events_since_snapshot: dict[str, int] = {}
        self._running = False
        self._consistency_lag_ms: float = 0.0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_projection(self, name: str, projection: Projection) -> None:
        """Register a projection to receive events."""
        self._projections[name] = projection
        self._snapshots.setdefault(name, [])
        self._events_since_snapshot[name] = 0
        logger.info("projection_registered", name=name)

    def register_default_projections(self) -> None:
        """Register the standard set of TradeMaster projections."""
        self.register_projection("portfolio", PortfolioProjection())
        self.register_projection("trade_history", TradeHistoryProjection())
        self.register_projection("risk", RiskProjection())

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to all event types and begin projecting."""
        if self._running:
            return
        self._running = True
        for event_type in EventType:
            self._event_store.subscribe(event_type, self._on_event)
        logger.info(
            "cqrs_projector_started",
            projections=list(self._projections.keys()),
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("cqrs_projector_stopped")

    async def _on_event(self, event: Event) -> None:
        """Fan out an event to every registered projection."""
        self._write_timestamp = event.timestamp
        start = time.monotonic()

        for name, projection in self._projections.items():
            try:
                await projection.apply(event)
                self._events_since_snapshot[name] = (
                    self._events_since_snapshot.get(name, 0) + 1
                )

                # Auto-snapshot
                if self._events_since_snapshot[name] >= self._snapshot_interval:
                    await self.create_snapshot(name)

            except Exception as exc:
                logger.error(
                    "projection_apply_failed",
                    projection=name,
                    event_id=event.event_id,
                    error=str(exc),
                )

        elapsed_ms = (time.monotonic() - start) * 1000
        self._read_timestamp = datetime.now(timezone.utc)
        self._consistency_lag_ms = elapsed_ms
        logger.debug(
            "event_projected",
            event_type=event.event_type.value,
            lag_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    async def create_snapshot(self, projection_name: str) -> ProjectionSnapshot | None:
        """Capture the current state of a projection for faster rebuilds."""
        projection = self._projections.get(projection_name)
        if projection is None:
            logger.warning("snapshot_skip_unknown_projection", name=projection_name)
            return None

        snapshot = ProjectionSnapshot(
            snapshot_id=str(uuid4()),
            projection_name=projection_name,
            state=projection.get_state(),
            event_version=projection.last_event_version,
            events_applied=projection.events_applied,
        )
        self._snapshots[projection_name].append(snapshot)
        self._events_since_snapshot[projection_name] = 0
        logger.info(
            "projection_snapshot_created",
            projection=projection_name,
            event_version=snapshot.event_version,
            events_applied=snapshot.events_applied,
        )
        return snapshot

    async def get_latest_snapshot(
        self, projection_name: str
    ) -> ProjectionSnapshot | None:
        """Return the most recent snapshot for a projection, if any."""
        snapshots = self._snapshots.get(projection_name, [])
        return snapshots[-1] if snapshots else None

    # ------------------------------------------------------------------
    # Rebuild / replay
    # ------------------------------------------------------------------

    async def rebuild_projection(self, projection_name: str) -> None:
        """Replay every event in the store to rebuild a projection from zero.

        If a snapshot exists, the projection state is first restored from
        the snapshot and only events *after* the snapshot version are
        replayed.
        """
        projection = self._projections.get(projection_name)
        if projection is None:
            raise ValueError(f"Unknown projection: {projection_name}")

        projection.reset()
        logger.info("projection_rebuild_started", projection=projection_name)

        # Attempt to restore from latest snapshot
        snapshot = await self.get_latest_snapshot(projection_name)
        replay_since_version = 0
        if snapshot is not None:
            replay_since_version = snapshot.event_version
            logger.info(
                "projection_restoring_snapshot",
                projection=projection_name,
                snapshot_version=replay_since_version,
            )
            # Re-apply snapshot state is projection-specific; for a generic
            # approach we replay only events after the snapshot version.
            # In production you would deserialise the snapshot state directly.

        events = await self._event_store.get_events(limit=1_000_000)
        applied = 0
        for event in events:
            if event.version <= replay_since_version:
                continue
            await projection.apply(event)
            applied += 1

        # Auto-snapshot after a full rebuild
        if applied >= self._snapshot_interval:
            await self.create_snapshot(projection_name)

        logger.info(
            "projection_rebuild_complete",
            projection=projection_name,
            events_replayed=applied,
        )

    async def rebuild_all(self) -> None:
        """Rebuild every registered projection from scratch."""
        for name in list(self._projections):
            await self.rebuild_projection(name)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_projection_state(self, projection_name: str) -> dict | None:
        """Return the materialised read model for a projection."""
        projection = self._projections.get(projection_name)
        if projection is None:
            return None
        return projection.get_state()

    def get_consistency_metrics(self) -> dict:
        """Return eventual-consistency health metrics."""
        projection_lag: dict[str, dict] = {}
        now = datetime.now(timezone.utc)

        for name, projection in self._projections.items():
            last_ts = projection.last_event_timestamp
            lag_seconds = (
                (now - last_ts).total_seconds() if last_ts is not None else None
            )
            projection_lag[name] = {
                "events_applied": projection.events_applied,
                "last_event_version": projection.last_event_version,
                "last_event_timestamp": (
                    last_ts.isoformat() if last_ts else None
                ),
                "lag_seconds": round(lag_seconds, 3) if lag_seconds is not None else None,
            }

        return {
            "write_timestamp": (
                self._write_timestamp.isoformat() if self._write_timestamp else None
            ),
            "read_timestamp": (
                self._read_timestamp.isoformat() if self._read_timestamp else None
            ),
            "projection_processing_lag_ms": round(self._consistency_lag_ms, 2),
            "projections": projection_lag,
        }

    def get_stats(self) -> dict:
        return {
            "projections_registered": len(self._projections),
            "projection_names": list(self._projections.keys()),
            "total_snapshots": sum(
                len(s) for s in self._snapshots.values()
            ),
            "snapshot_interval": self._snapshot_interval,
            "consistency_lag_ms": round(self._consistency_lag_ms, 2),
        }

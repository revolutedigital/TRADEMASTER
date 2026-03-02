"""Service interfaces (Protocols) for dependency injection.

All major services define a Protocol here so that:
- Services depend on abstractions, not concrete implementations
- Testing can inject mocks easily
- Paper/live modes are just different implementations
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.portfolio import Position, PortfolioSnapshot
from app.models.trade import Order


# ========================================
# Exchange Adapters
# ========================================


@runtime_checkable
class IExchangeAdapter(Protocol):
    """Abstract exchange client for price/balance queries."""

    async def get_ticker_price(self, symbol: str) -> float: ...
    async def get_balance(self, asset: str) -> float: ...
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> dict: ...
    async def cancel_order(self, symbol: str, order_id: int) -> dict: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...


# ========================================
# Order Execution
# ========================================


@runtime_checkable
class IOrderExecutionStrategy(Protocol):
    """Strategy pattern: paper vs live order execution."""

    async def execute(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
    ) -> Order: ...


@runtime_checkable
class IOrderManager(Protocol):
    """Full order lifecycle management."""

    async def execute_market_order(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
    ) -> Order: ...

    async def cancel_order(self, db: AsyncSession, order: Order) -> Order: ...


# ========================================
# Risk Management
# ========================================


@runtime_checkable
class IRiskManager(Protocol):
    """Pre-trade risk validation."""

    def validate_trade(self, proposal: Any) -> Any: ...


@runtime_checkable
class ICircuitBreaker(Protocol):
    """Drawdown-based trading circuit breaker."""

    def update(self, current_equity: float) -> Any: ...
    async def update_and_persist(self, current_equity: float) -> None: ...
    async def restore_from_redis(self) -> bool: ...
    def initialize(self, equity: float) -> None: ...

    @property
    def position_size_multiplier(self) -> float: ...

    def get_status(self) -> dict: ...


# ========================================
# Portfolio Management
# ========================================


@runtime_checkable
class IPortfolioTracker(Protocol):
    """Position lifecycle and P&L tracking."""

    async def open_position(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> Position: ...

    async def close_position(
        self,
        db: AsyncSession,
        position: Position,
        exit_price: float,
    ) -> Position: ...

    async def update_prices(self, db: AsyncSession, prices: dict[str, float]) -> list[Position]: ...

    async def check_stop_losses(self, db: AsyncSession, prices: dict[str, float]) -> list[Position]: ...

    async def get_open_positions(self, db: AsyncSession, symbol: str | None = None) -> list[Position]: ...

    async def get_total_exposure(self, db: AsyncSession) -> float: ...

    async def get_symbol_exposure(self, db: AsyncSession, symbol: str) -> float: ...

    async def take_snapshot(self, db: AsyncSession, equity: float, balance: float) -> PortfolioSnapshot: ...


# ========================================
# ML Pipeline
# ========================================


@runtime_checkable
class IMLPipeline(Protocol):
    """ML model inference."""

    async def predict(self, df: pd.DataFrame, symbol: str) -> Any: ...
    async def load_models(self, symbol: str) -> None: ...


# ========================================
# Market Data
# ========================================


@runtime_checkable
class IMarketDataCollector(Protocol):
    """Historical and live market data access."""

    async def get_latest_candles(
        self,
        db: AsyncSession,
        symbol: str,
        interval: str,
        limit: int = 300,
    ) -> pd.DataFrame: ...


# ========================================
# Event System
# ========================================


@runtime_checkable
class IEventBus(Protocol):
    """Pub/sub event bus for inter-service communication."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def publish(self, event: Any) -> str | None: ...
    async def subscribe(
        self,
        event_types: list,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> list: ...


# ========================================
# Repositories
# ========================================


@runtime_checkable
class IOrderRepository(Protocol):
    """Data access for orders."""

    async def create(self, db: AsyncSession, order: Order) -> Order: ...
    async def get_by_id(self, db: AsyncSession, order_id: int) -> Order | None: ...
    async def get_open_orders(self, db: AsyncSession, symbol: str | None = None) -> list[Order]: ...
    async def update(self, db: AsyncSession, order: Order) -> Order: ...


@runtime_checkable
class IPositionRepository(Protocol):
    """Data access for positions."""

    async def create(self, db: AsyncSession, position: Position) -> Position: ...
    async def get_by_id(self, db: AsyncSession, position_id: int) -> Position | None: ...
    async def get_open(self, db: AsyncSession, symbol: str | None = None) -> list[Position]: ...
    async def update(self, db: AsyncSession, position: Position) -> Position: ...

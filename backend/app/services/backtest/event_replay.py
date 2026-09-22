"""Causal event-level order execution for microstructure research."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from app.schemas.microstructure import MarketEventType, MicrostructureEvent


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class ReplayOrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class LiquidityRole(StrEnum):
    MAKER = "MAKER"
    TAKER = "TAKER"


@dataclass(frozen=True)
class ReplayOrder:
    order_id: str
    side: OrderSide
    quantity: float
    submitted_at: datetime
    order_type: ReplayOrderType = ReplayOrderType.MARKET
    limit_price: float | None = None
    latency_ms: int = 100
    maker_queue_ahead_quantity: float = 0.0

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id is required")
        if not math.isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("quantity must be finite and positive")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.maker_queue_ahead_quantity < 0:
            raise ValueError("maker queue position cannot be negative")
        if self.order_type == ReplayOrderType.LIMIT:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("limit orders require a positive limit_price")

    @property
    def active_at(self) -> datetime:
        return self.submitted_at + timedelta(milliseconds=self.latency_ms)


@dataclass(frozen=True)
class ReplayFill:
    order_id: str
    event_time: datetime
    side: OrderSide
    quantity: float
    price: float
    fee_bps: float
    liquidity_role: LiquidityRole

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def fee(self) -> float:
        return self.notional * self.fee_bps / 10_000


@dataclass
class ReplayOrderState:
    order: ReplayOrder
    remaining_quantity: float = field(init=False)
    queue_ahead_quantity: float = field(init=False)
    fills: list[ReplayFill] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remaining_quantity = self.order.quantity
        self.queue_ahead_quantity = self.order.maker_queue_ahead_quantity

    @property
    def filled_quantity(self) -> float:
        return self.order.quantity - self.remaining_quantity

    @property
    def complete(self) -> bool:
        return self.remaining_quantity <= 1e-12

    @property
    def average_fill_price(self) -> float | None:
        if not self.fills:
            return None
        return sum(fill.price * fill.quantity for fill in self.fills) / sum(
            fill.quantity for fill in self.fills
        )


@dataclass(frozen=True)
class ReplayExecutionConfig:
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    market_slippage_bps: float = 1.0

    def __post_init__(self) -> None:
        values = (self.taker_fee_bps, self.maker_fee_bps, self.market_slippage_bps)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("fees and slippage must be finite and non-negative")


class EventExecutionReplay:
    """Execute orders from an event stream without looking beyond each event."""

    def __init__(self, config: ReplayExecutionConfig | None = None) -> None:
        self._config = config or ReplayExecutionConfig()
        self._orders: dict[str, ReplayOrderState] = {}
        self._last_event_time: datetime | None = None

    def submit(self, order: ReplayOrder) -> ReplayOrderState:
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order_id: {order.order_id}")
        if self._last_event_time is not None and order.submitted_at < self._last_event_time:
            raise ValueError("cannot submit an order in the replay past")
        state = ReplayOrderState(order)
        self._orders[order.order_id] = state
        return state

    def process(self, event: MicrostructureEvent) -> list[ReplayFill]:
        if self._last_event_time is not None and event.event_time < self._last_event_time:
            raise ValueError("events must be replayed in chronological order")
        self._last_event_time = event.event_time
        new_fills: list[ReplayFill] = []
        for state in self._orders.values():
            if state.complete or event.event_time < state.order.active_at:
                continue
            if state.order.order_type == ReplayOrderType.MARKET:
                fill = self._market_fill(state, event)
            else:
                fill = self._maker_fill(state, event)
            if fill is not None:
                state.fills.append(fill)
                state.remaining_quantity = max(0.0, state.remaining_quantity - fill.quantity)
                new_fills.append(fill)
        return new_fills

    def state(self, order_id: str) -> ReplayOrderState:
        return self._orders[order_id]

    def _market_fill(
        self, state: ReplayOrderState, event: MicrostructureEvent
    ) -> ReplayFill | None:
        if event.event_type not in {
            MarketEventType.BOOK_TICKER,
            MarketEventType.DEPTH,
        }:
            return None
        if state.order.side == OrderSide.BUY:
            quote_price = event.ask_price
            quote_quantity = event.ask_quantity
            direction = 1
        else:
            quote_price = event.bid_price
            quote_quantity = event.bid_quantity
            direction = -1
        if quote_price is None or quote_quantity is None or quote_quantity <= 0:
            return None
        fill_quantity = min(state.remaining_quantity, quote_quantity)
        slippage_multiplier = 1 + (direction * self._config.market_slippage_bps / 10_000)
        return ReplayFill(
            order_id=state.order.order_id,
            event_time=event.event_time,
            side=state.order.side,
            quantity=fill_quantity,
            price=quote_price * slippage_multiplier,
            fee_bps=self._config.taker_fee_bps,
            liquidity_role=LiquidityRole.TAKER,
        )

    def _maker_fill(self, state: ReplayOrderState, event: MicrostructureEvent) -> ReplayFill | None:
        if event.event_type not in {
            MarketEventType.AGG_TRADE,
            MarketEventType.TRADE,
        }:
            return None
        if event.price is None or event.quantity is None or event.quantity <= 0:
            return None
        limit_price = state.order.limit_price
        if limit_price is None:
            return None
        execution_reaches_limit = (
            state.order.side == OrderSide.BUY
            and event.is_buyer_maker is True
            and event.price <= limit_price
        ) or (
            state.order.side == OrderSide.SELL
            and event.is_buyer_maker is False
            and event.price >= limit_price
        )
        if not execution_reaches_limit:
            return None
        executable_quantity = event.quantity
        if state.queue_ahead_quantity > 0:
            consumed_ahead = min(state.queue_ahead_quantity, executable_quantity)
            state.queue_ahead_quantity -= consumed_ahead
            executable_quantity -= consumed_ahead
        fill_quantity = min(state.remaining_quantity, executable_quantity)
        if fill_quantity <= 0:
            return None
        return ReplayFill(
            order_id=state.order.order_id,
            event_time=event.event_time,
            side=state.order.side,
            quantity=fill_quantity,
            price=limit_price,
            fee_bps=self._config.maker_fee_bps,
            liquidity_role=LiquidityRole.MAKER,
        )


def funding_cashflow(*, notional: float, funding_rate: float, side: OrderSide) -> float:
    """Return signed account cashflow: longs pay positive rates, shorts receive."""
    if not math.isfinite(notional) or notional < 0:
        raise ValueError("notional must be finite and non-negative")
    if not math.isfinite(funding_rate):
        raise ValueError("funding_rate must be finite")
    direction = -1 if side == OrderSide.BUY else 1
    return direction * notional * funding_rate

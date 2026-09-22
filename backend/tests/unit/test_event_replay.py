"""Event replay never invents fills before latency or visible liquidity."""

from datetime import UTC, datetime, timedelta

import pytest

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.backtest.event_replay import (
    EventExecutionReplay,
    LiquidityRole,
    OrderSide,
    ReplayExecutionConfig,
    ReplayOrder,
    ReplayOrderType,
    funding_cashflow,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def quote(milliseconds: int, *, ask_quantity: float = 1) -> MicrostructureEvent:
    return MicrostructureEvent(
        product="usdm_perpetual",
        symbol="BTCUSDT",
        event_type=MarketEventType.BOOK_TICKER,
        event_time=NOW + timedelta(milliseconds=milliseconds),
        bid_price=99,
        bid_quantity=2,
        ask_price=101,
        ask_quantity=ask_quantity,
    )


def trade(
    milliseconds: int,
    *,
    price: float,
    quantity: float,
    is_buyer_maker: bool,
) -> MicrostructureEvent:
    return MicrostructureEvent(
        product="usdm_perpetual",
        symbol="BTCUSDT",
        event_type=MarketEventType.TRADE,
        event_time=NOW + timedelta(milliseconds=milliseconds),
        price=price,
        quantity=quantity,
        is_buyer_maker=is_buyer_maker,
    )


def test_market_order_waits_for_latency_and_partial_fills_visible_quantity() -> None:
    replay = EventExecutionReplay(ReplayExecutionConfig(taker_fee_bps=5, market_slippage_bps=10))
    state = replay.submit(
        ReplayOrder(
            order_id="entry",
            side=OrderSide.BUY,
            quantity=1.5,
            submitted_at=NOW,
            latency_ms=100,
        )
    )

    assert replay.process(quote(99)) == []
    first = replay.process(quote(100, ask_quantity=1))
    second = replay.process(quote(101, ask_quantity=1))

    assert first[0].quantity == 1
    assert second[0].quantity == 0.5
    assert first[0].price == pytest.approx(101.101)
    assert first[0].liquidity_role == LiquidityRole.TAKER
    assert state.complete
    assert sum(fill.fee for fill in state.fills) > 0


def test_limit_order_requires_aggressor_cross_and_consumes_queue_ahead() -> None:
    replay = EventExecutionReplay()
    state = replay.submit(
        ReplayOrder(
            order_id="maker-buy",
            side=OrderSide.BUY,
            quantity=1,
            submitted_at=NOW,
            order_type=ReplayOrderType.LIMIT,
            limit_price=100,
            latency_ms=0,
            maker_queue_ahead_quantity=2,
        )
    )

    assert replay.process(trade(1, price=100, quantity=1, is_buyer_maker=False)) == []
    assert replay.process(trade(2, price=100, quantity=1.5, is_buyer_maker=True)) == []
    fills = replay.process(trade(3, price=99, quantity=1, is_buyer_maker=True))

    assert fills[0].quantity == 0.5
    assert state.remaining_quantity == 0.5


def test_replay_rejects_lookback_and_duplicate_order_ids() -> None:
    replay = EventExecutionReplay()
    order = ReplayOrder("one", OrderSide.SELL, 1, NOW)
    replay.submit(order)
    with pytest.raises(ValueError, match="duplicate"):
        replay.submit(order)
    replay.process(quote(100))
    with pytest.raises(ValueError, match="chronological"):
        replay.process(quote(99))


def test_funding_cashflow_has_correct_sign() -> None:
    assert funding_cashflow(notional=10_000, funding_rate=0.0001, side=OrderSide.BUY) == -1
    assert funding_cashflow(notional=10_000, funding_rate=0.0001, side=OrderSide.SELL) == 1

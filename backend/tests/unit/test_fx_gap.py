"""Tests for gap-aware protective fills."""

import pandas as pd
import pytest

from app.services.backtest.fx_gap import (
    NOT_TRIGGERED,
    ProtectiveFill,
    reopen_after_gap,
    resolve_protective_fill,
    stop_fill,
    take_profit_fill,
)


def _long_stop(**overrides) -> ProtectiveFill:
    bar = {
        "side": "LONG",
        "stop_price": 1.1000,
        "bar_open_bid": 1.1040,
        "bar_open_ask": 1.1041,
        "bar_low_bid": 1.1020,
        "bar_high_ask": 1.1060,
    }
    return stop_fill(**(bar | overrides))


def test_a_long_stop_is_not_triggered_while_the_bid_stays_above_it() -> None:
    assert _long_stop() == NOT_TRIGGERED


def test_a_long_stop_touched_inside_the_bar_fills_at_the_stop_price() -> None:
    fill = _long_stop(bar_low_bid=1.0995)

    assert fill == ProtectiveFill(triggered=True, price=1.1000, gapped=False)


def test_a_long_stop_crossed_by_a_sunday_gap_fills_at_the_gapped_open_not_the_stop() -> None:
    fill = _long_stop(bar_open_bid=1.0950, bar_open_ask=1.0953, bar_low_bid=1.0940)

    assert fill.triggered and fill.gapped
    assert fill.price == pytest.approx(1.0950)  # 50 pips worse than the 1.1000 stop


def test_a_long_stop_exactly_at_the_open_counts_as_gapped() -> None:
    fill = _long_stop(bar_open_bid=1.1000, bar_low_bid=1.0990)

    assert fill.gapped and fill.price == pytest.approx(1.1000)


def test_a_short_stop_is_compared_with_the_ask_and_buys_at_the_gapped_open() -> None:
    quiet = stop_fill(
        side="SHORT",
        stop_price=1.1100,
        bar_open_bid=1.1040,
        bar_open_ask=1.1041,
        bar_low_bid=1.1030,
        bar_high_ask=1.1090,
    )
    touched = stop_fill(
        side="SHORT",
        stop_price=1.1100,
        bar_open_bid=1.1040,
        bar_open_ask=1.1041,
        bar_low_bid=1.1030,
        bar_high_ask=1.1105,
    )
    gapped = stop_fill(
        side="SHORT",
        stop_price=1.1100,
        bar_open_bid=1.1150,
        bar_open_ask=1.1153,
        bar_low_bid=1.1140,
        bar_high_ask=1.1170,
    )

    assert quiet == NOT_TRIGGERED
    assert touched == ProtectiveFill(triggered=True, price=1.1100, gapped=False)
    assert gapped.gapped and gapped.price == pytest.approx(1.1153)


def test_take_profit_fills_at_the_limit_even_when_the_market_gaps_beyond_it() -> None:
    fill = take_profit_fill(side="LONG", limit_price=1.1100, bar_high_bid=1.1180, bar_low_ask=1.1150)

    assert fill == ProtectiveFill(triggered=True, price=1.1100)


def test_take_profit_needs_the_bid_to_reach_a_long_limit_and_the_ask_a_short_limit() -> None:
    assert take_profit_fill(
        side="LONG", limit_price=1.1100, bar_high_bid=1.1099, bar_low_ask=1.1000
    ) == NOT_TRIGGERED
    assert take_profit_fill(
        side="SHORT", limit_price=1.0900, bar_high_bid=1.1000, bar_low_ask=1.0901
    ) == NOT_TRIGGERED
    assert take_profit_fill(
        side="SHORT", limit_price=1.0900, bar_high_bid=1.1000, bar_low_ask=1.0900
    ).triggered


def test_when_a_bar_touches_both_orders_the_stop_wins() -> None:
    stop = ProtectiveFill(triggered=True, price=1.0990)
    target = ProtectiveFill(triggered=True, price=1.1100)

    assert resolve_protective_fill(stop, target) == stop
    assert resolve_protective_fill(NOT_TRIGGERED, target) == target
    assert resolve_protective_fill(NOT_TRIGGERED, NOT_TRIGGERED) == NOT_TRIGGERED


def test_reopen_after_gap_flags_only_the_first_bar_after_the_weekend() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-12 20:00",
            "2024-01-12 21:00",
            "2024-01-14 22:00",  # Sunday reopen: 49 hours later
            "2024-01-14 23:00",
            "2024-01-15 00:00",
        ],
        tz="UTC",
    )

    flags = reopen_after_gap(index)

    assert flags.tolist() == [False, False, True, False, False]


def test_reopen_after_gap_requires_a_sorted_index() -> None:
    unsorted = pd.DatetimeIndex(["2024-01-02", "2024-01-01"], tz="UTC")

    with pytest.raises(ValueError, match="sorted"):
        reopen_after_gap(unsorted)

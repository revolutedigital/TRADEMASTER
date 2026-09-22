"""Trailing changes affect only later events and portfolios reject overlaps."""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.event_replay import OrderSide
from app.services.backtest.trailing_portfolio import (
    ExitReason,
    HistoricalTrailingSimulator,
    TrailingPolicy,
    replay_one_position_portfolio,
    summarize_portfolio,
)


POLICY = TrailingPolicy("test", 100, 100, 20, 150, 50)


def test_trailing_stop_activates_after_causal_event_and_locks_profit() -> None:
    simulator = HistoricalTrailingSimulator(
        np.array([0, 100, 200, 300, 400]),
        np.array([100, 100, 102, 101.4, 99]),
        latency_ms=100,
        expected_round_trip_bps=10,
        stress_round_trip_bps=20,
    )

    trade = simulator.simulate(
        decision_time_ms=0,
        side=OrderSide.BUY,
        horizon_seconds=1,
        probability=0.9,
        policy=POLICY,
    )

    assert trade.entry_time_ms == 100
    assert trade.exit_time_ms == 300
    assert trade.exit_reason == ExitReason.TRAILING_STOP
    assert trade.gross_bps > 100
    assert trade.expected_net_bps == pytest.approx(trade.gross_bps - 10)


def test_gap_through_initial_stop_fills_at_observed_worse_price() -> None:
    simulator = HistoricalTrailingSimulator(
        np.array([0, 100, 200]),
        np.array([100, 100, 95]),
        latency_ms=100,
    )
    trade = simulator.simulate(
        decision_time_ms=0,
        side=OrderSide.BUY,
        horizon_seconds=1,
        probability=0.8,
        policy=POLICY,
    )
    assert trade.exit_reason == ExitReason.INITIAL_STOP
    assert trade.exit_price == 95


def test_partial_profit_is_banked_before_runner_reverses() -> None:
    policy = TrailingPolicy("hybrid", 100, 50, 20, 100, 50, 50, 0.5)
    simulator = HistoricalTrailingSimulator(
        np.array([0, 100, 200, 300]),
        np.array([100, 100, 101, 100.2]),
        latency_ms=100,
        expected_round_trip_bps=10,
    )

    trade = simulator.simulate(
        decision_time_ms=0,
        side=OrderSide.BUY,
        horizon_seconds=1,
        probability=0.8,
        policy=policy,
    )

    assert trade.partial_take_fraction == 0.5
    assert trade.gross_bps > 30
    assert trade.expected_net_bps > 20


def test_portfolio_keeps_only_one_position() -> None:
    simulator = HistoricalTrailingSimulator(
        np.arange(0, 2300, 100),
        np.linspace(100, 101, 23),
        latency_ms=0,
    )
    predictions = pd.DataFrame(
        {
            "decision_time_ms": [0, 100, 1100],
            "side": ["BUY", "SELL", "BUY"],
            "horizon_seconds": [1, 1, 1],
            "probability": [0.9, 0.8, 0.9],
            "threshold_top_5%": [0.7, 0.7, 0.7],
        }
    )

    trades = replay_one_position_portfolio(
        predictions, simulator, tail_fraction=0.05, policy=POLICY
    )
    summary = summarize_portfolio(trades)

    assert len(trades) == 2
    assert summary["trade_count"] == 2

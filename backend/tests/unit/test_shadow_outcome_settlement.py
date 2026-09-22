"""Shadow settlement replays hypothetical signals without exchange execution."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.models.research_experiment import ResearchShadowSignal
from app.services.backtest.trailing_portfolio import HistoricalTrailingSimulator, TrailingPolicy
from app.services.research.shadow_outcome_settlement import (
    settle_shadow_signal,
    settle_shadow_signals,
)


POLICY = TrailingPolicy("test", 100, 100, 20, 150, 50)


def test_settle_shadow_signal_replays_would_enter_signal() -> None:
    simulator = HistoricalTrailingSimulator(
        np.array([0, 100, 200, 300, 400]),
        np.array([100, 100, 102, 101.4, 99]),
        latency_ms=100,
        expected_round_trip_bps=10,
        stress_round_trip_bps=20,
    )
    signal = _signal(probability=0.9, threshold=0.7, would_enter=True)

    settlement = settle_shadow_signal(signal, simulator=simulator, policy=POLICY)

    assert settlement.signal_id == 1
    assert settlement.would_enter is True
    assert settlement.expected_net_bps > 0
    assert settlement.stress_net_bps == settlement.expected_net_bps - 10
    assert len(settlement.label_sha256) == 64
    assert settlement.to_recorder_kwargs() == {
        "signal_id": 1,
        "expected_net_bps": settlement.expected_net_bps,
        "stress_net_bps": settlement.stress_net_bps,
        "label_sha256": settlement.label_sha256,
    }


def test_settle_shadow_signal_returns_zero_for_no_entry_decision() -> None:
    simulator = HistoricalTrailingSimulator(
        np.array([0, 100, 200]),
        np.array([100, 120, 140]),
        latency_ms=0,
    )
    signal = _signal(probability=0.2, threshold=0.7, would_enter=False)

    settlement = settle_shadow_signal(signal, simulator=simulator, policy=POLICY)

    assert settlement.would_enter is False
    assert settlement.expected_net_bps == 0
    assert settlement.stress_net_bps == 0
    assert len(settlement.label_sha256) == 64


def test_settle_shadow_signals_is_deterministic_for_batches() -> None:
    simulator = HistoricalTrailingSimulator(
        np.array([0, 100, 200, 300, 400]),
        np.array([100, 100, 102, 101.4, 99]),
        latency_ms=100,
        expected_round_trip_bps=10,
        stress_round_trip_bps=20,
    )
    signals = [
        _signal(signal_id=1, probability=0.9, threshold=0.7, would_enter=True),
        _signal(signal_id=2, probability=0.2, threshold=0.7, would_enter=False),
    ]

    first = settle_shadow_signals(signals, simulator=simulator, policy=POLICY)
    second = settle_shadow_signals(signals, simulator=simulator, policy=POLICY)

    assert first == second
    assert [row.signal_id for row in first] == [1, 2]


def _signal(
    *,
    signal_id: int = 1,
    probability: float,
    threshold: float,
    would_enter: bool,
) -> ResearchShadowSignal:
    return ResearchShadowSignal(
        id=signal_id,
        experiment_id="experiment",
        decision_time=datetime.fromtimestamp(0, tz=UTC),
        recorded_at=datetime.fromtimestamp(0, tz=UTC),
        side="BUY",
        horizon_seconds=1,
        probability=probability,
        threshold=threshold,
        would_enter=would_enter,
        model_sha256="d" * 64,
        feature_vector_sha256="e" * 64,
    )

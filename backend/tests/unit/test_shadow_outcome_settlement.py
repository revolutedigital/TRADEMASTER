"""Shadow settlement replays hypothetical signals without exchange execution."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.research_experiment import ResearchExperiment, ResearchShadowSignal
from app.services.backtest.trailing_portfolio import HistoricalTrailingSimulator, TrailingPolicy
from app.services.research.shadow_outcome_settlement import (
    settle_pending_shadow_outcomes,
    settle_shadow_signal,
    settle_shadow_signals,
)


POLICY = TrailingPolicy("test", 100, 100, 20, 150, 50)


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


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


@pytest.mark.asyncio
async def test_settle_pending_shadow_outcomes_records_only_mature_signals(
    db: AsyncSession,
) -> None:
    now = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
    mature_time = now - timedelta(seconds=180)
    immature_time = now - timedelta(seconds=60)
    db.add(
        ResearchExperiment(
            id="experiment",
            name="shadow",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    mature = _signal(
        signal_id=None,
        decision_time=mature_time,
        horizon_seconds=120,
        probability=0.9,
        threshold=0.7,
        would_enter=True,
    )
    immature = _signal(
        signal_id=None,
        decision_time=immature_time,
        horizon_seconds=300,
        probability=0.9,
        threshold=0.7,
        would_enter=True,
    )
    db.add_all([mature, immature])
    await db.flush()
    decision_ms = int(mature_time.timestamp() * 1000)
    simulator = HistoricalTrailingSimulator(
        np.array([decision_ms, decision_ms + 100, decision_ms + 1_000, decision_ms + 120_000]),
        np.array([100, 100, 101, 102]),
        latency_ms=100,
        expected_round_trip_bps=10,
        stress_round_trip_bps=20,
    )

    settlements = await settle_pending_shadow_outcomes(
        db,
        experiment_id="experiment",
        simulator=simulator,
        policy=POLICY,
        now=now,
    )

    assert [settlement.signal_id for settlement in settlements] == [mature.id]
    assert mature.outcome_json is not None
    assert immature.outcome_json is None
    outcome = json.loads(mature.outcome_json)
    assert outcome["research_only"] is True
    assert outcome["order_submission_allowed"] is False
    assert "order_id" not in outcome


def _signal(
    *,
    signal_id: int | None = 1,
    decision_time: datetime | None = None,
    horizon_seconds: int = 1,
    probability: float,
    threshold: float,
    would_enter: bool,
) -> ResearchShadowSignal:
    return ResearchShadowSignal(
        id=signal_id,
        experiment_id="experiment",
        decision_time=decision_time or datetime.fromtimestamp(0, tz=UTC),
        recorded_at=datetime.fromtimestamp(0, tz=UTC),
        side="BUY",
        horizon_seconds=horizon_seconds,
        probability=probability,
        threshold=threshold,
        would_enter=would_enter,
        model_sha256="d" * 64,
        feature_vector_sha256="e" * 64,
    )

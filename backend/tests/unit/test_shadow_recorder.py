"""Shadow evidence is append-only and impossible before explicit partition open."""

import json
import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.research_experiment import ResearchDataUse, ResearchExperiment
from app.services.research.shadow_recorder import (
    ResearchShadowRecorder,
    ShadowRecorderError,
)


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


async def seed(db: AsyncSession, *, opened: bool, status: str = "FROZEN") -> None:
    now = datetime.now(UTC)
    db.add(
        ResearchExperiment(
            id="experiment",
            name="shadow",
            status=status,
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    db.add(
        ResearchDataUse(
            experiment_id="experiment",
            role="PROSPECTIVE_SHADOW",
            start_at=now,
            end_at=now + timedelta(days=20),
            manifest_sha256="c" * 64,
            opened_at=now if opened else None,
        )
    )
    await db.flush()


@pytest.mark.asyncio
async def test_shadow_signal_records_hypothetical_decision_only(db: AsyncSession) -> None:
    await seed(db, opened=True)
    signal = await ResearchShadowRecorder().record(
        db,
        experiment_id="experiment",
        decision_time=datetime.now(UTC),
        side="BUY",
        horizon_seconds=300,
        probability=0.8,
        threshold=0.7,
        model_sha256="d" * 64,
        feature_vector={"flow": 0.5},
    )
    assert signal.would_enter is True
    assert signal.outcome_json is None
    assert not hasattr(signal, "order_id")


@pytest.mark.asyncio
async def test_shadow_signal_fails_before_prospective_partition_open(db: AsyncSession) -> None:
    await seed(db, opened=False)
    with pytest.raises(ShadowRecorderError, match="explicitly opened"):
        await ResearchShadowRecorder().record(
            db,
            experiment_id="experiment",
            decision_time=datetime.now(UTC),
            side="SELL",
            horizon_seconds=120,
            probability=0.2,
            threshold=0.7,
            model_sha256="d" * 64,
            feature_vector={"flow": -0.5},
        )


@pytest.mark.asyncio
async def test_shadow_outcome_is_recorded_once_without_execution_fields(db: AsyncSession) -> None:
    await seed(db, opened=True)
    recorder = ResearchShadowRecorder()
    signal = await recorder.record(
        db,
        experiment_id="experiment",
        decision_time=datetime.now(UTC),
        side="BUY",
        horizon_seconds=300,
        probability=0.8,
        threshold=0.7,
        model_sha256="d" * 64,
        feature_vector={"flow": 0.5},
    )

    updated = await recorder.record_outcome(
        db,
        signal_id=signal.id,
        expected_net_bps=1.2,
        stress_net_bps=0.4,
        label_sha256="e" * 64,
    )

    outcome = json.loads(updated.outcome_json or "{}")
    assert outcome["expected_net_bps"] == 1.2
    assert outcome["stress_net_bps"] == 0.4
    assert outcome["research_only"] is True
    assert outcome["order_submission_allowed"] is False
    assert outcome["execution_authorization"] == "none"
    assert "order_id" not in outcome
    with pytest.raises(ShadowRecorderError, match="immutable"):
        await recorder.record_outcome(
            db,
            signal_id=signal.id,
            expected_net_bps=2.0,
            stress_net_bps=1.0,
            label_sha256="e" * 64,
        )


@pytest.mark.asyncio
async def test_shadow_outcome_rejects_non_finite_values(db: AsyncSession) -> None:
    await seed(db, opened=True)
    recorder = ResearchShadowRecorder()
    signal = await recorder.record(
        db,
        experiment_id="experiment",
        decision_time=datetime.now(UTC),
        side="SELL",
        horizon_seconds=120,
        probability=0.8,
        threshold=0.7,
        model_sha256="d" * 64,
        feature_vector={"flow": 0.5},
    )

    with pytest.raises(ShadowRecorderError, match="finite"):
        await recorder.record_outcome(
            db,
            signal_id=signal.id,
            expected_net_bps=math.nan,
            stress_net_bps=0.4,
            label_sha256="e" * 64,
        )

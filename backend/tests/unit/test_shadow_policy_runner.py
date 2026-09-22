"""Frozen policy shadow runner records scored decisions without execution fields."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.research_experiment import ResearchDataUse, ResearchExperiment, ResearchShadowSignal
from app.services.research.shadow_policy_runner import (
    ShadowPolicyRunnerError,
    record_frozen_top_p_shadow_batch,
    record_frozen_top_p_shadow_decision,
    record_frozen_top_p_shadow_selection,
    record_frozen_top_p_shadow_signal,
    score_frozen_top_p_shadow_frame,
)
from app.services.research.top_p_model import (
    freeze_top_p_policy,
    predict_frozen_top_p_probability,
)
from scripts.research.run_shadow_policy_batch import _write_json_report


@pytest.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_frozen_policy_runner_records_scored_shadow_signal(db: AsyncSession) -> None:
    now = datetime.now(UTC)
    await _seed_shadow_experiment(
        db,
        partition_start=now - timedelta(days=1),
        partition_end=now + timedelta(days=19),
    )
    artifact = _artifact()
    feature_vector = _feature_vector()
    probability = predict_frozen_top_p_probability(artifact, feature_vector)

    signal = await record_frozen_top_p_shadow_signal(
        db,
        experiment_id="experiment",
        decision_time=datetime.now(UTC),
        side="BUY",
        artifact=artifact,
        feature_vector=feature_vector,
    )

    assert signal.model_sha256 == artifact["model_sha256"]
    assert signal.probability == pytest.approx(probability)
    assert signal.threshold == artifact["probability_threshold"]
    assert signal.would_enter is (probability >= artifact["probability_threshold"])
    assert not hasattr(signal, "order_id")


@pytest.mark.asyncio
async def test_frozen_policy_runner_rejects_tampered_artifact(db: AsyncSession) -> None:
    now = datetime.now(UTC)
    await _seed_shadow_experiment(
        db,
        partition_start=now - timedelta(days=1),
        partition_end=now + timedelta(days=19),
    )
    artifact = _artifact()
    artifact["probability_threshold"] = 0.0

    with pytest.raises(ShadowPolicyRunnerError, match="hash"):
        await record_frozen_top_p_shadow_signal(
            db,
            experiment_id="experiment",
            decision_time=datetime.now(UTC),
            side="BUY",
            artifact=artifact,
            feature_vector=_feature_vector(),
        )


def test_frozen_policy_shadow_scoring_selects_entries_only() -> None:
    artifact = _artifact()
    frame = _shadow_frame()

    selection = score_frozen_top_p_shadow_frame(frame, artifact=artifact)

    assert selection.model_sha256 == artifact["model_sha256"]
    assert selection.scored_rows == 3
    assert selection.selected_count == 2
    assert {decision.side for decision in selection.decisions} == {"BUY", "SELL"}
    assert all(decision.would_enter for decision in selection.decisions)
    assert all(decision.probability >= decision.threshold for decision in selection.decisions)
    assert selection.order_submission_allowed is False
    assert selection.execution_authorization == "none"


def test_shadow_policy_batch_cli_report_writer_creates_json(tmp_path) -> None:
    output = tmp_path / "shadow" / "batch-report.json"
    payload = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "dry_run": True,
    }

    _write_json_report(output, payload)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.asyncio
async def test_frozen_policy_shadow_batch_records_entries_idempotently(
    db: AsyncSession,
) -> None:
    partition_start = datetime(2026, 1, 1, tzinfo=UTC)
    await _seed_shadow_experiment(
        db,
        partition_start=partition_start,
        partition_end=partition_start + timedelta(days=30),
    )
    artifact = _artifact()

    first = await record_frozen_top_p_shadow_batch(
        db,
        experiment_id="experiment",
        artifact=artifact,
        frame=_shadow_frame(),
    )
    second = await record_frozen_top_p_shadow_batch(
        db,
        experiment_id="experiment",
        artifact=artifact,
        frame=_shadow_frame(),
    )

    signals = (
        await db.execute(
            select(ResearchShadowSignal).order_by(
                ResearchShadowSignal.decision_time,
                ResearchShadowSignal.side,
            )
        )
    ).scalars().all()
    assert first.scored_rows == 3
    assert first.selected_count == 2
    assert first.recorded_count == 2
    assert first.skipped_existing_count == 0
    assert second.recorded_count == 0
    assert second.skipped_existing_count == 2
    assert len(signals) == 2
    assert {signal.side for signal in signals} == {"BUY", "SELL"}
    assert all(signal.would_enter for signal in signals)
    assert all(signal.model_sha256 == artifact["model_sha256"] for signal in signals)
    assert first.order_submission_allowed is False
    assert first.execution_authorization == "none"


@pytest.mark.asyncio
async def test_frozen_policy_shadow_selection_records_prescored_online_entries_idempotently(
    db: AsyncSession,
) -> None:
    partition_start = datetime(2026, 1, 1, tzinfo=UTC)
    await _seed_shadow_experiment(
        db,
        partition_start=partition_start,
        partition_end=partition_start + timedelta(days=30),
    )
    artifact = _artifact()
    selection = score_frozen_top_p_shadow_frame(_shadow_frame(), artifact=artifact)

    first = await record_frozen_top_p_shadow_selection(
        db,
        experiment_id="experiment",
        selection=selection,
    )
    second = await record_frozen_top_p_shadow_selection(
        db,
        experiment_id="experiment",
        selection=selection,
    )

    signals = (await db.execute(select(ResearchShadowSignal))).scalars().all()
    assert first.scored_rows == 3
    assert first.selected_count == 2
    assert first.recorded_count == 2
    assert first.skipped_existing_count == 0
    assert second.recorded_count == 0
    assert second.skipped_existing_count == 2
    assert len(signals) == 2
    assert {signal.feature_vector_sha256 for signal in signals} == {
        decision.feature_vector_sha256 for decision in selection.decisions
    }
    assert first.order_submission_allowed is False
    assert first.execution_authorization == "none"


@pytest.mark.asyncio
async def test_event_driven_frozen_policy_shadow_decision_records_selected_idempotently(
    db: AsyncSession,
) -> None:
    partition_start = datetime(2026, 1, 1, tzinfo=UTC)
    await _seed_shadow_experiment(
        db,
        partition_start=partition_start,
        partition_end=partition_start + timedelta(days=30),
    )
    artifact = _artifact()
    decision_time = datetime(2026, 1, 8, tzinfo=UTC)

    first = await record_frozen_top_p_shadow_decision(
        db,
        experiment_id="experiment",
        decision_time=decision_time,
        side="BUY",
        artifact=artifact,
        feature_vector=_base_feature_vector(),
    )
    second = await record_frozen_top_p_shadow_decision(
        db,
        experiment_id="experiment",
        decision_time=decision_time,
        side="BUY",
        artifact=artifact,
        feature_vector=_base_feature_vector(),
    )

    signals = (await db.execute(select(ResearchShadowSignal))).scalars().all()
    assert first.recorded is True
    assert first.skipped_existing is False
    assert first.signal is not None
    assert first.decision.would_enter is True
    assert second.recorded is False
    assert second.skipped_existing is True
    assert second.signal is not None
    assert second.signal.id == first.signal.id
    assert len(signals) == 1
    assert first.order_submission_allowed is False
    assert first.execution_authorization == "none"


@pytest.mark.asyncio
async def test_event_driven_frozen_policy_shadow_decision_skips_non_entry_by_default(
    db: AsyncSession,
) -> None:
    partition_start = datetime(2026, 1, 1, tzinfo=UTC)
    await _seed_shadow_experiment(
        db,
        partition_start=partition_start,
        partition_end=partition_start + timedelta(days=30),
    )

    result = await record_frozen_top_p_shadow_decision(
        db,
        experiment_id="experiment",
        decision_time=datetime(2026, 1, 8, tzinfo=UTC),
        side="BUY",
        artifact=_artifact(),
        feature_vector=_loser_feature_vector(),
    )

    signals = (await db.execute(select(ResearchShadowSignal))).scalars().all()
    assert result.recorded is False
    assert result.skipped_existing is False
    assert result.signal is None
    assert result.decision.would_enter is False
    assert len(signals) == 0


async def _seed_shadow_experiment(
    db: AsyncSession,
    *,
    partition_start: datetime,
    partition_end: datetime,
) -> None:
    now = datetime.now(UTC)
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
    db.add(
        ResearchDataUse(
            experiment_id="experiment",
            role="PROSPECTIVE_SHADOW",
            start_at=partition_start,
            end_at=partition_end,
            manifest_sha256="c" * 64,
            opened_at=now,
        )
    )
    await db.flush()


def _artifact() -> dict[str, object]:
    policy = freeze_top_p_policy(
        _model_frame(),
        horizon_seconds=120,
        feature_set="flow",
        tail_fraction=0.10,
        calibration_date="2026-01-06",
        dataset_manifest_sha256="d" * 64,
        embargo_seconds=300,
    )
    return policy.to_dict()


def _feature_vector() -> dict[str, float]:
    return {
        "flow_imbalance_1s": 2.0,
        "directed_flow_imbalance_1s": 2.0,
        "trade_count_1s": 12.0,
        "quote_volume_1s": 140.0,
        "mean_interarrival_ms_1s": 20.0,
    }


def _base_feature_vector() -> dict[str, float]:
    return {
        "flow_imbalance_1s": 2.0,
        "trade_count_1s": 12.0,
        "quote_volume_1s": 140.0,
        "mean_interarrival_ms_1s": 20.0,
    }


def _loser_feature_vector() -> dict[str, float]:
    return {
        "flow_imbalance_1s": -3.0,
        "directed_flow_imbalance_1s": -3.0,
        "trade_count_1s": 13.0,
        "quote_volume_1s": 160.0,
        "mean_interarrival_ms_1s": 20.0,
    }


def _shadow_frame() -> pd.DataFrame:
    base_time = pd.Timestamp("2026-01-08T00:00:00Z")
    rows = []
    for offset, side, feature_vector in (
        (0, "BUY", _feature_vector()),
        (5, "SELL", _feature_vector()),
        (10, "BUY", _loser_feature_vector()),
    ):
        rows.append(
            {
                "decision_time_ms": int(
                    (base_time + pd.Timedelta(seconds=offset)).timestamp() * 1000
                ),
                "side": side,
                "horizon_seconds": 120,
                **feature_vector,
            }
        )
    return pd.DataFrame(rows)


def _model_frame() -> pd.DataFrame:
    random = np.random.default_rng(456)
    rows = []
    for day in range(7):
        day_start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
        for sample in range(60):
            signal = random.normal()
            rows.append(
                {
                    "decision_time_ms": int(
                        (day_start + pd.Timedelta(minutes=sample * 10)).timestamp() * 1000
                    ),
                    "horizon_seconds": 120,
                    "target": int(signal + random.normal(scale=0.3) > 0),
                    "flow_imbalance_1s": signal,
                    "directed_flow_imbalance_1s": signal,
                    "trade_count_1s": 10 + abs(signal),
                    "quote_volume_1s": 100 + abs(signal) * 20,
                    "mean_interarrival_ms_1s": 20,
                }
            )
    return pd.DataFrame(rows)

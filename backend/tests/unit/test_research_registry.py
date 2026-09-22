"""Research registry freezes definitions and refuses burned-data reuse."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchExperimentEvent,
    ResearchHypothesisAttempt,
)
from app.repositories.research_experiment_repo import (
    research_experiment_repository,
    verify_research_event_chain,
)
from app.services.data.research_registry import (
    BurnedDataConflict,
    ExperimentDefinition,
    FrozenExperimentError,
    PartitionDefinition,
    ResearchRegistry,
    ResearchRegistryError,
    V1_GATE,
    V1_PRODUCT,
)


@pytest.fixture
async def database() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        for table in (
            ResearchExperiment.__table__,
            ResearchDataUse.__table__,
            ResearchHypothesisAttempt.__table__,
            ResearchExperimentEvent.__table__,
        ):
            await connection.run_sync(table.create)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _definition(*, audit_hash: str = _hash("a"), training_hash: str = _hash("t")):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    day = timedelta(days=1)
    return ExperimentDefinition(
        name="BTCUSDT path probability v1",
        code_revision="1" * 40,
        protocol_sha256="2" * 64,
        product=dict(V1_PRODUCT),
        cost_profile={
            "maker_fee_bps_per_side": 2,
            "taker_fee_bps_per_side": 5,
            "expected_slippage_bps_per_side": 1,
            "expected_latency_ms": 100,
            "stress_roundtrip_bps": 24,
            "default_order_style": "marketable_taker",
        },
        approval_gate={
            **V1_GATE,
            "min_expected_cost_lcb_bps": 0.0001,
            "min_stress_cost_mean_bps": 0.0001,
        },
        partitions=(
            PartitionDefinition("DEVELOPMENT", start, start + day, _hash("d")),
            PartitionDefinition("TRAINING", start + day, start + 2 * day, training_hash),
            PartitionDefinition("SELECTION", start + 2 * day, start + 3 * day, _hash("s")),
            PartitionDefinition("AUDIT", start + 3 * day, start + 4 * day, audit_hash),
        ),
    )


async def _draft_with_hypothesis(
    registry: ResearchRegistry,
    database: AsyncSession,
    *,
    audit_hash: str = _hash("a"),
    training_hash: str = _hash("t"),
) -> ResearchExperiment:
    experiment = await registry.create_draft(
        database,
        _definition(audit_hash=audit_hash, training_hash=training_hash),
    )
    await registry.register_hypothesis(
        database,
        experiment.id,
        kind="PATH_MODEL",
        definition={"horizon_seconds": 30, "top_p_pct": 5},
    )
    return experiment


@pytest.mark.asyncio
async def test_freeze_is_deterministic_and_definition_becomes_immutable(database) -> None:
    registry = ResearchRegistry()
    experiment = await _draft_with_hypothesis(registry, database)

    frozen = await registry.freeze(database, experiment.id)
    first_hash = frozen.experiment_sha256

    assert frozen.status == "FROZEN"
    assert first_hash is not None and len(first_hash) == 64
    with pytest.raises(FrozenExperimentError):
        await registry.register_hypothesis(
            database,
            experiment.id,
            kind="PATH_MODEL",
            definition={"horizon_seconds": 120, "top_p_pct": 1},
        )
    with pytest.raises(FrozenExperimentError):
        await registry.freeze(database, experiment.id)
    assert frozen.experiment_sha256 == first_hash


@pytest.mark.asyncio
async def test_create_draft_requires_book_evidence_gate(database) -> None:
    registry = ResearchRegistry()
    definition = _definition()
    definition.approval_gate.pop("book_evidence_min_complete_days")

    with pytest.raises(ValueError, match="book_evidence_min_complete_days"):
        await registry.create_draft(database, definition)


@pytest.mark.asyncio
async def test_prospective_shadow_partition_must_be_20_to_30_days(database) -> None:
    registry = ResearchRegistry()
    definition = _definition()
    audit_end = definition.partitions[-1].end_at
    short_shadow = PartitionDefinition(
        "PROSPECTIVE_SHADOW",
        audit_end,
        audit_end + timedelta(days=19, hours=23),
        _hash("p"),
    )
    definition = ExperimentDefinition(
        name=definition.name,
        code_revision=definition.code_revision,
        protocol_sha256=definition.protocol_sha256,
        product=definition.product,
        cost_profile=definition.cost_profile,
        approval_gate=definition.approval_gate,
        partitions=(*definition.partitions, short_shadow),
    )

    with pytest.raises(ValueError, match="20 to 30 days"):
        await registry.create_draft(database, definition)


@pytest.mark.asyncio
async def test_opened_audit_partition_is_burned_globally(database) -> None:
    registry = ResearchRegistry()
    burned_hash = _hash("b")
    first = await _draft_with_hypothesis(registry, database, audit_hash=burned_hash)
    await registry.freeze(database, first.id)
    await registry.open_partition(database, first.id, role="AUDIT")

    second = await _draft_with_hypothesis(
        registry,
        database,
        audit_hash=_hash("z"),
        training_hash=burned_hash,
    )

    with pytest.raises(BurnedDataConflict):
        await registry.freeze(database, second.id)


@pytest.mark.asyncio
async def test_open_partition_is_idempotent_and_draft_cannot_open_data(database) -> None:
    registry = ResearchRegistry()
    experiment = await _draft_with_hypothesis(registry, database)

    with pytest.raises(ValueError, match="FROZEN"):
        await registry.open_partition(database, experiment.id, role="TRAINING")

    await registry.freeze(database, experiment.id)
    first = await registry.open_partition(database, experiment.id, role="TRAINING")
    second = await registry.open_partition(database, experiment.id, role="TRAINING")

    assert first.id == second.id
    assert first.opened_at == second.opened_at


@pytest.mark.asyncio
async def test_decision_is_terminal_and_carries_no_execution_authority(database) -> None:
    registry = ResearchRegistry()
    experiment = await _draft_with_hypothesis(registry, database)
    await registry.freeze(database, experiment.id)

    approved = await registry.record_decision(
        database,
        experiment.id,
        status="APPROVED",
        reasons=["All offline and prospective evidence gates passed."],
        evidence={
            "statistical_gate_sha256": "d" * 64,
            "statistical_gate_decision": "APPROVED",
            "order_submission_allowed": False,
            "execution_authorization": "none",
        },
    )

    assert approved.status == "APPROVED"
    assert not hasattr(approved, "target_execution_mode")
    with pytest.raises(FrozenExperimentError):
        await registry.record_decision(
            database,
            experiment.id,
            status="REJECTED",
            reasons=["Cannot overwrite an approval."],
        )


@pytest.mark.asyncio
async def test_lifecycle_events_are_hash_chained_and_tamper_evident(database) -> None:
    registry = ResearchRegistry()
    experiment = await _draft_with_hypothesis(registry, database)
    await registry.freeze(database, experiment.id)
    await registry.open_partition(database, experiment.id, role="TRAINING")

    events = await research_experiment_repository.list_events(database, experiment.id)
    status = verify_research_event_chain(events)

    assert [event.kind for event in events] == [
        "DRAFT_CREATED",
        "HYPOTHESIS_REGISTERED",
        "EXPERIMENT_FROZEN",
        "PARTITION_OPENED",
    ]
    assert status.verified is True
    assert status.event_count == 4
    assert status.latest_event_sha256 == events[-1].event_sha256
    assert events[0].previous_event_sha256 is None
    for previous_event, current_event in zip(events, events[1:]):
        assert len(previous_event.event_sha256 or "") == 64
        assert current_event.previous_event_sha256 == previous_event.event_sha256

    events[1].payload_json = '{"kind":"tampered"}'
    tampered_status = verify_research_event_chain(events)

    assert tampered_status.verified is False
    assert tampered_status.reasons == (f"event_{events[1].id}_hash_mismatch",)


@pytest.mark.asyncio
async def test_approved_decision_requires_gate_evidence(database) -> None:
    registry = ResearchRegistry()
    experiment = await _draft_with_hypothesis(registry, database)
    await registry.freeze(database, experiment.id)

    with pytest.raises(ResearchRegistryError, match="require gate evidence"):
        await registry.record_decision(
            database,
            experiment.id,
            status="APPROVED",
            reasons=["Cannot approve without statistical gate evidence."],
        )

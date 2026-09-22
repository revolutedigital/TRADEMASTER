"""Research-only evidence-gate API never grants execution access."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.v1 import research
from app.models.base import Base
from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchExperimentEvent,
    ResearchHypothesisAttempt,
    ResearchShadowSignal,
)
from app.repositories.research_experiment_repo import (
    build_research_event_hash,
    research_experiment_repository,
)
from app.schemas.research_experiment import (
    RecordExperimentDecisionRequest,
    RecordFrozenTopPShadowSignalRequest,
    RecordShadowOutcomeRequest,
    RecordShadowSignalRequest,
    RecordTestnetReleaseRequest,
)
from app.services.market.microstructure_wal_audit import build_evidence_gate_status
from app.services.research.shadow_recorder import (
    shadow_outcome_event_payload,
    shadow_signal_event_payload,
)
from app.services.research.top_p_model import freeze_top_p_policy


@pytest.fixture
async def db() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def test_evidence_gate_api_returns_ineligible_placeholder_when_artifact_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_artifact = tmp_path / "missing.json"
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(missing_artifact),
    )

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is False
    assert response.book_evidence_gate.eligible is False
    assert response.status_reasons == ["evidence_gate_artifact_missing"]
    assert response.safety.order_submission_allowed is False
    assert response.safety.execution_authorization == "none"


async def test_evidence_gate_api_reads_valid_small_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    payload = build_evidence_gate_status(
        (),
        artifact_available=True,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_reasons=("no_complete_days_yet",),
    )
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is True
    assert response.audited_start_date is None
    assert response.audited_end_date is None
    assert response.book_evidence_gate.required_complete_days == 60
    assert response.book_evidence_gate.required_streams == (
        "TRADE",
        "DEPTH",
        "MARK_PRICE",
        "SPOT_TRADE",
    )
    assert response.book_evidence_gate.streak_start is None
    assert response.book_evidence_gate.streak_end is None
    assert response.status_reasons == ["no_complete_days_yet"]


async def test_evidence_gate_api_reads_remote_artifact_when_configured(monkeypatch) -> None:
    payload = build_evidence_gate_status(
        (),
        artifact_available=True,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {"timeout": 3.0, "follow_redirects": False}

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            assert url == "http://recorder.internal/evidence-gate-status.json"
            return FakeResponse()

    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_url",
        "http://recorder.internal/evidence-gate-status.json",
    )
    monkeypatch.setattr(research.httpx, "AsyncClient", FakeAsyncClient)

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is True
    assert response.book_evidence_gate.required_streams == (
        "TRADE",
        "DEPTH",
        "MARK_PRICE",
        "SPOT_TRADE",
    )
    assert response.safety.order_submission_allowed is False


async def test_evidence_gate_api_fails_closed_when_remote_artifact_is_unavailable(
    monkeypatch,
) -> None:
    class FakeAsyncClient:
        def __init__(self, **_kwargs) -> None:
            return None

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, _url: str) -> Response:
            raise research.httpx.ConnectError("recorder unavailable")

    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_url",
        "http://recorder.internal/evidence-gate-status.json",
    )
    monkeypatch.setattr(research.httpx, "AsyncClient", FakeAsyncClient)

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is False
    assert response.book_evidence_gate.eligible is False
    assert response.status_reasons == [
        "evidence_gate_artifact_remote_unreadable:ConnectError"
    ]
    assert response.safety.order_submission_allowed is False
    assert response.safety.execution_authorization == "none"


async def test_evidence_gate_api_rejects_unreadable_artifact_as_ineligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is False
    assert response.book_evidence_gate.eligible is False
    assert response.status_reasons == ["evidence_gate_artifact_unreadable:JSONDecodeError"]


def test_evidence_gate_status_accepts_iso_dates_in_artifact() -> None:
    payload = build_evidence_gate_status(
        (),
        artifact_available=False,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    payload["audited_start_date"] = "2026-01-01"
    payload["audited_end_date"] = "2026-01-02"
    payload["book_evidence_gate"]["streak_start"] = "2026-01-01"
    payload["book_evidence_gate"]["streak_end"] = "2026-01-02"
    payload["book_evidence_gate"]["incomplete_days"] = ["2026-01-01"]
    payload["book_evidence_gate"].pop("required_streams", None)

    parsed = research.EvidenceGateStatusResponse.model_validate(payload)

    assert parsed.audited_start_date == date(2026, 1, 1)
    assert parsed.audited_end_date == date(2026, 1, 2)
    assert parsed.book_evidence_gate.incomplete_days == [date(2026, 1, 1)]
    assert parsed.book_evidence_gate.required_streams == (
        "TRADE",
        "DEPTH",
        "MARK_PRICE",
        "SPOT_TRADE",
    )


async def test_partition_open_api_is_idempotent_and_research_only(db: AsyncSession) -> None:
    await _seed_shadow_experiment(db, opened=False)

    first = await research.open_experiment_partition(
        "experiment",
        "PROSPECTIVE_SHADOW",
        db=db,
        _user={"sub": "operator"},
    )
    second = await research.open_experiment_partition(
        "experiment",
        "PROSPECTIVE_SHADOW",
        db=db,
        _user={"sub": "operator"},
    )

    assert first.id == second.id
    assert first.opened_at == second.opened_at
    assert first.role == "PROSPECTIVE_SHADOW"
    assert first.safety.order_submission_allowed is False
    assert first.safety.execution_authorization == "none"

    signal = await research.record_shadow_signal(
        "experiment",
        RecordShadowSignalRequest(
            decision_time=datetime.now(UTC),
            side="BUY",
            horizon_seconds=300,
            probability=0.8,
            threshold=0.7,
            model_sha256="d" * 64,
            feature_vector={"flow": 0.5},
        ),
        db=db,
        _user={"sub": "operator"},
    )
    assert signal.outcome_recorded is False
    assert signal.safety.order_submission_allowed is False


async def test_shadow_signal_api_records_hypothetical_signal_without_order_fields(
    db: AsyncSession,
) -> None:
    await _seed_shadow_experiment(db, opened=True)

    response = await research.record_shadow_signal(
        "experiment",
        RecordShadowSignalRequest(
            decision_time=datetime.now(UTC),
            side="BUY",
            horizon_seconds=300,
            probability=0.8,
            threshold=0.7,
            model_sha256="d" * 64,
            feature_vector={"flow": 0.5},
        ),
        db=db,
        _user={"sub": "operator"},
    )

    assert response.experiment_id == "experiment"
    assert response.would_enter is True
    assert response.outcome_recorded is False
    assert response.expected_net_bps is None
    assert response.safety.order_submission_allowed is False
    assert not hasattr(response, "order_id")


async def test_frozen_top_p_shadow_signal_api_scores_artifact_and_records_entry_idempotently(
    db: AsyncSession,
) -> None:
    await _seed_shadow_experiment(db, opened=True)
    artifact = _frozen_top_p_artifact()
    decision_time = datetime.now(UTC)

    first = await research.record_frozen_top_p_shadow_signal(
        "experiment",
        RecordFrozenTopPShadowSignalRequest(
            decision_time=decision_time,
            side="BUY",
            policy_artifact=artifact,
            feature_vector=_top_p_entry_feature_vector(),
        ),
        response=Response(),
        db=db,
        _user={"sub": "operator"},
    )
    retry_response = Response()
    second = await research.record_frozen_top_p_shadow_signal(
        "experiment",
        RecordFrozenTopPShadowSignalRequest(
            decision_time=decision_time,
            side="BUY",
            policy_artifact=artifact,
            feature_vector=_top_p_entry_feature_vector(),
        ),
        response=retry_response,
        db=db,
        _user={"sub": "operator"},
    )

    assert first.recorded is True
    assert first.skipped_existing is False
    assert first.signal is not None
    assert first.would_enter is True
    assert first.signal.model_sha256 == artifact["model_sha256"]
    assert first.order_submission_allowed is False
    assert first.execution_authorization == "none"
    assert second.recorded is False
    assert second.skipped_existing is True
    assert second.signal is not None
    assert first.signal.id == second.signal.id
    assert retry_response.status_code == 200


async def test_frozen_top_p_shadow_signal_api_does_not_record_non_entry_by_default(
    db: AsyncSession,
) -> None:
    await _seed_shadow_experiment(db, opened=True)
    api_response = Response()

    response = await research.record_frozen_top_p_shadow_signal(
        "experiment",
        RecordFrozenTopPShadowSignalRequest(
            decision_time=datetime.now(UTC),
            side="BUY",
            policy_artifact=_frozen_top_p_artifact(),
            feature_vector=_top_p_non_entry_feature_vector(),
        ),
        response=api_response,
        db=db,
        _user={"sub": "operator"},
    )

    signals = (await db.execute(select(ResearchShadowSignal))).scalars().all()
    assert response.recorded is False
    assert response.skipped_existing is False
    assert response.signal is None
    assert response.would_enter is False
    assert response.safety.order_submission_allowed is False
    assert len(signals) == 0
    assert api_response.status_code == 200


async def test_shadow_outcome_api_records_once_and_rejects_overwrite(db: AsyncSession) -> None:
    await _seed_shadow_experiment(db, opened=True)
    now = datetime.now(UTC)
    signal = await research.record_shadow_signal(
        "experiment",
        RecordShadowSignalRequest(
            decision_time=now - timedelta(seconds=121),
            side="SELL",
            horizon_seconds=120,
            probability=0.8,
            threshold=0.7,
            model_sha256="d" * 64,
            feature_vector={"flow": -0.5},
        ),
        db=db,
        _user={"sub": "operator"},
    )

    outcome = await research.record_shadow_outcome(
        signal.id,
        RecordShadowOutcomeRequest(
            expected_net_bps=1.2,
            stress_net_bps=0.4,
            label_sha256="e" * 64,
        ),
        db=db,
        _user={"sub": "operator"},
    )

    assert outcome.outcome_recorded is True
    assert outcome.expected_net_bps == 1.2
    assert outcome.stress_net_bps == 0.4
    assert outcome.label_sha256 == "e" * 64
    with pytest.raises(research.HTTPException) as error:
        await research.record_shadow_outcome(
            signal.id,
            RecordShadowOutcomeRequest(
                expected_net_bps=2.0,
                stress_net_bps=1.0,
                label_sha256="e" * 64,
            ),
            db=db,
            _user={"sub": "operator"},
    )
    assert error.value.status_code == 409


async def test_shadow_outcome_api_rejects_immature_signal_horizon(
    db: AsyncSession,
) -> None:
    await _seed_shadow_experiment(db, opened=True)
    signal = await research.record_shadow_signal(
        "experiment",
        RecordShadowSignalRequest(
            decision_time=datetime.now(UTC),
            side="SELL",
            horizon_seconds=120,
            probability=0.8,
            threshold=0.7,
            model_sha256="d" * 64,
            feature_vector={"flow": -0.5},
        ),
        db=db,
        _user={"sub": "operator"},
    )

    with pytest.raises(research.HTTPException) as error:
        await research.record_shadow_outcome(
            signal.id,
            RecordShadowOutcomeRequest(
                expected_net_bps=1.2,
                stress_net_bps=0.4,
                label_sha256="e" * 64,
            ),
            db=db,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "horizon matures" in error.value.detail


async def test_testnet_eligibility_remains_metadata_without_explicit_release(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256="e" * 64,
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    await db.flush()
    await _append_shadow_ledger_events(db)

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.eligible is False
    assert response.book_evidence_eligible is True
    assert response.book_evidence_contiguous_days == 60
    assert response.prospective_shadow_days == 20
    assert response.prospective_shadow_outcome_days == 20
    assert response.prospective_shadow_signal_count == 20
    assert response.prospective_shadow_outcome_signal_count == 20
    assert response.prospective_shadow_expected_mean_bps == 1.2
    assert response.prospective_shadow_stress_mean_bps == 0.4
    assert response.prospective_shadow_positive is True
    assert response.approved_statistical_gate_verified is True
    assert response.explicit_testnet_release is False
    assert response.release_request_required is True
    assert response.order_submission_allowed is False
    assert response.execution_authorization == "none"
    assert response.reasons == ["explicit_testnet_release_is_missing"]


async def test_testnet_eligibility_rejects_shadow_rows_without_event_ledger(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256=f"{day_offset:064x}"[-64:],
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    await db.flush()

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.eligible is False
    assert response.prospective_shadow_positive is True
    assert response.shadow_ledger_verified is False
    assert "shadow_ledger_unverified" in response.reasons
    assert "shadow_signal_1_event_missing" in response.reasons


async def test_testnet_eligibility_rejects_legacy_approved_status_without_gate_event(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="legacy-approved",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256=f"{day_offset:064x}"[-64:],
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    await db.flush()

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.eligible is False
    assert response.approved_statistical_gate_verified is False
    assert "approved_statistical_gate_evidence_missing" in response.reasons
    assert response.order_submission_allowed is False
    assert response.execution_authorization == "none"


async def test_testnet_eligibility_rejects_approved_gate_event_without_research_only_boundary(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="legacy-gate-event",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    unsafe_event = _approved_decision_event()
    event_payload = json.loads(unsafe_event.payload_json)
    event_payload["evidence"].pop("research_only", None)
    unsafe_event.payload_json = json.dumps(event_payload, sort_keys=True)
    db.add(unsafe_event)
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256=f"{day_offset:064x}"[-64:],
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    await db.flush()

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.eligible is False
    assert response.approved_statistical_gate_verified is False
    assert response.prospective_shadow_positive is True
    assert "approved_statistical_gate_evidence_missing" in response.reasons
    assert response.order_submission_allowed is False
    assert response.execution_authorization == "none"


async def test_testnet_eligibility_requires_positive_shadow_outcomes(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="SELL",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256="e" * 64,
            )
        )
    await db.flush()

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.eligible is False
    assert response.prospective_shadow_days == 20
    assert response.prospective_shadow_outcome_days == 0
    assert response.prospective_shadow_signal_count == 20
    assert response.prospective_shadow_outcome_signal_count == 0
    assert response.prospective_shadow_positive is False
    assert response.approved_statistical_gate_verified is True
    assert response.prospective_shadow_expected_mean_bps is None
    assert "prospective_shadow_outcomes_incomplete" in response.reasons
    assert "prospective_shadow_block_not_positive" in response.reasons
    assert response.order_submission_allowed is False


async def test_testnet_eligibility_ignores_unsafe_or_immature_shadow_outcomes(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    unsafe_outcomes = [
        json.dumps({"expected_net_bps": 1.2, "stress_net_bps": 0.4}),
        _safe_shadow_outcome_json(extra={"order_id": "manual-order-would-be-execution"}),
        _safe_shadow_outcome_json(recorded_at="2026-03-02T00:01:00+00:00"),
    ]
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256=f"{day_offset:064x}"[-64:],
                outcome_json=unsafe_outcomes[day_offset % len(unsafe_outcomes)],
            )
        )
    await db.flush()

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.eligible is False
    assert response.prospective_shadow_days == 20
    assert response.prospective_shadow_outcome_days == 0
    assert response.prospective_shadow_signal_count == 20
    assert response.prospective_shadow_outcome_signal_count == 0
    assert response.prospective_shadow_expected_mean_bps is None
    assert response.prospective_shadow_stress_mean_bps is None
    assert response.prospective_shadow_positive is False
    assert "prospective_shadow_outcomes_incomplete" in response.reasons
    assert "prospective_shadow_block_not_positive" in response.reasons
    assert response.order_submission_allowed is False


async def test_research_testnet_release_requires_all_prior_gates(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    await db.flush()

    with pytest.raises(research.HTTPException) as error:
        await research.record_testnet_release(
            "experiment",
            RecordTestnetReleaseRequest(
                confirmation_phrase="REQUEST RESEARCH TESTNET RELEASE",
                reasons=["manual_release_after_review"],
            ),
            response=Response(),
            db=db,
            user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "experiment_status_is_not_approved" in error.value.detail["reasons"]
    assert "prospective_shadow_has_fewer_than_20_days" in error.value.detail["reasons"]
    assert error.value.detail["order_submission_allowed"] is False
    assert error.value.detail["execution_authorization"] == "none"


async def test_research_testnet_release_requires_eligible_book_gate_even_with_60_day_streak(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    payload = _eligible_evidence_payload()
    payload["book_evidence_gate"]["eligible"] = False
    payload["book_evidence_gate"]["reasons"] = ["spot_trade_missing_from_evidence_gate"]
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            decision_reasons_json=json.dumps(["all_gates_passed"]),
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256=f"{day_offset:064x}"[-64:],
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    await db.flush()
    await _append_shadow_ledger_events(db)

    with pytest.raises(research.HTTPException) as error:
        await research.record_testnet_release(
            "experiment",
            RecordTestnetReleaseRequest(
                confirmation_phrase="REQUEST RESEARCH TESTNET RELEASE",
                reasons=["manual_release_after_review"],
            ),
            response=Response(),
            db=db,
            user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "book_evidence_gate_not_eligible" in error.value.detail["reasons"]
    assert "book_evidence_has_fewer_than_60_complete_days" not in error.value.detail["reasons"]


async def test_research_testnet_release_rejects_legacy_approved_without_gate_event(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="legacy-approved",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            decision_reasons_json=json.dumps(["legacy_status_only"]),
        )
    )
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256=f"{day_offset:064x}"[-64:],
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    await db.flush()

    with pytest.raises(research.HTTPException) as error:
        await research.record_testnet_release(
            "experiment",
            RecordTestnetReleaseRequest(
                confirmation_phrase="REQUEST RESEARCH TESTNET RELEASE",
                reasons=["manual_release_after_review"],
            ),
            response=Response(),
            db=db,
            user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "approved_statistical_gate_evidence_missing" in error.value.detail["reasons"]
    assert error.value.detail["order_submission_allowed"] is False
    assert error.value.detail["execution_authorization"] == "none"


async def test_research_testnet_release_makes_checklist_eligible_without_execution(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            decision_reasons_json=json.dumps(["all_gates_passed"]),
            decided_at=datetime(2026, 3, 3, tzinfo=UTC),
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256=f"{day_offset:064x}"[-64:],
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    await db.flush()
    await _append_shadow_ledger_events(db)

    release_response = Response()
    release = await research.record_testnet_release(
        "experiment",
        RecordTestnetReleaseRequest(
            confirmation_phrase="REQUEST RESEARCH TESTNET RELEASE",
            reasons=["manual_release_after_review"],
        ),
        response=release_response,
        db=db,
        user={"sub": "operator"},
    )
    second_response = Response()
    same_release = await research.record_testnet_release(
        "experiment",
        RecordTestnetReleaseRequest(
            confirmation_phrase="REQUEST RESEARCH TESTNET RELEASE",
            reasons=["manual_release_after_review"],
        ),
        response=second_response,
        db=db,
        user={"sub": "operator"},
    )
    eligibility = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert second_response.status_code == 200
    assert same_release.release_sha256 == release.release_sha256
    assert release.explicit_testnet_release is True
    assert release.release_request_required is False
    assert release.order_submission_allowed is False
    assert release.execution_authorization == "none"
    assert release.evidence_snapshot["book_evidence_contiguous_days"] == 60
    assert release.evidence_snapshot["book_evidence_eligible"] is True
    assert release.evidence_snapshot["approved_statistical_gate_verified"] is True
    assert eligibility.eligible is True
    assert eligibility.book_evidence_eligible is True
    assert eligibility.approved_statistical_gate_verified is True
    assert eligibility.explicit_testnet_release is True
    assert eligibility.release_request_required is False
    assert eligibility.order_submission_allowed is False
    assert eligibility.execution_authorization == "none"


async def test_testnet_eligibility_rejects_missing_outcome_for_second_signal_on_same_day(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    for day_offset in range(20):
        db.add(
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=day_offset),
                recorded_at=start + timedelta(days=day_offset, seconds=1),
                side="BUY",
                horizon_seconds=120,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256="e" * 64,
                outcome_json=_safe_shadow_outcome_json(),
            )
        )
    db.add(
        ResearchShadowSignal(
            experiment_id="experiment",
            decision_time=start + timedelta(minutes=5),
            recorded_at=start + timedelta(minutes=5, seconds=1),
            side="SELL",
            horizon_seconds=120,
            probability=0.8,
            threshold=0.7,
            would_enter=True,
            model_sha256="d" * 64,
            feature_vector_sha256="f" * 64,
        )
    )
    await db.flush()

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.prospective_shadow_days == 20
    assert response.prospective_shadow_outcome_days == 20
    assert response.prospective_shadow_signal_count == 21
    assert response.prospective_shadow_outcome_signal_count == 20
    assert response.prospective_shadow_positive is False
    assert response.approved_statistical_gate_verified is True
    assert "prospective_shadow_signal_outcomes_incomplete" in response.reasons
    assert "prospective_shadow_block_not_positive" in response.reasons
    assert response.order_submission_allowed is False


async def test_experiment_report_exposes_evidence_and_shadow_metrics(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            decision_reasons_json=json.dumps(["passed_shadow"]),
            decided_at=datetime(2026, 3, 3, tzinfo=UTC),
        )
    )
    db.add(_approved_decision_event())
    start = datetime(2026, 3, 2, tzinfo=UTC)
    db.add_all(
        [
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start,
                recorded_at=start + timedelta(seconds=1),
                side="BUY",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256="e" * 64,
                outcome_json=_safe_shadow_outcome_json(
                    expected_net_bps=2.0,
                    stress_net_bps=1.0,
                    label_sha256="e" * 64,
                ),
            ),
            ResearchShadowSignal(
                experiment_id="experiment",
                decision_time=start + timedelta(days=1),
                recorded_at=start + timedelta(days=1, seconds=1),
                side="SELL",
                horizon_seconds=300,
                probability=0.8,
                threshold=0.7,
                would_enter=True,
                model_sha256="d" * 64,
                feature_vector_sha256="f" * 64,
                outcome_json=_safe_shadow_outcome_json(
                    expected_net_bps=4.0,
                    stress_net_bps=3.0,
                    label_sha256="f" * 64,
                ),
            ),
            ResearchHypothesisAttempt(
                experiment_id="experiment",
                kind="FEATURE_SET",
                fingerprint_sha256="1" * 64,
                definition_json=json.dumps({"feature_set": "flow_price_book_aux_session"}),
                status="REGISTERED",
            ),
            ResearchHypothesisAttempt(
                experiment_id="experiment",
                kind="TOP_P_TAIL",
                fingerprint_sha256="2" * 64,
                definition_json=json.dumps({"tail_fraction": 0.05}),
                status="EVALUATED",
            ),
        ]
    )
    await db.flush()

    response = await research.get_experiment_report(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response["decision_reasons"] == ["passed_shadow"]
    assert response["artifact_sha256"] is not None
    assert len(response["artifact_sha256"]) == 64
    assert response["metrics"]["book_evidence"]["longest_complete_streak_days"] == 60
    assert response["metrics"]["shadow"]["signal_count"] == 2
    assert response["metrics"]["shadow"]["outcome_signal_count"] == 2
    assert response["metrics"]["shadow"]["expected_mean_bps"] == 3.0
    assert response["metrics"]["shadow"]["stress_mean_bps"] == 2.0
    assert response["metrics"]["shadow"]["complete"] is True
    assert response["metrics"]["shadow"]["positive"] is True
    assert response["metrics"]["hypothesis_ledger"]["attempt_count"] == 2
    assert response["metrics"]["hypothesis_ledger"]["attempts"][0]["kind"] == "FEATURE_SET"
    assert (
        response["metrics"]["hypothesis_ledger"]["attempts"][0]["definition"]["feature_set"]
        == "flow_price_book_aux_session"
    )
    assert response["metrics"]["experiment_event_chain"]["verified"] is True
    assert response["metrics"]["experiment_event_chain"]["event_count"] == 1
    assert len(response["metrics"]["experiment_event_chain"]["latest_event_sha256"]) == 64
    assert (
        response["metrics"]["testnet_boundary"]["approved_statistical_gate_verified"]
        is True
    )
    assert response["metrics"]["testnet_boundary"]["order_submission_allowed"] is False
    assert response["safety"]["execution_authorization"] == "none"


async def test_experiment_report_flags_tampered_event_chain(
    tmp_path: Path,
    monkeypatch,
    db: AsyncSession,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text(json.dumps(_eligible_evidence_payload()), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="APPROVED",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            decision_reasons_json=json.dumps(["passed_shadow"]),
            decided_at=datetime(2026, 3, 3, tzinfo=UTC),
        )
    )
    event = _approved_decision_event()
    event.payload_json = event.payload_json.replace("APPROVED", "REJECTED", 1)
    db.add(event)
    await db.flush()

    response = await research.get_experiment_report(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response["metrics"]["experiment_event_chain"]["verified"] is False
    assert response["metrics"]["experiment_event_chain"]["reasons"] == [
        "event_1_hash_mismatch"
    ]
    assert (
        response["metrics"]["testnet_boundary"]["approved_statistical_gate_verified"]
        is False
    )


async def test_record_experiment_decision_is_terminal_and_research_only(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()

    response = await research.record_experiment_decision(
        "experiment",
        RecordExperimentDecisionRequest(
            status="REJECTED",
            reasons=["stress_mean_bps_not_positive"],
        ),
        db=db,
        _user={"sub": "operator"},
    )

    assert response["status"] == "REJECTED"
    assert response["safety"]["order_submission_allowed"] is False
    assert response["safety"]["execution_authorization"] == "none"
    report = await research.get_experiment_report(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )
    assert report["decision_reasons"] == ["stress_mean_bps_not_positive"]
    with pytest.raises(research.HTTPException) as error:
        await research.record_experiment_decision(
            "experiment",
            RecordExperimentDecisionRequest(
                status="APPROVED",
                reasons=["cannot_overwrite_terminal_decision"],
            ),
            db=db,
            _user={"sub": "operator"},
        )
    assert error.value.status_code == 409


async def test_approved_experiment_decision_requires_statistical_gate_artifact(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()

    with pytest.raises(research.HTTPException) as error:
        await research.record_experiment_decision(
            "experiment",
            RecordExperimentDecisionRequest(
                status="APPROVED",
                reasons=["all_gates_passed"],
            ),
            db=db,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert error.value.detail["reasons"] == [
        "statistical_gate_artifact_required_for_approval"
    ]
    assert error.value.detail["order_submission_allowed"] is False
    assert error.value.detail["execution_authorization"] == "none"


async def test_approved_experiment_decision_records_statistical_gate_hash(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()
    statistical_gate = _approved_statistical_gate_payload()

    response = await research.record_experiment_decision(
        "experiment",
        RecordExperimentDecisionRequest(
            status="APPROVED",
            reasons=["all_statistical_gates_passed"],
            statistical_gate=statistical_gate,
        ),
        db=db,
        _user={"sub": "operator"},
    )

    assert response["status"] == "APPROVED"
    assert response["safety"]["order_submission_allowed"] is False
    assert response["safety"]["execution_authorization"] == "none"
    result = await db.execute(
        select(ResearchExperimentEvent)
        .where(
            ResearchExperimentEvent.experiment_id == "experiment",
            ResearchExperimentEvent.kind == "DECISION_RECORDED",
        )
        .order_by(ResearchExperimentEvent.id.desc())
    )
    event = result.scalars().first()
    assert event is not None
    payload = json.loads(event.payload_json)
    assert payload["status"] == "APPROVED"
    assert payload["evidence"]["statistical_gate_decision"] == "APPROVED"
    assert payload["evidence"]["attempted_hypotheses"] == 44
    assert payload["evidence"]["approved_strategy_count"] == 1
    assert payload["evidence"]["statistical_gate_sha256"] == statistical_gate["artifact_sha256"]
    assert payload["evidence"]["research_only"] is True
    assert payload["evidence"]["order_submission_allowed"] is False
    assert payload["evidence"]["execution_authorization"] == "none"


async def test_approved_experiment_decision_rejects_tampered_statistical_gate_hash(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()
    statistical_gate = _approved_statistical_gate_payload()
    statistical_gate["artifact_sha256"] = "0" * 64

    with pytest.raises(research.HTTPException) as error:
        await research.record_experiment_decision(
            "experiment",
            RecordExperimentDecisionRequest(
                status="APPROVED",
                reasons=["all_statistical_gates_passed"],
                statistical_gate=statistical_gate,
            ),
            db=db,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "statistical_gate_artifact_sha256_mismatch" in error.value.detail["reasons"]
    assert error.value.detail["order_submission_allowed"] is False
    assert error.value.detail["execution_authorization"] == "none"


async def test_approved_experiment_decision_rejects_failed_statistical_gate(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()
    statistical_gate = _approved_statistical_gate_payload()
    statistical_gate["results"][0]["conditions"]["positive_stress_mean"] = False

    with pytest.raises(research.HTTPException) as error:
        await research.record_experiment_decision(
            "experiment",
            RecordExperimentDecisionRequest(
                status="APPROVED",
                reasons=["all_statistical_gates_passed"],
                statistical_gate=statistical_gate,
            ),
            db=db,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "approved_result_0_positive_stress_mean_failed" in error.value.detail["reasons"]
    assert error.value.detail["order_submission_allowed"] is False


async def test_approved_experiment_decision_requires_all_statistical_gate_conditions(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()
    statistical_gate = _approved_statistical_gate_payload()
    statistical_gate["results"][0]["conditions"].pop("three_temporal_folds")

    with pytest.raises(research.HTTPException) as error:
        await research.record_experiment_decision(
            "experiment",
            RecordExperimentDecisionRequest(
                status="APPROVED",
                reasons=["all_statistical_gates_passed"],
                statistical_gate=statistical_gate,
            ),
            db=db,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "approved_result_0_three_temporal_folds_condition_missing" in error.value.detail[
        "reasons"
    ]
    assert error.value.detail["order_submission_allowed"] is False


async def test_approved_experiment_decision_rejects_inconsistent_decision_counts(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()
    statistical_gate = _approved_statistical_gate_payload()
    statistical_gate["decision_counts"] = {"APPROVED": 2}

    with pytest.raises(research.HTTPException) as error:
        await research.record_experiment_decision(
            "experiment",
            RecordExperimentDecisionRequest(
                status="APPROVED",
                reasons=["all_statistical_gates_passed"],
                statistical_gate=statistical_gate,
            ),
            db=db,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "statistical_gate_decision_counts_do_not_match_results" in error.value.detail[
        "reasons"
    ]
    assert error.value.detail["order_submission_allowed"] is False


async def test_approved_experiment_decision_rejects_inconsistent_adjusted_alpha(
    db: AsyncSession,
) -> None:
    db.add(
        ResearchExperiment(
            id="experiment",
            name="candidate",
            status="FROZEN",
            code_revision="a" * 40,
            protocol_sha256="b" * 64,
            product_json="{}",
            cost_profile_json="{}",
            approval_gate_json="{}",
            experiment_sha256="9" * 64,
            frozen_at=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )
    await db.flush()
    statistical_gate = _approved_statistical_gate_payload()
    statistical_gate["results"][0]["adjusted_one_sided_alpha"] = 0.05

    with pytest.raises(research.HTTPException) as error:
        await research.record_experiment_decision(
            "experiment",
            RecordExperimentDecisionRequest(
                status="APPROVED",
                reasons=["all_statistical_gates_passed"],
                statistical_gate=statistical_gate,
            ),
            db=db,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    assert "approved_result_0_adjusted_alpha_inconsistent" in error.value.detail["reasons"]
    assert error.value.detail["order_submission_allowed"] is False


def _eligible_evidence_payload() -> dict[str, object]:
    safety = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }
    return {
        "artifact_available": True,
        "audited_start_date": "2026-01-01",
        "audited_end_date": "2026-03-01",
        "audited_days": 60,
        "latest_daily_status": "VALID",
        "latest_daily_manifest_sha256": "1" * 64,
        "book_evidence_gate": {
            "eligible": True,
            "required_complete_days": 60,
            "required_streams": ["TRADE", "DEPTH", "MARK_PRICE", "SPOT_TRADE"],
            "audited_days": 60,
            "complete_days": 60,
            "longest_complete_streak_days": 60,
            "streak_start": "2026-01-01",
            "streak_end": "2026-03-01",
            "incomplete_days": [],
            "manifest_sha256": "2" * 64,
            "reasons": [],
            "safety": safety,
        },
        "status_reasons": [],
        "safety": safety,
        "generated_at": "2026-03-02T00:00:00+00:00",
    }


def _frozen_top_p_artifact() -> dict[str, object]:
    return freeze_top_p_policy(
        _top_p_model_frame(),
        horizon_seconds=120,
        feature_set="flow",
        tail_fraction=0.10,
        calibration_date="2026-01-06",
        dataset_manifest_sha256="d" * 64,
        embargo_seconds=300,
    ).to_dict()


def _top_p_entry_feature_vector() -> dict[str, float]:
    return {
        "flow_imbalance_1s": 2.0,
        "trade_count_1s": 12.0,
        "quote_volume_1s": 140.0,
        "mean_interarrival_ms_1s": 20.0,
    }


def _top_p_non_entry_feature_vector() -> dict[str, float]:
    return {
        "flow_imbalance_1s": -3.0,
        "trade_count_1s": 13.0,
        "quote_volume_1s": 160.0,
        "mean_interarrival_ms_1s": 20.0,
    }


def _top_p_model_frame() -> pd.DataFrame:
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


def _safe_shadow_outcome_json(
    *,
    expected_net_bps: float = 1.2,
    stress_net_bps: float = 0.4,
    label_sha256: str = "e" * 64,
    recorded_at: str = "2026-04-30T00:00:00+00:00",
    extra: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "expected_net_bps": expected_net_bps,
        "stress_net_bps": stress_net_bps,
        "label_sha256": label_sha256,
        "recorded_at": recorded_at,
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }
    if extra is not None:
        payload.update(extra)
    return json.dumps(
        payload,
        sort_keys=True,
    )


def _approved_statistical_gate_payload() -> dict[str, object]:
    payload = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "attempted_hypotheses": 44,
        "top_p_monotonic": True,
        "top_p_monotonic_reasons": [],
        "prospective_shadow_positive": True,
        "prospective_shadow_reasons": [],
        "decision_counts": {"APPROVED": 1},
        "results": [
            {
                "strategy": "policy=trail_12bps-top=5%",
                "decision": "APPROVED",
                "trade_count": 240,
                "distinct_days": 24,
                "expected_mean_bps": 2.4,
                "stress_mean_bps": 0.8,
                "adjusted_one_sided_alpha": 0.0011363636363636363,
                "adjusted_lower_confidence_bound_bps": 0.2,
                "probability_of_backtest_overfitting": 0.1,
                "conditions": {
                    "three_temporal_folds": True,
                    "minimum_200_trades": True,
                    "minimum_20_days": True,
                    "positive_expected_mean": True,
                    "adjusted_lower_bound_positive": True,
                    "positive_stress_mean": True,
                    "top_p_monotonic": True,
                    "pbo_at_most_20_percent": True,
                    "prospective_positive": True,
                },
                "reasons": [],
            }
        ],
    }
    return _with_statistical_gate_hash(payload)


def _with_statistical_gate_hash(payload: dict[str, object]) -> dict[str, object]:
    canonical_payload = {
        key: value for key, value in payload.items() if key != "artifact_sha256"
    }
    return {
        "artifact_sha256": research._report_sha256(canonical_payload),
        **canonical_payload,
    }


def _approved_decision_event(experiment_id: str = "experiment") -> ResearchExperimentEvent:
    payload_json = json.dumps(
        {
            "status": "APPROVED",
            "reasons": ["all_statistical_gates_passed"],
            "evidence": {
                "statistical_gate_sha256": "a" * 64,
                "statistical_gate_decision": "APPROVED",
                "attempted_hypotheses": 44,
                "approved_strategy_count": 1,
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
            },
        },
        sort_keys=True,
    )
    occurred_at = datetime(2026, 3, 2, tzinfo=UTC)
    return ResearchExperimentEvent(
        experiment_id=experiment_id,
        kind="DECISION_RECORDED",
        payload_json=payload_json,
        occurred_at=occurred_at,
        previous_event_sha256=None,
        event_sha256=build_research_event_hash(
            experiment_id=experiment_id,
            kind="DECISION_RECORDED",
            payload_json=payload_json,
            occurred_at=occurred_at,
            previous_event_sha256=None,
        ),
    )


async def _seed_shadow_experiment(db: AsyncSession, *, opened: bool) -> None:
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
            start_at=now - timedelta(hours=1),
            end_at=now + timedelta(days=20),
            manifest_sha256="c" * 64,
            opened_at=now if opened else None,
        )
    )
    await db.flush()


async def _append_shadow_ledger_events(
    db: AsyncSession,
    experiment_id: str = "experiment",
) -> None:
    signals = await research_experiment_repository.list_shadow_signals(db, experiment_id)
    event_rows: list[tuple[datetime, str, dict[str, object]]] = []
    for signal in signals:
        event_rows.append(
            (
                _as_utc(signal.recorded_at),
                "SHADOW_SIGNAL_RECORDED",
                shadow_signal_event_payload(signal),
            )
        )
        if signal.outcome_json is None:
            continue
        outcome_payload = json.loads(signal.outcome_json)
        event_rows.append(
            (
                _parse_test_datetime(str(outcome_payload["recorded_at"])),
                "SHADOW_OUTCOME_RECORDED",
                shadow_outcome_event_payload(signal, outcome_payload),
            )
        )

    for occurred_at, kind, payload in sorted(event_rows, key=lambda row: row[0]):
        await research_experiment_repository.append_event(
            db,
            ResearchExperimentEvent(
                experiment_id=experiment_id,
                kind=kind,
                payload_json=json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                occurred_at=occurred_at,
            ),
        )


def _parse_test_datetime(value: str) -> datetime:
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

"""Research-only evidence-gate API never grants execution access."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import Response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.v1 import research
from app.models.base import Base
from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchHypothesisAttempt,
    ResearchShadowSignal,
)
from app.schemas.research_experiment import (
    RecordExperimentDecisionRequest,
    RecordShadowOutcomeRequest,
    RecordShadowSignalRequest,
    RecordTestnetReleaseRequest,
)
from app.services.market.microstructure_wal_audit import build_evidence_gate_status


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
    assert response.book_evidence_gate.streak_start is None
    assert response.book_evidence_gate.streak_end is None
    assert response.status_reasons == ["no_complete_days_yet"]


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

    parsed = research.EvidenceGateStatusResponse.model_validate(payload)

    assert parsed.audited_start_date == date(2026, 1, 1)
    assert parsed.audited_end_date == date(2026, 1, 2)
    assert parsed.book_evidence_gate.incomplete_days == [date(2026, 1, 1)]


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


async def test_shadow_outcome_api_records_once_and_rejects_overwrite(db: AsyncSession) -> None:
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
                outcome_json=json.dumps({"expected_net_bps": 1.2, "stress_net_bps": 0.4}),
            )
        )
    await db.flush()

    response = await research.get_testnet_eligibility(
        "experiment",
        db=db,
        _user={"sub": "operator"},
    )

    assert response.eligible is False
    assert response.book_evidence_contiguous_days == 60
    assert response.prospective_shadow_days == 20
    assert response.prospective_shadow_outcome_days == 20
    assert response.prospective_shadow_signal_count == 20
    assert response.prospective_shadow_outcome_signal_count == 20
    assert response.prospective_shadow_expected_mean_bps == 1.2
    assert response.prospective_shadow_stress_mean_bps == 0.4
    assert response.prospective_shadow_positive is True
    assert response.explicit_testnet_release is False
    assert response.release_request_required is True
    assert response.order_submission_allowed is False
    assert response.execution_authorization == "none"
    assert response.reasons == ["explicit_testnet_release_is_missing"]


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
    assert response.prospective_shadow_expected_mean_bps is None
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
                outcome_json=json.dumps({"expected_net_bps": 1.2, "stress_net_bps": 0.4}),
            )
        )
    await db.flush()

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
    assert eligibility.eligible is True
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
                outcome_json=json.dumps({"expected_net_bps": 1.2, "stress_net_bps": 0.4}),
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
                outcome_json=json.dumps({"expected_net_bps": 2.0, "stress_net_bps": 1.0}),
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
                outcome_json=json.dumps({"expected_net_bps": 4.0, "stress_net_bps": 3.0}),
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
    assert response["metrics"]["testnet_boundary"]["order_submission_allowed"] is False
    assert response["safety"]["execution_authorization"] == "none"


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
            start_at=now,
            end_at=now + timedelta(days=20),
            manifest_sha256="c" * 64,
            opened_at=now if opened else None,
        )
    )
    await db.flush()

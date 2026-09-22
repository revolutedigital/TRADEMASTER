"""Research approval alone never activates or authorizes Testnet."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.research_experiment import ResearchExperiment
from app.repositories.research_experiment_repo import ResearchEventChainStatus
from app.services.research.testnet_release_gate import (
    evaluate_testnet_eligibility,
    research_testnet_release_readiness,
)


def experiment(status: str) -> ResearchExperiment:
    return ResearchExperiment(
        id="experiment",
        name="candidate",
        status=status,
        code_revision="a" * 40,
        protocol_sha256="b" * 64,
        product_json="{}",
        cost_profile_json="{}",
        approval_gate_json="{}",
    )


def test_approved_research_still_needs_separate_testnet_release() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=True,
        book_evidence_contiguous_days=60,
        prospective_shadow_days=30,
        prospective_shadow_outcome_days=30,
        prospective_shadow_signal_count=30,
        prospective_shadow_outcome_signal_count=30,
        prospective_shadow_positive=True,
        approved_statistical_gate_verified=True,
        unresolved_failures=0,
        explicit_testnet_release=False,
    )
    assert result.eligible is False
    assert result.order_submission_allowed is False
    assert result.execution_authorization == "none"


def test_eligibility_is_metadata_not_execution_authorization() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=True,
        book_evidence_contiguous_days=60,
        prospective_shadow_days=30,
        prospective_shadow_outcome_days=30,
        prospective_shadow_signal_count=30,
        prospective_shadow_outcome_signal_count=30,
        prospective_shadow_positive=True,
        approved_statistical_gate_verified=True,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )
    assert result.eligible is True
    assert result.order_submission_allowed is False
    assert result.execution_authorization == "none"


def test_testnet_requires_sixty_complete_book_evidence_days() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=True,
        book_evidence_contiguous_days=59,
        prospective_shadow_days=20,
        prospective_shadow_outcome_days=20,
        prospective_shadow_signal_count=20,
        prospective_shadow_outcome_signal_count=20,
        prospective_shadow_positive=True,
        approved_statistical_gate_verified=True,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )
    assert result.eligible is False
    assert "book_evidence_has_fewer_than_60_complete_days" in result.reasons
    assert result.order_submission_allowed is False


def test_testnet_requires_book_evidence_gate_to_be_eligible() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=False,
        book_evidence_contiguous_days=60,
        prospective_shadow_days=20,
        prospective_shadow_outcome_days=20,
        prospective_shadow_signal_count=20,
        prospective_shadow_outcome_signal_count=20,
        prospective_shadow_positive=True,
        approved_statistical_gate_verified=True,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )
    assert result.eligible is False
    assert "book_evidence_gate_not_eligible" in result.reasons
    assert "book_evidence_has_fewer_than_60_complete_days" not in result.reasons


def test_shadow_window_cannot_exceed_preregistered_maximum() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=True,
        book_evidence_contiguous_days=60,
        prospective_shadow_days=31,
        prospective_shadow_outcome_days=31,
        prospective_shadow_signal_count=31,
        prospective_shadow_outcome_signal_count=31,
        prospective_shadow_positive=True,
        approved_statistical_gate_verified=True,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )
    assert result.eligible is False
    assert "prospective_shadow_exceeds_30_days" in result.reasons


def test_testnet_requires_positive_prospective_shadow_block() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=True,
        book_evidence_contiguous_days=60,
        prospective_shadow_days=20,
        prospective_shadow_outcome_days=20,
        prospective_shadow_signal_count=20,
        prospective_shadow_outcome_signal_count=20,
        prospective_shadow_positive=False,
        approved_statistical_gate_verified=True,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )
    assert result.eligible is False
    assert "prospective_shadow_block_not_positive" in result.reasons


def test_testnet_requires_complete_shadow_outcomes() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=True,
        book_evidence_contiguous_days=60,
        prospective_shadow_days=20,
        prospective_shadow_outcome_days=19,
        prospective_shadow_signal_count=20,
        prospective_shadow_outcome_signal_count=19,
        prospective_shadow_positive=True,
        approved_statistical_gate_verified=True,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )
    assert result.eligible is False
    assert "prospective_shadow_outcomes_incomplete" in result.reasons
    assert "prospective_shadow_signal_outcomes_incomplete" in result.reasons


def test_testnet_requires_approved_statistical_gate_evidence() -> None:
    result = evaluate_testnet_eligibility(
        experiment("APPROVED"),
        book_evidence_eligible=True,
        book_evidence_contiguous_days=60,
        prospective_shadow_days=20,
        prospective_shadow_outcome_days=20,
        prospective_shadow_signal_count=20,
        prospective_shadow_outcome_signal_count=20,
        prospective_shadow_positive=True,
        approved_statistical_gate_verified=False,
        unresolved_failures=0,
        explicit_testnet_release=True,
    )

    assert result.eligible is False
    assert result.approved_statistical_gate_verified is False
    assert "approved_statistical_gate_evidence_missing" in result.reasons


@pytest.mark.asyncio
async def test_runtime_testnet_readiness_requires_a_recorded_release() -> None:
    database = AsyncMock()
    result = MagicMock()
    result.first.return_value = None
    database.execute = AsyncMock(return_value=result)

    readiness = await research_testnet_release_readiness(database)

    assert readiness.ready is False
    assert readiness.reasons == ("research_testnet_release_missing",)


@pytest.mark.asyncio
async def test_runtime_testnet_readiness_accepts_intact_research_only_release() -> None:
    database = AsyncMock()
    release = SimpleNamespace(
        release_sha256="c" * 64,
        evidence_snapshot_json=json.dumps(_eligible_release_snapshot()),
    )
    approved_experiment = experiment("APPROVED")
    approved_experiment.experiment_sha256 = "9" * 64
    result = MagicMock()
    result.first.return_value = SimpleNamespace(tuple=lambda: (release, approved_experiment))
    database.execute = AsyncMock(return_value=result)

    with patch(
        "app.services.research.testnet_release_gate.research_experiment_repository.event_chain_status",
        new=AsyncMock(
            return_value=ResearchEventChainStatus(
                event_count=3,
                verified=True,
                latest_event_sha256="a" * 64,
                reasons=(),
            )
        ),
    ):
        readiness = await research_testnet_release_readiness(database)

    assert readiness.ready is True
    assert readiness.reasons == ()
    assert readiness.release_sha256 == "c" * 64


@pytest.mark.asyncio
async def test_runtime_testnet_readiness_rejects_broken_event_chain() -> None:
    database = AsyncMock()
    release = SimpleNamespace(
        release_sha256="c" * 64,
        evidence_snapshot_json=json.dumps(_eligible_release_snapshot()),
    )
    approved_experiment = experiment("APPROVED")
    approved_experiment.experiment_sha256 = "9" * 64
    result = MagicMock()
    result.first.return_value = SimpleNamespace(tuple=lambda: (release, approved_experiment))
    database.execute = AsyncMock(return_value=result)

    with patch(
        "app.services.research.testnet_release_gate.research_experiment_repository.event_chain_status",
        new=AsyncMock(
            return_value=ResearchEventChainStatus(
                event_count=3,
                verified=False,
                latest_event_sha256="a" * 64,
                reasons=("event_2_hash_mismatch",),
            )
        ),
    ):
        readiness = await research_testnet_release_readiness(database)

    assert readiness.ready is False
    assert "research_event_chain_unverified" in readiness.reasons


def _eligible_release_snapshot() -> dict[str, object]:
    return {
        "experiment_status": "APPROVED",
        "experiment_sha256": "9" * 64,
        "book_evidence_contiguous_days": 60,
        "book_evidence_eligible": True,
        "evidence_artifact_available": True,
        "evidence_manifest_sha256": "2" * 64,
        "prospective_shadow_days": 20,
        "prospective_shadow_outcome_days": 20,
        "prospective_shadow_signal_count": 20,
        "prospective_shadow_outcome_signal_count": 20,
        "prospective_shadow_expected_mean_bps": 1.2,
        "prospective_shadow_stress_mean_bps": 0.4,
        "prospective_shadow_positive": True,
        "approved_statistical_gate_verified": True,
        "unresolved_failures": 0,
        "generated_at": "2026-03-22T12:00:00+00:00",
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }

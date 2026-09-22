"""Research approval alone never activates or authorizes Testnet."""

from app.models.research_experiment import ResearchExperiment
from app.services.research.testnet_release_gate import evaluate_testnet_eligibility


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

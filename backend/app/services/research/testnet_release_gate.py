"""Research-to-Testnet eligibility checks; this module cannot submit orders."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.research_experiment import ResearchExperiment


MIN_BOOK_EVIDENCE_DAYS = 60
MIN_PROSPECTIVE_SHADOW_DAYS = 20
MAX_PROSPECTIVE_SHADOW_DAYS = 30


@dataclass(frozen=True)
class TestnetEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    book_evidence_eligible: bool
    book_evidence_contiguous_days: int
    prospective_shadow_days: int
    prospective_shadow_outcome_days: int
    prospective_shadow_signal_count: int
    prospective_shadow_outcome_signal_count: int
    prospective_shadow_positive: bool
    approved_statistical_gate_verified: bool
    execution_authorization: str = "none"
    order_submission_allowed: bool = False


def evaluate_testnet_eligibility(
    experiment: ResearchExperiment,
    *,
    book_evidence_contiguous_days: int,
    book_evidence_eligible: bool,
    prospective_shadow_days: int,
    prospective_shadow_outcome_days: int,
    prospective_shadow_signal_count: int,
    prospective_shadow_outcome_signal_count: int,
    prospective_shadow_positive: bool,
    approved_statistical_gate_verified: bool,
    unresolved_failures: int,
    explicit_testnet_release: bool,
) -> TestnetEligibility:
    """Require all evidence plus a separate human release; never activate anything."""
    reasons: list[str] = []
    if experiment.status != "APPROVED":
        reasons.append("experiment_status_is_not_approved")
    if not approved_statistical_gate_verified:
        reasons.append("approved_statistical_gate_evidence_missing")
    if not book_evidence_eligible:
        reasons.append("book_evidence_gate_not_eligible")
    if book_evidence_contiguous_days < MIN_BOOK_EVIDENCE_DAYS:
        reasons.append("book_evidence_has_fewer_than_60_complete_days")
    if prospective_shadow_days < MIN_PROSPECTIVE_SHADOW_DAYS:
        reasons.append("prospective_shadow_has_fewer_than_20_days")
    if prospective_shadow_days > MAX_PROSPECTIVE_SHADOW_DAYS:
        reasons.append("prospective_shadow_exceeds_30_days")
    if prospective_shadow_outcome_days < prospective_shadow_days:
        reasons.append("prospective_shadow_outcomes_incomplete")
    if prospective_shadow_outcome_signal_count < prospective_shadow_signal_count:
        reasons.append("prospective_shadow_signal_outcomes_incomplete")
    if not prospective_shadow_positive:
        reasons.append("prospective_shadow_block_not_positive")
    if unresolved_failures > 0:
        reasons.append("unresolved_reconciliation_or_data_failures")
    if not explicit_testnet_release:
        reasons.append("explicit_testnet_release_is_missing")
    return TestnetEligibility(
        eligible=not reasons,
        reasons=tuple(reasons),
        book_evidence_eligible=book_evidence_eligible,
        book_evidence_contiguous_days=book_evidence_contiguous_days,
        prospective_shadow_days=prospective_shadow_days,
        prospective_shadow_outcome_days=prospective_shadow_outcome_days,
        prospective_shadow_signal_count=prospective_shadow_signal_count,
        prospective_shadow_outcome_signal_count=prospective_shadow_outcome_signal_count,
        prospective_shadow_positive=prospective_shadow_positive,
        approved_statistical_gate_verified=approved_statistical_gate_verified,
    )

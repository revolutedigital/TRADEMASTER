"""Research-to-Testnet eligibility checks; this module cannot submit orders."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.research_experiment import ResearchExperiment


@dataclass(frozen=True)
class TestnetEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    execution_authorization: str = "none"
    order_submission_allowed: bool = False


def evaluate_testnet_eligibility(
    experiment: ResearchExperiment,
    *,
    prospective_days: int,
    unresolved_failures: int,
    explicit_testnet_release: bool,
) -> TestnetEligibility:
    """Require all evidence plus a separate human release; never activate anything."""
    reasons: list[str] = []
    if experiment.status != "APPROVED":
        reasons.append("experiment_status_is_not_approved")
    if prospective_days < 20:
        reasons.append("prospective_shadow_has_fewer_than_20_days")
    if unresolved_failures > 0:
        reasons.append("unresolved_reconciliation_or_data_failures")
    if not explicit_testnet_release:
        reasons.append("explicit_testnet_release_is_missing")
    return TestnetEligibility(eligible=not reasons, reasons=tuple(reasons))

"""Research-to-Testnet eligibility checks; this module cannot submit orders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import ResearchExperiment, ResearchTestnetRelease
from app.repositories.research_experiment_repo import research_experiment_repository


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


@dataclass(frozen=True)
class ResearchTestnetReleaseReadiness:
    """Runtime-safe summary for whether a research release can unlock Testnet activation."""

    ready: bool
    reasons: tuple[str, ...]
    experiment_id: str | None = None
    release_sha256: str | None = None


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


async def research_testnet_release_readiness(
    db: AsyncSession,
) -> ResearchTestnetReleaseReadiness:
    """Require an intact research-only release before runtime TESTNET activation."""
    result = await db.execute(
        select(ResearchTestnetRelease, ResearchExperiment)
        .join(ResearchExperiment, ResearchExperiment.id == ResearchTestnetRelease.experiment_id)
        .order_by(ResearchTestnetRelease.released_at.desc(), ResearchTestnetRelease.id.desc())
        .limit(1)
    )
    row = result.first()
    if row is None:
        return ResearchTestnetReleaseReadiness(
            ready=False,
            reasons=("research_testnet_release_missing",),
        )

    release, experiment = row.tuple()
    reasons = _release_snapshot_rejection_reasons(release, experiment)
    event_chain = await research_experiment_repository.event_chain_status(db, experiment.id)
    if not event_chain.verified:
        reasons.append("research_event_chain_unverified")

    return ResearchTestnetReleaseReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        experiment_id=experiment.id,
        release_sha256=release.release_sha256,
    )


def _release_snapshot_rejection_reasons(
    release: ResearchTestnetRelease,
    experiment: ResearchExperiment,
) -> list[str]:
    reasons: list[str] = []
    if experiment.status != "APPROVED":
        reasons.append("research_experiment_status_is_not_approved")
    if not _is_sha256(experiment.experiment_sha256):
        reasons.append("research_experiment_hash_missing")
    if not _is_sha256(release.release_sha256):
        reasons.append("research_testnet_release_hash_invalid")

    try:
        snapshot = json.loads(release.evidence_snapshot_json)
    except json.JSONDecodeError:
        return [*reasons, "research_testnet_release_snapshot_unreadable"]
    if not isinstance(snapshot, dict):
        return [*reasons, "research_testnet_release_snapshot_invalid"]

    if snapshot.get("experiment_status") != "APPROVED":
        reasons.append("release_snapshot_experiment_not_approved")
    if snapshot.get("experiment_sha256") != experiment.experiment_sha256:
        reasons.append("release_snapshot_experiment_hash_mismatch")
    if snapshot.get("book_evidence_eligible") is not True:
        reasons.append("release_snapshot_book_evidence_not_eligible")
    if _int_value(snapshot.get("book_evidence_contiguous_days")) < MIN_BOOK_EVIDENCE_DAYS:
        reasons.append("release_snapshot_book_evidence_has_fewer_than_60_complete_days")
    shadow_days = _int_value(snapshot.get("prospective_shadow_days"))
    if shadow_days < MIN_PROSPECTIVE_SHADOW_DAYS:
        reasons.append("release_snapshot_shadow_has_fewer_than_20_days")
    if shadow_days > MAX_PROSPECTIVE_SHADOW_DAYS:
        reasons.append("release_snapshot_shadow_exceeds_30_days")
    if _int_value(snapshot.get("prospective_shadow_outcome_days")) < shadow_days:
        reasons.append("release_snapshot_shadow_outcomes_incomplete")
    if _int_value(snapshot.get("prospective_shadow_signal_count")) <= 0:
        reasons.append("release_snapshot_shadow_signals_missing")
    if _int_value(snapshot.get("prospective_shadow_outcome_signal_count")) < _int_value(
        snapshot.get("prospective_shadow_signal_count")
    ):
        reasons.append("release_snapshot_shadow_signal_outcomes_incomplete")
    if snapshot.get("prospective_shadow_positive") is not True:
        reasons.append("release_snapshot_shadow_not_positive")
    if snapshot.get("shadow_ledger_verified") is not True:
        reasons.append("release_snapshot_shadow_ledger_unverified")
        shadow_ledger = snapshot.get("shadow_ledger")
        if isinstance(shadow_ledger, dict):
            ledger_reasons = shadow_ledger.get("reasons")
            if isinstance(ledger_reasons, list):
                reasons.extend(
                    f"release_snapshot_{reason}"
                    for reason in ledger_reasons
                    if isinstance(reason, str)
                )
    expected_mean = _float_value(snapshot.get("prospective_shadow_expected_mean_bps"))
    stress_mean = _float_value(snapshot.get("prospective_shadow_stress_mean_bps"))
    if expected_mean <= 0:
        reasons.append("release_snapshot_expected_mean_not_positive")
    if stress_mean <= 0:
        reasons.append("release_snapshot_stress_mean_not_positive")
    if snapshot.get("approved_statistical_gate_verified") is not True:
        reasons.append("release_snapshot_statistical_gate_not_verified")
    if _int_value(snapshot.get("unresolved_failures")) != 0:
        reasons.append("release_snapshot_has_unresolved_failures")
    if snapshot.get("order_submission_allowed") is not False:
        reasons.append("release_snapshot_order_submission_boundary_invalid")
    if snapshot.get("execution_authorization") != "none":
        reasons.append("release_snapshot_execution_authorization_invalid")
    return reasons


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return 0.0
    return parsed if isfinite(parsed) else 0.0

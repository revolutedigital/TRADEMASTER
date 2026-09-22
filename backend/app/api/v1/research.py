"""Research metadata endpoints with no execution capabilities."""

from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path as ApiPath, Query, Response, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db, require_auth
from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchExperimentEvent,
    ResearchShadowSignal,
    ResearchTestnetRelease,
)
from app.repositories.research_experiment_repo import research_experiment_repository
from app.schemas.research_experiment import (
    CreateExperimentRequest,
    DatasetPartitionRole,
    EvidenceGateStatusResponse,
    ExperimentReportResponse,
    ExperimentResponse,
    OpenedPartitionResponse,
    RecordExperimentDecisionRequest,
    RecordShadowOutcomeRequest,
    RecordShadowSignalRequest,
    RecordTestnetReleaseRequest,
    ResearchTestnetReleaseResponse,
    ShadowSignalResponse,
    StatisticalGateEvidence,
    TestnetEligibilityResponse,
)
from app.services.data.research_registry import (
    BurnedDataConflict,
    ExperimentDefinition,
    FrozenExperimentError,
    PartitionDefinition,
    ResearchRegistryError,
    research_registry,
)
from app.services.market.microstructure_wal_audit import build_evidence_gate_status
from app.services.research.shadow_recorder import ShadowRecorderError, research_shadow_recorder
from app.services.research.testnet_release_gate import evaluate_testnet_eligibility


router = APIRouter()

REQUIRED_STATISTICAL_GATE_CONDITIONS = frozenset(
    {
        "three_temporal_folds",
        "minimum_200_trades",
        "minimum_20_days",
        "positive_expected_mean",
        "adjusted_lower_bound_positive",
        "positive_stress_mean",
        "top_p_monotonic",
        "pbo_at_most_20_percent",
        "prospective_positive",
    }
)


@router.get("/evidence-gate", response_model=EvidenceGateStatusResponse)
async def get_evidence_gate_status(
    _user: dict = Depends(require_auth),
) -> EvidenceGateStatusResponse:
    return await _read_evidence_gate_status()


@router.get(
    "/experiments/{experiment_id}/testnet-eligibility",
    response_model=TestnetEligibilityResponse,
)
async def get_testnet_eligibility(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> TestnetEligibilityResponse:
    experiment = await research_experiment_repository.get(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Research experiment was not found")

    evidence_status = await _read_evidence_gate_status()
    shadow_signals = await research_experiment_repository.list_shadow_signals(
        db,
        experiment_id,
    )
    shadow_summary = _summarize_shadow_outcomes(shadow_signals)
    unresolved_failures = len(evidence_status.status_reasons)
    testnet_release = await _get_testnet_release(db, experiment_id)
    explicit_testnet_release = testnet_release is not None
    approved_statistical_gate_verified = await _approved_statistical_gate_verified(
        db,
        experiment_id,
    )
    eligibility = evaluate_testnet_eligibility(
        experiment,
        book_evidence_eligible=(
            evidence_status.artifact_available and evidence_status.book_evidence_gate.eligible
        ),
        book_evidence_contiguous_days=(
            evidence_status.book_evidence_gate.longest_complete_streak_days
            if evidence_status.artifact_available
            else 0
        ),
        prospective_shadow_days=shadow_summary["decision_days"],
        prospective_shadow_outcome_days=shadow_summary["outcome_days"],
        prospective_shadow_signal_count=shadow_summary["signal_count"],
        prospective_shadow_outcome_signal_count=shadow_summary["outcome_signal_count"],
        prospective_shadow_positive=shadow_summary["positive"],
        approved_statistical_gate_verified=approved_statistical_gate_verified,
        unresolved_failures=unresolved_failures,
        explicit_testnet_release=explicit_testnet_release,
    )
    return TestnetEligibilityResponse(
        experiment_id=experiment.id,
        experiment_status=experiment.status,
        eligible=eligibility.eligible,
        reasons=list(eligibility.reasons),
        book_evidence_eligible=eligibility.book_evidence_eligible,
        book_evidence_contiguous_days=eligibility.book_evidence_contiguous_days,
        prospective_shadow_days=eligibility.prospective_shadow_days,
        prospective_shadow_outcome_days=eligibility.prospective_shadow_outcome_days,
        prospective_shadow_signal_count=eligibility.prospective_shadow_signal_count,
        prospective_shadow_outcome_signal_count=eligibility.prospective_shadow_outcome_signal_count,
        prospective_shadow_expected_mean_bps=shadow_summary["expected_mean_bps"],
        prospective_shadow_stress_mean_bps=shadow_summary["stress_mean_bps"],
        prospective_shadow_positive=eligibility.prospective_shadow_positive,
        approved_statistical_gate_verified=eligibility.approved_statistical_gate_verified,
        unresolved_failures=unresolved_failures,
        explicit_testnet_release=explicit_testnet_release,
        release_request_required=not explicit_testnet_release,
        evidence_artifact_available=evidence_status.artifact_available,
        order_submission_allowed=False,
        execution_authorization="none",
        safety=_safety(),
        generated_at=datetime.now(UTC),
    )


@router.post(
    "/experiments/{experiment_id}/testnet-release",
    response_model=ResearchTestnetReleaseResponse,
    status_code=status.HTTP_201_CREATED,
)
async def record_testnet_release(
    experiment_id: str,
    body: RecordTestnetReleaseRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_auth),
) -> ResearchTestnetReleaseResponse:
    experiment = await research_experiment_repository.get(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Research experiment was not found")

    existing_release = await _get_testnet_release(db, experiment_id)
    if existing_release is not None:
        response.status_code = status.HTTP_200_OK
        return _serialize_testnet_release(existing_release)

    evidence_status = await _read_evidence_gate_status()
    shadow_signals = await research_experiment_repository.list_shadow_signals(db, experiment_id)
    shadow_summary = _summarize_shadow_outcomes(shadow_signals)
    unresolved_failures = len(evidence_status.status_reasons)
    approved_statistical_gate_verified = await _approved_statistical_gate_verified(
        db,
        experiment_id,
    )
    eligibility = evaluate_testnet_eligibility(
        experiment,
        book_evidence_eligible=(
            evidence_status.artifact_available and evidence_status.book_evidence_gate.eligible
        ),
        book_evidence_contiguous_days=(
            evidence_status.book_evidence_gate.longest_complete_streak_days
            if evidence_status.artifact_available
            else 0
        ),
        prospective_shadow_days=shadow_summary["decision_days"],
        prospective_shadow_outcome_days=shadow_summary["outcome_days"],
        prospective_shadow_signal_count=shadow_summary["signal_count"],
        prospective_shadow_outcome_signal_count=shadow_summary["outcome_signal_count"],
        prospective_shadow_positive=shadow_summary["positive"],
        approved_statistical_gate_verified=approved_statistical_gate_verified,
        unresolved_failures=unresolved_failures,
        explicit_testnet_release=True,
    )
    if not eligibility.eligible:
        raise HTTPException(
            status_code=409,
            detail={
                "reasons": list(eligibility.reasons),
                "order_submission_allowed": False,
                "execution_authorization": "none",
            },
        )

    release_time = datetime.now(UTC)
    evidence_snapshot = _testnet_release_evidence_snapshot(
        experiment=experiment,
        evidence_status=evidence_status,
        shadow_summary=shadow_summary,
        unresolved_failures=unresolved_failures,
        approved_statistical_gate_verified=approved_statistical_gate_verified,
        generated_at=release_time,
    )
    requested_by = str(user.get("sub", "operator"))[:120]
    reasons = body.reasons
    release_payload = {
        "experiment_id": experiment.id,
        "experiment_sha256": experiment.experiment_sha256,
        "requested_by": requested_by,
        "reasons": reasons,
        "evidence_snapshot": evidence_snapshot,
        "released_at": release_time.isoformat(),
        "safety": _safety(),
    }
    release = ResearchTestnetRelease(
        experiment_id=experiment.id,
        release_sha256=_report_sha256(release_payload),
        requested_by=requested_by,
        reasons_json=json.dumps(reasons, ensure_ascii=False, sort_keys=True),
        evidence_snapshot_json=json.dumps(
            evidence_snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        released_at=release_time,
    )
    db.add(release)
    db.add(
        ResearchExperimentEvent(
            experiment_id=experiment.id,
            kind="TESTNET_RELEASE_RECORDED",
            payload_json=json.dumps(
                {
                    "release_sha256": release.release_sha256,
                    "order_submission_allowed": False,
                    "execution_authorization": "none",
                },
                sort_keys=True,
            ),
            occurred_at=release_time,
        )
    )
    await db.flush()
    return _serialize_testnet_release(release)


@router.post(
    "/experiments/{experiment_id}/partitions/{role}/open",
    response_model=OpenedPartitionResponse,
)
async def open_experiment_partition(
    experiment_id: str,
    role: Annotated[DatasetPartitionRole, ApiPath()],
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> OpenedPartitionResponse:
    try:
        partition = await research_registry.open_partition(db, experiment_id, role=role)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ResearchRegistryError, BurnedDataConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _serialize_opened_partition(partition)


@router.post(
    "/experiments/{experiment_id}/shadow-signals",
    response_model=ShadowSignalResponse,
    status_code=status.HTTP_201_CREATED,
)
async def record_shadow_signal(
    experiment_id: str,
    body: RecordShadowSignalRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> ShadowSignalResponse:
    try:
        signal = await research_shadow_recorder.record(
            db,
            experiment_id=experiment_id,
            decision_time=body.decision_time,
            side=body.side,
            horizon_seconds=body.horizon_seconds,
            probability=body.probability,
            threshold=body.threshold,
            model_sha256=body.model_sha256,
            feature_vector=body.feature_vector,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ShadowRecorderError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _serialize_shadow_signal(signal)


@router.post(
    "/shadow-signals/{signal_id}/outcome",
    response_model=ShadowSignalResponse,
)
async def record_shadow_outcome(
    signal_id: Annotated[int, ApiPath(ge=1)],
    body: RecordShadowOutcomeRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> ShadowSignalResponse:
    try:
        signal = await research_shadow_recorder.record_outcome(
            db,
            signal_id=signal_id,
            expected_net_bps=body.expected_net_bps,
            stress_net_bps=body.stress_net_bps,
            label_sha256=body.label_sha256,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ShadowRecorderError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return _serialize_shadow_signal(signal)


async def _read_evidence_gate_status() -> EvidenceGateStatusResponse:
    if settings.microstructure_evidence_status_url:
        return await _read_remote_evidence_gate_status(settings.microstructure_evidence_status_url)
    artifact_path = Path(settings.microstructure_evidence_status_path)
    if not artifact_path.exists():
        return _fallback_evidence_gate_status("evidence_gate_artifact_missing")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        return EvidenceGateStatusResponse.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        return _fallback_evidence_gate_status(
            f"evidence_gate_artifact_unreadable:{type(error).__name__}"
        )


async def _read_remote_evidence_gate_status(url: str) -> EvidenceGateStatusResponse:
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
            response = await client.get(url)
            response.raise_for_status()
        return EvidenceGateStatusResponse.model_validate(response.json())
    except (
        httpx.HTTPError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as error:
        return _fallback_evidence_gate_status(
            f"evidence_gate_artifact_remote_unreadable:{type(error).__name__}"
        )


@router.post("/experiments", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    body: CreateExperimentRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    try:
        experiment = await research_registry.create_draft(
            db,
            ExperimentDefinition(
                name=body.name,
                code_revision=body.code_revision,
                protocol_sha256=body.protocol_sha256,
                product=body.product.model_dump(mode="json"),
                cost_profile=body.cost_profile.model_dump(mode="json"),
                approval_gate=body.approval_gate.model_dump(mode="json"),
                partitions=tuple(
                    PartitionDefinition(**partition.model_dump())
                    for partition in body.dataset_partitions
                ),
            ),
        )
        for hypothesis in body.hypotheses:
            await research_registry.register_hypothesis(
                db,
                experiment.id,
                kind=hypothesis.kind,
                definition=hypothesis.definition,
            )
        return await _serialize_experiment(db, experiment)
    except (ResearchRegistryError, BurnedDataConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/experiments", response_model=list[ExperimentResponse])
async def list_experiments(
    experiment_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> list[dict[str, object]]:
    if experiment_status is not None and experiment_status not in {
        "DRAFT",
        "FROZEN",
        "REJECTED",
        "INCONCLUSIVE",
        "APPROVED",
    }:
        raise HTTPException(status_code=422, detail="Invalid experiment status")
    experiments = await research_experiment_repository.list(
        db, status=experiment_status, limit=limit
    )
    return [await _serialize_experiment(db, experiment) for experiment in experiments]


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    experiment = await research_experiment_repository.get(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Research experiment was not found")
    return await _serialize_experiment(db, experiment)


@router.post("/experiments/{experiment_id}/freeze", response_model=ExperimentResponse)
async def freeze_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    try:
        experiment = await research_registry.freeze(db, experiment_id)
        return await _serialize_experiment(db, experiment)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ResearchRegistryError, FrozenExperimentError, BurnedDataConflict) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/experiments/{experiment_id}/decision", response_model=ExperimentResponse)
async def record_experiment_decision(
    experiment_id: str,
    body: RecordExperimentDecisionRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    try:
        decision_evidence = _decision_evidence_or_raise(body)
        experiment = await research_registry.record_decision(
            db,
            experiment_id,
            status=body.status,
            reasons=body.reasons,
            evidence=decision_evidence,
        )
        return await _serialize_experiment(db, experiment)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ResearchRegistryError, FrozenExperimentError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/experiments/{experiment_id}/report", response_model=ExperimentReportResponse)
async def get_experiment_report(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    experiment = await research_experiment_repository.get(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Research experiment was not found")
    if experiment.status == "DRAFT":
        raise HTTPException(status_code=409, detail="Draft experiment has no frozen report")
    evidence_status = await _read_evidence_gate_status()
    hypotheses = await research_experiment_repository.list_hypotheses(db, experiment.id)
    shadow_signals = await research_experiment_repository.list_shadow_signals(db, experiment.id)
    shadow_summary = _summarize_shadow_outcomes(shadow_signals)
    testnet_release = await _get_testnet_release(db, experiment.id)
    explicit_testnet_release = testnet_release is not None
    approved_statistical_gate_verified = await _approved_statistical_gate_verified(
        db,
        experiment.id,
    )
    metrics = {
        "experiment_sha256": experiment.experiment_sha256,
        "book_evidence": {
            "artifact_available": evidence_status.artifact_available,
            "eligible": evidence_status.book_evidence_gate.eligible,
            "required_complete_days": evidence_status.book_evidence_gate.required_complete_days,
            "longest_complete_streak_days": (
                evidence_status.book_evidence_gate.longest_complete_streak_days
            ),
            "complete_days": evidence_status.book_evidence_gate.complete_days,
            "manifest_sha256": evidence_status.book_evidence_gate.manifest_sha256,
            "status_reasons": evidence_status.status_reasons,
            "gate_reasons": evidence_status.book_evidence_gate.reasons,
        },
        "shadow": {
            **shadow_summary,
            "complete": (
                shadow_summary["signal_count"] > 0
                and shadow_summary["outcome_signal_count"] == shadow_summary["signal_count"]
                and shadow_summary["outcome_days"] == shadow_summary["decision_days"]
            ),
        },
        "hypothesis_ledger": {
            "attempt_count": len(hypotheses),
            "attempts": [
                {
                    "kind": hypothesis.kind,
                    "status": hypothesis.status,
                    "fingerprint_sha256": hypothesis.fingerprint_sha256,
                    "definition": json.loads(hypothesis.definition_json),
                }
                for hypothesis in hypotheses
            ],
        },
        "testnet_boundary": {
            "approved_statistical_gate_verified": approved_statistical_gate_verified,
            "release_request_required": not explicit_testnet_release,
            "explicit_testnet_release": explicit_testnet_release,
            "order_submission_allowed": False,
            "execution_authorization": "none",
        },
    }
    decision_reasons = json.loads(experiment.decision_reasons_json)
    return {
        "experiment_id": experiment.id,
        "status": experiment.status,
        "decision_reasons": decision_reasons,
        "metrics": metrics,
        "artifact_sha256": _report_sha256(
            {
                "experiment_id": experiment.id,
                "status": experiment.status,
                "decision_reasons": decision_reasons,
                "metrics": metrics,
                "safety": _safety(),
            }
        ),
        "safety": _safety(),
        "generated_at": experiment.decided_at or experiment.frozen_at or datetime.now(UTC),
    }


async def _serialize_experiment(
    db: AsyncSession, experiment: ResearchExperiment
) -> dict[str, object]:
    partitions = await research_experiment_repository.list_partitions(db, experiment.id)
    return {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
        "code_revision": experiment.code_revision,
        "protocol_sha256": experiment.protocol_sha256,
        "experiment_sha256": experiment.experiment_sha256,
        "product": json.loads(experiment.product_json),
        "cost_profile": json.loads(experiment.cost_profile_json),
        "dataset_partitions": [
            {
                "role": partition.role,
                "start_at": partition.start_at,
                "end_at": partition.end_at,
                "manifest_sha256": partition.manifest_sha256,
            }
            for partition in partitions
        ],
        "approval_gate": json.loads(experiment.approval_gate_json),
        "safety": _safety(),
        "created_at": experiment.created_at,
        "frozen_at": experiment.frozen_at,
    }


def _safety() -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def _fallback_evidence_gate_status(reason: str) -> EvidenceGateStatusResponse:
    payload = build_evidence_gate_status(
        (),
        artifact_available=False,
        status_reasons=(reason,),
    )
    return EvidenceGateStatusResponse.model_validate(payload)


def _serialize_shadow_signal(signal: ResearchShadowSignal) -> ShadowSignalResponse:
    outcome = _parse_shadow_outcome(signal)
    return ShadowSignalResponse(
        id=signal.id,
        experiment_id=signal.experiment_id,
        decision_time=signal.decision_time,
        recorded_at=signal.recorded_at,
        side=signal.side,
        horizon_seconds=signal.horizon_seconds,
        probability=signal.probability,
        threshold=signal.threshold,
        would_enter=signal.would_enter,
        model_sha256=signal.model_sha256,
        feature_vector_sha256=signal.feature_vector_sha256,
        outcome_recorded=outcome is not None,
        expected_net_bps=None if outcome is None else outcome["expected_net_bps"],
        stress_net_bps=None if outcome is None else outcome["stress_net_bps"],
        label_sha256=None if outcome is None else outcome["label_sha256"],
        safety=_safety(),
    )


def _decision_evidence_or_raise(
    body: RecordExperimentDecisionRequest,
) -> dict[str, object] | None:
    statistical_gate = body.statistical_gate
    if body.status != "APPROVED":
        if statistical_gate is None:
            return None
        gate_payload = statistical_gate.model_dump(mode="json")
        return {
            "statistical_gate_sha256": _report_sha256(gate_payload),
            "statistical_gate_decision": _best_statistical_gate_decision(statistical_gate),
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
        }

    if statistical_gate is None:
        _raise_approval_gate_conflict(("statistical_gate_artifact_required_for_approval",))

    reasons = _statistical_gate_approval_reasons(statistical_gate)
    if reasons:
        _raise_approval_gate_conflict(tuple(reasons))

    gate_payload = statistical_gate.model_dump(mode="json")
    return {
        "statistical_gate_sha256": _report_sha256(gate_payload),
        "statistical_gate_decision": "APPROVED",
        "attempted_hypotheses": statistical_gate.attempted_hypotheses,
        "approved_strategy_count": _approved_statistical_gate_count(statistical_gate),
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def _statistical_gate_approval_reasons(gate: StatisticalGateEvidence) -> list[str]:
    reasons: list[str] = []
    if gate.top_p_monotonic is not True:
        reasons.append("top_p_calibration_not_monotonic")
        reasons.extend(gate.top_p_monotonic_reasons)
    if gate.prospective_shadow_positive is not True:
        reasons.append("prospective_shadow_not_positive")
        reasons.extend(gate.prospective_shadow_reasons)
    approved_results = [
        result for result in gate.results if str(result.get("decision", "")).upper() == "APPROVED"
    ]
    reasons.extend(_statistical_gate_count_reasons(gate))
    if not approved_results:
        reasons.append("statistical_gate_has_no_approved_portfolio")
        return reasons
    for index, result in enumerate(approved_results):
        reasons.extend(_approved_result_reasons(index, result, gate.attempted_hypotheses))
    return reasons


def _statistical_gate_count_reasons(gate: StatisticalGateEvidence) -> list[str]:
    expected_counts: Counter[str] = Counter()
    reasons: list[str] = []
    for index, result in enumerate(gate.results):
        decision = str(result.get("decision", "")).upper()
        if decision not in {"APPROVED", "REJECTED", "INCONCLUSIVE"}:
            reasons.append(f"statistical_gate_result_{index}_decision_invalid")
            continue
        expected_counts[decision] += 1

    provided_counts: dict[str, int] = {}
    for decision, count in gate.decision_counts.items():
        normalized_decision = str(decision).upper()
        if (
            normalized_decision not in {"APPROVED", "REJECTED", "INCONCLUSIVE"}
            or type(count) is not int
            or count < 0
        ):
            reasons.append(f"statistical_gate_decision_count_{decision}_invalid")
            continue
        provided_counts[normalized_decision] = count

    expected = {
        decision: expected_counts[decision]
        for decision in ("APPROVED", "REJECTED", "INCONCLUSIVE")
        if expected_counts[decision] > 0
    }
    if provided_counts != expected:
        reasons.append("statistical_gate_decision_counts_do_not_match_results")
    return reasons


def _approved_result_reasons(
    index: int,
    result: dict[str, object],
    attempted_hypotheses: int,
) -> list[str]:
    prefix = f"approved_result_{index}"
    reasons: list[str] = []
    if int(result.get("trade_count") or 0) < 200:
        reasons.append(f"{prefix}_has_fewer_than_200_trades")
    if int(result.get("distinct_days") or 0) < 20:
        reasons.append(f"{prefix}_has_fewer_than_20_days")
    if (_finite_float(result.get("expected_mean_bps")) or 0.0) <= 0:
        reasons.append(f"{prefix}_expected_mean_not_positive")
    if (_finite_float(result.get("stress_mean_bps")) or 0.0) <= 0:
        reasons.append(f"{prefix}_stress_mean_not_positive")
    lower_bound = _finite_float(result.get("adjusted_lower_confidence_bound_bps"))
    if lower_bound is None or lower_bound <= 0:
        reasons.append(f"{prefix}_adjusted_lower_bound_not_positive")
    adjusted_alpha = _finite_float(result.get("adjusted_one_sided_alpha"))
    expected_alpha = 0.05 / attempted_hypotheses
    if adjusted_alpha is None or not math.isclose(
        adjusted_alpha,
        expected_alpha,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        reasons.append(f"{prefix}_adjusted_alpha_inconsistent")
    pbo = _finite_float(result.get("probability_of_backtest_overfitting"))
    if pbo is None or pbo > 0.20:
        reasons.append(f"{prefix}_pbo_above_20_percent")
    result_reasons = result.get("reasons")
    if isinstance(result_reasons, list) and result_reasons:
        reasons.append(f"{prefix}_has_failure_reasons")
    conditions = result.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        reasons.append(f"{prefix}_conditions_missing")
    else:
        missing_conditions = sorted(REQUIRED_STATISTICAL_GATE_CONDITIONS - set(conditions))
        reasons.extend(
            f"{prefix}_{condition}_condition_missing"
            for condition in missing_conditions
        )
        failed_conditions = [
            str(name)
            for name, passed in conditions.items()
            if passed is not True
        ]
        reasons.extend(f"{prefix}_{name}_failed" for name in failed_conditions)
    return reasons


def _approved_statistical_gate_count(gate: StatisticalGateEvidence) -> int:
    return sum(1 for result in gate.results if str(result.get("decision", "")).upper() == "APPROVED")


def _best_statistical_gate_decision(gate: StatisticalGateEvidence) -> str:
    if _approved_statistical_gate_count(gate):
        return "APPROVED"
    if int(gate.decision_counts.get("REJECTED", 0)) > 0:
        return "REJECTED"
    if int(gate.decision_counts.get("INCONCLUSIVE", 0)) > 0:
        return "INCONCLUSIVE"
    return "UNKNOWN"


def _raise_approval_gate_conflict(reasons: tuple[str, ...]) -> None:
    raise HTTPException(
        status_code=409,
        detail={
            "reasons": list(reasons),
            "order_submission_allowed": False,
            "execution_authorization": "none",
        },
    )


def _serialize_opened_partition(partition: ResearchDataUse) -> OpenedPartitionResponse:
    if partition.opened_at is None:
        raise HTTPException(status_code=409, detail="Research partition was not opened")
    return OpenedPartitionResponse(
        id=partition.id,
        experiment_id=partition.experiment_id,
        role=partition.role,
        start_at=_as_utc_datetime(partition.start_at),
        end_at=_as_utc_datetime(partition.end_at),
        manifest_sha256=partition.manifest_sha256,
        opened_at=_as_utc_datetime(partition.opened_at),
        safety=_safety(),
    )


async def _get_testnet_release(
    db: AsyncSession,
    experiment_id: str,
) -> ResearchTestnetRelease | None:
    result = await db.execute(
        select(ResearchTestnetRelease).where(
            ResearchTestnetRelease.experiment_id == experiment_id
        )
    )
    return result.scalar_one_or_none()


async def _approved_statistical_gate_verified(
    db: AsyncSession,
    experiment_id: str,
) -> bool:
    events = await research_experiment_repository.list_events(
        db,
        experiment_id,
        kind="DECISION_RECORDED",
    )
    for event in reversed(events):
        try:
            payload = json.loads(event.payload_json)
        except json.JSONDecodeError:
            return False
        if payload.get("status") != "APPROVED":
            return False
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            return False
        gate_sha256 = evidence.get("statistical_gate_sha256")
        if (
            not isinstance(gate_sha256, str)
            or len(gate_sha256) != 64
            or any(character not in "0123456789abcdef" for character in gate_sha256)
        ):
            return False
        approved_strategy_count = evidence.get("approved_strategy_count")
        if type(approved_strategy_count) is not int:
            return False
        return (
            evidence.get("statistical_gate_decision") == "APPROVED"
            and approved_strategy_count > 0
            and evidence.get("research_only") is True
            and evidence.get("order_submission_allowed") is False
            and evidence.get("execution_authorization") == "none"
        )
    return False


def _serialize_testnet_release(
    release: ResearchTestnetRelease,
) -> ResearchTestnetReleaseResponse:
    return ResearchTestnetReleaseResponse(
        id=release.id,
        experiment_id=release.experiment_id,
        release_sha256=release.release_sha256,
        requested_by=release.requested_by,
        reasons=json.loads(release.reasons_json),
        evidence_snapshot=json.loads(release.evidence_snapshot_json),
        released_at=_as_utc_datetime(release.released_at),
        explicit_testnet_release=True,
        release_request_required=False,
        order_submission_allowed=False,
        execution_authorization="none",
        safety=_safety(),
    )


def _testnet_release_evidence_snapshot(
    *,
    experiment: ResearchExperiment,
    evidence_status: EvidenceGateStatusResponse,
    shadow_summary: dict[str, object],
    unresolved_failures: int,
    approved_statistical_gate_verified: bool,
    generated_at: datetime,
) -> dict[str, object]:
    return {
        "experiment_status": experiment.status,
        "experiment_sha256": experiment.experiment_sha256,
        "book_evidence_contiguous_days": (
            evidence_status.book_evidence_gate.longest_complete_streak_days
            if evidence_status.artifact_available
            else 0
        ),
        "book_evidence_eligible": (
            evidence_status.artifact_available and evidence_status.book_evidence_gate.eligible
        ),
        "evidence_artifact_available": evidence_status.artifact_available,
        "evidence_manifest_sha256": evidence_status.book_evidence_gate.manifest_sha256,
        "prospective_shadow_days": shadow_summary["decision_days"],
        "prospective_shadow_outcome_days": shadow_summary["outcome_days"],
        "prospective_shadow_signal_count": shadow_summary["signal_count"],
        "prospective_shadow_outcome_signal_count": shadow_summary["outcome_signal_count"],
        "prospective_shadow_expected_mean_bps": shadow_summary["expected_mean_bps"],
        "prospective_shadow_stress_mean_bps": shadow_summary["stress_mean_bps"],
        "prospective_shadow_positive": shadow_summary["positive"],
        "approved_statistical_gate_verified": approved_statistical_gate_verified,
        "unresolved_failures": unresolved_failures,
        "generated_at": generated_at.isoformat(),
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def _parse_shadow_outcome(signal: ResearchShadowSignal) -> dict[str, object] | None:
    if signal.outcome_json is None:
        return None
    try:
        payload = json.loads(signal.outcome_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _validated_shadow_outcome(payload, signal)


def _summarize_shadow_outcomes(signals: list[ResearchShadowSignal]) -> dict[str, object]:
    decision_days = {_utc_day(signal.decision_time) for signal in signals}
    outcome_days = set()
    outcome_signal_count = 0
    expected_values: list[float] = []
    stress_values: list[float] = []

    for signal in signals:
        if not signal.outcome_json:
            continue
        try:
            payload = json.loads(signal.outcome_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        outcome = _validated_shadow_outcome(payload, signal)
        if outcome is None:
            continue
        expected_net_bps = outcome["expected_net_bps"]
        stress_net_bps = outcome["stress_net_bps"]
        expected_values.append(expected_net_bps)
        stress_values.append(stress_net_bps)
        outcome_days.add(_utc_day(signal.decision_time))
        outcome_signal_count += 1

    expected_mean = _mean(expected_values)
    stress_mean = _mean(stress_values)
    positive = (
        bool(decision_days)
        and outcome_days == decision_days
        and outcome_signal_count == len(signals)
        and expected_mean is not None
        and stress_mean is not None
        and expected_mean > 0
        and stress_mean > 0
    )
    return {
        "signal_count": len(signals),
        "outcome_signal_count": outcome_signal_count,
        "decision_days": len(decision_days),
        "outcome_days": len(outcome_days),
        "expected_mean_bps": expected_mean,
        "stress_mean_bps": stress_mean,
        "positive": positive,
    }


def _utc_day(value: datetime):
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).date()


def _as_utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _validated_shadow_outcome(
    payload: dict[str, object],
    signal: ResearchShadowSignal,
) -> dict[str, object] | None:
    expected_net_bps = _finite_float(payload.get("expected_net_bps"))
    stress_net_bps = _finite_float(payload.get("stress_net_bps"))
    label_sha256 = payload.get("label_sha256")
    recorded_at = payload.get("recorded_at")
    if expected_net_bps is None or stress_net_bps is None:
        return None
    if not isinstance(label_sha256, str) or not _is_lower_sha256(label_sha256):
        return None
    if payload.get("research_only") is not True:
        return None
    if payload.get("order_submission_allowed") is not False:
        return None
    if payload.get("execution_authorization") != "none":
        return None
    if not isinstance(recorded_at, str):
        return None
    recorded_at_datetime = _parse_iso_datetime(recorded_at)
    if recorded_at_datetime is None:
        return None
    horizon_end = _as_utc_datetime(signal.decision_time) + timedelta(
        seconds=signal.horizon_seconds
    )
    if _as_utc_datetime(recorded_at_datetime) < horizon_end:
        return None
    if _contains_order_or_execution_reference(payload):
        return None
    return {
        "expected_net_bps": expected_net_bps,
        "stress_net_bps": stress_net_bps,
        "label_sha256": label_sha256,
    }


def _is_lower_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _contains_order_or_execution_reference(payload: dict[str, object]) -> bool:
    forbidden_keys = {
        "client_order_id",
        "exchange_order_id",
        "execution_id",
        "execution_report",
        "order",
        "order_id",
        "orders",
    }
    return any(key in payload for key in forbidden_keys)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _report_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

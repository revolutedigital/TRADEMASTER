"""Research metadata endpoints with no execution capabilities."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db, require_auth
from app.models.research_experiment import ResearchExperiment, ResearchShadowSignal
from app.repositories.research_experiment_repo import research_experiment_repository
from app.schemas.research_experiment import (
    CreateExperimentRequest,
    EvidenceGateStatusResponse,
    ExperimentReportResponse,
    ExperimentResponse,
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
from app.services.research.testnet_release_gate import evaluate_testnet_eligibility


router = APIRouter()


@router.get("/evidence-gate", response_model=EvidenceGateStatusResponse)
async def get_evidence_gate_status(
    _user: dict = Depends(require_auth),
) -> EvidenceGateStatusResponse:
    return _read_evidence_gate_status()


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

    evidence_status = _read_evidence_gate_status()
    shadow_signals = await research_experiment_repository.list_shadow_signals(
        db,
        experiment_id,
    )
    shadow_summary = _summarize_shadow_outcomes(shadow_signals)
    unresolved_failures = len(evidence_status.status_reasons)
    eligibility = evaluate_testnet_eligibility(
        experiment,
        book_evidence_contiguous_days=(
            evidence_status.book_evidence_gate.longest_complete_streak_days
            if evidence_status.artifact_available
            else 0
        ),
        prospective_shadow_days=shadow_summary["decision_days"],
        prospective_shadow_outcome_days=shadow_summary["outcome_days"],
        prospective_shadow_positive=shadow_summary["positive"],
        unresolved_failures=unresolved_failures,
        explicit_testnet_release=False,
    )
    return TestnetEligibilityResponse(
        experiment_id=experiment.id,
        experiment_status=experiment.status,
        eligible=eligibility.eligible,
        reasons=list(eligibility.reasons),
        book_evidence_contiguous_days=eligibility.book_evidence_contiguous_days,
        prospective_shadow_days=eligibility.prospective_shadow_days,
        prospective_shadow_outcome_days=eligibility.prospective_shadow_outcome_days,
        prospective_shadow_signal_count=len(shadow_signals),
        prospective_shadow_expected_mean_bps=shadow_summary["expected_mean_bps"],
        prospective_shadow_stress_mean_bps=shadow_summary["stress_mean_bps"],
        prospective_shadow_positive=eligibility.prospective_shadow_positive,
        unresolved_failures=unresolved_failures,
        explicit_testnet_release=False,
        release_request_required=True,
        evidence_artifact_available=evidence_status.artifact_available,
        order_submission_allowed=False,
        execution_authorization="none",
        safety=_safety(),
        generated_at=datetime.now(UTC),
    )


def _read_evidence_gate_status() -> EvidenceGateStatusResponse:
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
    return {
        "experiment_id": experiment.id,
        "status": experiment.status,
        "decision_reasons": json.loads(experiment.decision_reasons_json),
        "metrics": {},
        "artifact_sha256": None,
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


def _summarize_shadow_outcomes(signals: list[ResearchShadowSignal]) -> dict[str, object]:
    decision_days = {_utc_day(signal.decision_time) for signal in signals}
    outcome_days = set()
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
        expected_net_bps = _finite_float(payload.get("expected_net_bps"))
        stress_net_bps = _finite_float(payload.get("stress_net_bps"))
        if expected_net_bps is None or stress_net_bps is None:
            continue
        expected_values.append(expected_net_bps)
        stress_values.append(stress_net_bps)
        outcome_days.add(_utc_day(signal.decision_time))

    expected_mean = _mean(expected_values)
    stress_mean = _mean(stress_values)
    positive = (
        bool(decision_days)
        and outcome_days == decision_days
        and expected_mean is not None
        and stress_mean is not None
        and expected_mean > 0
        and stress_mean > 0
    )
    return {
        "decision_days": len(decision_days),
        "outcome_days": len(outcome_days),
        "expected_mean_bps": expected_mean,
        "stress_mean_bps": stress_mean,
        "positive": positive,
    }


def _utc_day(value: datetime):
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).date()


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None

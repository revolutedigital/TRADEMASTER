"""Research metadata endpoints with no execution capabilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.models.research_experiment import ResearchExperiment
from app.repositories.research_experiment_repo import research_experiment_repository
from app.schemas.research_experiment import (
    CreateExperimentRequest,
    ExperimentReportResponse,
    ExperimentResponse,
)
from app.services.data.research_registry import (
    BurnedDataConflict,
    ExperimentDefinition,
    FrozenExperimentError,
    PartitionDefinition,
    ResearchRegistryError,
    research_registry,
)


router = APIRouter()


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

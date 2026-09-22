"""Database operations for the immutable research experiment ledger."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchExperimentEvent,
    ResearchHypothesisAttempt,
    ResearchShadowSignal,
)


class ResearchExperimentRepository:
    """Small transactional repository; lifecycle rules stay in the service."""

    async def get(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        for_update: bool = False,
    ) -> ResearchExperiment | None:
        query = select(ResearchExperiment).where(ResearchExperiment.id == experiment_id)
        if for_update:
            query = query.with_for_update()
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list(
        self,
        db: AsyncSession,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ResearchExperiment]:
        query = select(ResearchExperiment)
        if status is not None:
            query = query.where(ResearchExperiment.status == status)
        result = await db.execute(query.order_by(ResearchExperiment.created_at.desc()).limit(limit))
        return list(result.scalars().all())

    async def list_partitions(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        for_update: bool = False,
    ) -> list[ResearchDataUse]:
        query = (
            select(ResearchDataUse)
            .where(ResearchDataUse.experiment_id == experiment_id)
            .order_by(ResearchDataUse.start_at, ResearchDataUse.id)
        )
        if for_update:
            query = query.with_for_update()
        result = await db.execute(query)
        return list(result.scalars().all())

    async def list_hypotheses(
        self,
        db: AsyncSession,
        experiment_id: str,
    ) -> list[ResearchHypothesisAttempt]:
        result = await db.execute(
            select(ResearchHypothesisAttempt)
            .where(ResearchHypothesisAttempt.experiment_id == experiment_id)
            .order_by(ResearchHypothesisAttempt.id)
        )
        return list(result.scalars().all())

    async def list_shadow_decision_times(
        self,
        db: AsyncSession,
        experiment_id: str,
    ) -> list[datetime]:
        result = await db.execute(
            select(ResearchShadowSignal.decision_time)
            .where(ResearchShadowSignal.experiment_id == experiment_id)
            .order_by(ResearchShadowSignal.decision_time)
        )
        return list(result.scalars().all())

    async def opened_uses_for_manifest(
        self,
        db: AsyncSession,
        manifest_sha256: str,
        *,
        excluding_experiment_id: str,
    ) -> list[ResearchDataUse]:
        result = await db.execute(
            select(ResearchDataUse).where(
                ResearchDataUse.manifest_sha256 == manifest_sha256,
                ResearchDataUse.opened_at.is_not(None),
                ResearchDataUse.experiment_id != excluding_experiment_id,
            )
        )
        return list(result.scalars().all())

    async def append_event(
        self,
        db: AsyncSession,
        event: ResearchExperimentEvent,
    ) -> None:
        db.add(event)
        await db.flush()


research_experiment_repository = ResearchExperimentRepository()

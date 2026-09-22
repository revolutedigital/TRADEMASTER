"""Database operations for the immutable research experiment ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchExperimentEvent,
    ResearchHypothesisAttempt,
    ResearchShadowSignal,
)


@dataclass(frozen=True)
class ResearchEventChainStatus:
    """Tamper-evident verification summary for one experiment event chain."""

    event_count: int
    verified: bool
    latest_event_sha256: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_count": self.event_count,
            "verified": self.verified,
            "latest_event_sha256": self.latest_event_sha256,
            "reasons": list(self.reasons),
        }


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

    async def list_shadow_signals(
        self,
        db: AsyncSession,
        experiment_id: str,
    ) -> list[ResearchShadowSignal]:
        result = await db.execute(
            select(ResearchShadowSignal)
            .where(ResearchShadowSignal.experiment_id == experiment_id)
            .order_by(ResearchShadowSignal.decision_time, ResearchShadowSignal.id)
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
        last_event = await self._latest_event(db, event.experiment_id, for_update=True)
        previous_event_sha256 = last_event.event_sha256 if last_event is not None else None
        if last_event is not None and not _is_sha256(previous_event_sha256):
            raise ValueError("Previous research event is missing a valid hash")
        event.previous_event_sha256 = previous_event_sha256
        event.event_sha256 = build_research_event_hash(
            experiment_id=event.experiment_id,
            kind=event.kind,
            payload_json=event.payload_json,
            occurred_at=event.occurred_at,
            previous_event_sha256=event.previous_event_sha256,
        )
        db.add(event)
        await db.flush()

    async def list_events(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        kind: str | None = None,
    ) -> list[ResearchExperimentEvent]:
        query = select(ResearchExperimentEvent).where(
            ResearchExperimentEvent.experiment_id == experiment_id
        )
        if kind is not None:
            query = query.where(ResearchExperimentEvent.kind == kind)
        result = await db.execute(
            query.order_by(ResearchExperimentEvent.occurred_at, ResearchExperimentEvent.id)
        )
        return list(result.scalars().all())

    async def event_chain_status(
        self,
        db: AsyncSession,
        experiment_id: str,
    ) -> ResearchEventChainStatus:
        events = await self.list_events(db, experiment_id)
        return verify_research_event_chain(events)

    async def _latest_event(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        for_update: bool = False,
    ) -> ResearchExperimentEvent | None:
        query = (
            select(ResearchExperimentEvent)
            .where(ResearchExperimentEvent.experiment_id == experiment_id)
            .order_by(ResearchExperimentEvent.occurred_at.desc(), ResearchExperimentEvent.id.desc())
            .limit(1)
        )
        if for_update:
            query = query.with_for_update()
        result = await db.execute(query)
        return result.scalar_one_or_none()


def build_research_event_hash(
    *,
    experiment_id: str,
    kind: str,
    payload_json: str,
    occurred_at: datetime,
    previous_event_sha256: str | None,
) -> str:
    """Return the stable hash for one persisted research event."""
    encoded = json.dumps(
        {
            "experiment_id": experiment_id,
            "kind": kind,
            "payload_json": payload_json,
            "occurred_at": _utc_iso(occurred_at),
            "previous_event_sha256": previous_event_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_research_event_chain(
    events: list[ResearchExperimentEvent],
) -> ResearchEventChainStatus:
    """Verify each event hash and its pointer to the previous event."""
    reasons: list[str] = []
    previous_event_sha256: str | None = None
    latest_event_sha256: str | None = None
    seen_hashes: set[str] = set()

    for index, event in enumerate(events):
        event_label = f"event_{event.id or index}"
        if event.previous_event_sha256 != previous_event_sha256:
            reasons.append(f"{event_label}_previous_hash_mismatch")
        if not _is_sha256(event.event_sha256):
            reasons.append(f"{event_label}_hash_missing")
            latest_event_sha256 = event.event_sha256
            previous_event_sha256 = event.event_sha256
            continue
        if event.event_sha256 in seen_hashes:
            reasons.append(f"{event_label}_hash_duplicate")
        seen_hashes.add(event.event_sha256)
        expected_event_sha256 = build_research_event_hash(
            experiment_id=event.experiment_id,
            kind=event.kind,
            payload_json=event.payload_json,
            occurred_at=event.occurred_at,
            previous_event_sha256=event.previous_event_sha256,
        )
        if event.event_sha256 != expected_event_sha256:
            reasons.append(f"{event_label}_hash_mismatch")
        latest_event_sha256 = event.event_sha256
        previous_event_sha256 = event.event_sha256

    return ResearchEventChainStatus(
        event_count=len(events),
        verified=not reasons,
        latest_event_sha256=latest_event_sha256,
        reasons=tuple(reasons),
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


research_experiment_repository = ResearchExperimentRepository()

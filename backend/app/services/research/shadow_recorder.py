"""Prospective hypothetical signal ledger with an explicit no-order boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchShadowSignal,
)


SHA256 = re.compile(r"^[a-f0-9]{64}$")


class ShadowRecorderError(ValueError):
    pass


class ResearchShadowRecorder:
    """Records what a frozen model would do, never what an exchange should do."""

    async def record(
        self,
        db: AsyncSession,
        *,
        experiment_id: str,
        decision_time: datetime,
        side: str,
        horizon_seconds: int,
        probability: float,
        threshold: float,
        model_sha256: str,
        feature_vector: dict[str, float],
    ) -> ResearchShadowSignal:
        experiment = await db.get(ResearchExperiment, experiment_id)
        if experiment is None:
            raise LookupError("research experiment was not found")
        if experiment.status != "FROZEN":
            raise ShadowRecorderError("shadow recording requires a FROZEN experiment")
        prospective = (
            await db.execute(
                select(ResearchDataUse).where(
                    ResearchDataUse.experiment_id == experiment_id,
                    ResearchDataUse.role == "PROSPECTIVE_SHADOW",
                )
            )
        ).scalar_one_or_none()
        if prospective is None or prospective.opened_at is None:
            raise ShadowRecorderError(
                "prospective partition must be explicitly opened before shadow recording"
            )
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ShadowRecorderError("shadow side must be BUY or SELL")
        if horizon_seconds not in {120, 300}:
            raise ShadowRecorderError("shadow horizon must be 120 or 300 seconds")
        if decision_time.tzinfo is None:
            raise ShadowRecorderError("decision_time must be timezone-aware")
        normalized_decision_time = decision_time.astimezone(UTC)
        partition_start = _normalize_utc(prospective.start_at)
        partition_end = _normalize_utc(prospective.end_at)
        if not (partition_start <= normalized_decision_time < partition_end):
            raise ShadowRecorderError(
                "decision_time must be inside the opened prospective shadow partition"
            )
        if normalized_decision_time + timedelta(seconds=horizon_seconds) > partition_end:
            raise ShadowRecorderError(
                "shadow horizon must finish inside the opened prospective shadow partition"
            )
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (probability, threshold)):
            raise ShadowRecorderError("probability and threshold must be in [0, 1]")
        if not SHA256.fullmatch(model_sha256):
            raise ShadowRecorderError("model_sha256 must be a lowercase SHA-256")
        if not feature_vector or not all(
            math.isfinite(float(value)) for value in feature_vector.values()
        ):
            raise ShadowRecorderError("feature vector must contain finite values")
        feature_sha256 = _sha256(feature_vector)
        signal = ResearchShadowSignal(
            experiment_id=experiment_id,
            decision_time=decision_time.astimezone(UTC),
            recorded_at=datetime.now(UTC),
            side=normalized_side,
            horizon_seconds=horizon_seconds,
            probability=probability,
            threshold=threshold,
            would_enter=probability >= threshold,
            model_sha256=model_sha256,
            feature_vector_sha256=feature_sha256,
        )
        db.add(signal)
        await db.flush()
        return signal

    async def record_outcome(
        self,
        db: AsyncSession,
        *,
        signal_id: int,
        expected_net_bps: float,
        stress_net_bps: float,
        label_sha256: str,
    ) -> ResearchShadowSignal:
        signal = await db.get(ResearchShadowSignal, signal_id)
        if signal is None:
            raise LookupError("research shadow signal was not found")
        if signal.outcome_json is not None:
            raise ShadowRecorderError("shadow outcome is immutable once recorded")
        if not math.isfinite(expected_net_bps) or not math.isfinite(stress_net_bps):
            raise ShadowRecorderError("shadow outcome bps must be finite")
        if not SHA256.fullmatch(label_sha256):
            raise ShadowRecorderError("label_sha256 must be a lowercase SHA-256")
        signal.outcome_json = _canonical_json(
            {
                "expected_net_bps": expected_net_bps,
                "stress_net_bps": stress_net_bps,
                "label_sha256": label_sha256,
                "recorded_at": datetime.now(UTC).isoformat(),
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
            }
        )
        await db.flush()
        return signal


def _sha256(value: Any) -> str:
    encoded = _canonical_json(value).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


research_shadow_recorder = ResearchShadowRecorder()

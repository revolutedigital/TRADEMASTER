"""Prospective hypothetical signal ledger with an explicit no-order boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchExperimentEvent,
    ResearchShadowSignal,
)
from app.repositories.research_experiment_repo import (
    ResearchExperimentRepository,
    research_experiment_repository,
)


SHA256 = re.compile(r"^[a-f0-9]{64}$")
MIN_PROSPECTIVE_SHADOW_DURATION = timedelta(days=20)
MAX_PROSPECTIVE_SHADOW_DURATION = timedelta(days=30)


class ShadowRecorderError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedShadowSignalInput:
    """Normalized signal inputs after the research-only shadow boundary check."""

    decision_time: datetime
    side: str
    horizon_seconds: int
    probability: float
    threshold: float
    model_sha256: str
    feature_vector_sha256: str


class ResearchShadowRecorder:
    """Records what a frozen model would do, never what an exchange should do."""

    def __init__(
        self,
        repository: ResearchExperimentRepository | None = None,
    ) -> None:
        self._repository = repository or research_experiment_repository

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
        validated = await validate_shadow_signal_recording(
            db,
            experiment_id=experiment_id,
            decision_time=decision_time,
            side=side,
            horizon_seconds=horizon_seconds,
            probability=probability,
            threshold=threshold,
            model_sha256=model_sha256,
            feature_vector=feature_vector,
        )
        signal = ResearchShadowSignal(
            experiment_id=experiment_id,
            decision_time=validated.decision_time,
            recorded_at=datetime.now(UTC),
            side=validated.side,
            horizon_seconds=validated.horizon_seconds,
            probability=validated.probability,
            threshold=validated.threshold,
            would_enter=validated.probability >= validated.threshold,
            model_sha256=validated.model_sha256,
            feature_vector_sha256=validated.feature_vector_sha256,
        )
        db.add(signal)
        await db.flush()
        await self._repository.append_event(
            db,
            ResearchExperimentEvent(
                experiment_id=experiment_id,
                kind="SHADOW_SIGNAL_RECORDED",
                payload_json=_canonical_json(shadow_signal_event_payload(signal)),
                occurred_at=_normalize_utc(signal.recorded_at),
            ),
        )
        return signal

    async def record_outcome(
        self,
        db: AsyncSession,
        *,
        signal_id: int,
        expected_net_bps: float,
        stress_net_bps: float,
        label_sha256: str,
        now: datetime | None = None,
    ) -> ResearchShadowSignal:
        signal = await db.get(ResearchShadowSignal, signal_id)
        if signal is None:
            raise LookupError("research shadow signal was not found")
        if signal.outcome_json is not None:
            raise ShadowRecorderError("shadow outcome is immutable once recorded")
        recorded_at = max(
            _normalize_utc(now or datetime.now(UTC)),
            _normalize_utc(signal.recorded_at),
        )
        horizon_end = _normalize_utc(signal.decision_time) + timedelta(
            seconds=signal.horizon_seconds
        )
        if recorded_at < horizon_end:
            raise ShadowRecorderError(
                "shadow outcome cannot be recorded before the signal horizon matures"
            )
        if not math.isfinite(expected_net_bps) or not math.isfinite(stress_net_bps):
            raise ShadowRecorderError("shadow outcome bps must be finite")
        if not SHA256.fullmatch(label_sha256):
            raise ShadowRecorderError("label_sha256 must be a lowercase SHA-256")
        outcome_payload = {
            "expected_net_bps": expected_net_bps,
            "stress_net_bps": stress_net_bps,
            "label_sha256": label_sha256,
            "recorded_at": recorded_at.isoformat(),
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
        }
        signal.outcome_json = _canonical_json(outcome_payload)
        await db.flush()
        await self._repository.append_event(
            db,
            ResearchExperimentEvent(
                experiment_id=signal.experiment_id,
                kind="SHADOW_OUTCOME_RECORDED",
                payload_json=_canonical_json(shadow_outcome_event_payload(signal, outcome_payload)),
                occurred_at=recorded_at,
            ),
        )
        return signal


async def validate_shadow_signal_recording(
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
) -> ValidatedShadowSignalInput:
    """Validate a prospective shadow signal without mutating the database."""
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
    partition_duration = partition_end - partition_start
    if not (
        MIN_PROSPECTIVE_SHADOW_DURATION
        <= partition_duration
        <= MAX_PROSPECTIVE_SHADOW_DURATION
    ):
        raise ShadowRecorderError("prospective shadow partition must be 20 to 30 days")
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
    try:
        feature_values_are_finite = all(
            math.isfinite(float(value)) for value in feature_vector.values()
        )
    except (TypeError, ValueError) as error:
        raise ShadowRecorderError("feature vector must contain finite values") from error
    if not feature_vector or not feature_values_are_finite:
        raise ShadowRecorderError("feature vector must contain finite values")
    return ValidatedShadowSignalInput(
        decision_time=normalized_decision_time,
        side=normalized_side,
        horizon_seconds=horizon_seconds,
        probability=probability,
        threshold=threshold,
        model_sha256=model_sha256,
        feature_vector_sha256=_sha256(feature_vector),
    )


def shadow_signal_event_payload(signal: ResearchShadowSignal) -> dict[str, object]:
    """Stable event payload proving a shadow decision row was not execution intent."""
    return {
        "signal_id": signal.id,
        "decision_time": _normalize_utc(signal.decision_time).isoformat(),
        "recorded_at": _normalize_utc(signal.recorded_at).isoformat(),
        "side": signal.side,
        "horizon_seconds": signal.horizon_seconds,
        "probability": signal.probability,
        "threshold": signal.threshold,
        "would_enter": signal.would_enter,
        "model_sha256": signal.model_sha256,
        "feature_vector_sha256": signal.feature_vector_sha256,
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def shadow_outcome_event_payload(
    signal: ResearchShadowSignal,
    outcome_payload: dict[str, object],
) -> dict[str, object]:
    """Stable event payload proving a settled outcome is replay-only evidence."""
    return {
        "signal_id": signal.id,
        "decision_time": _normalize_utc(signal.decision_time).isoformat(),
        "horizon_seconds": signal.horizon_seconds,
        "expected_net_bps": outcome_payload.get("expected_net_bps"),
        "stress_net_bps": outcome_payload.get("stress_net_bps"),
        "label_sha256": outcome_payload.get("label_sha256"),
        "recorded_at": outcome_payload.get("recorded_at"),
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


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

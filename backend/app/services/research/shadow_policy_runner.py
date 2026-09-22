"""Run frozen research policies into the shadow ledger, never into an exchange."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import ResearchShadowSignal
from app.services.research.shadow_recorder import ResearchShadowRecorder, research_shadow_recorder
from app.services.research.top_p_model import (
    predict_frozen_top_p_probability,
    verify_frozen_top_p_policy,
)


class ShadowPolicyRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenTopPShadowDecision:
    """One scored research-only decision selected by a frozen top-p policy."""

    decision_time: datetime
    side: str
    horizon_seconds: int
    probability: float
    threshold: float
    would_enter: bool
    model_sha256: str
    feature_vector_sha256: str
    feature_vector: dict[str, float]


@dataclass(frozen=True)
class FrozenTopPShadowSelection:
    """Pure scoring output before anything is written to the database."""

    model_sha256: str
    scored_rows: int
    selected_count: int
    decisions: tuple[FrozenTopPShadowDecision, ...]
    order_submission_allowed: bool = False
    execution_authorization: str = "none"


@dataclass(frozen=True)
class ShadowPolicyBatchResult:
    """Database append result for a frozen top-p shadow batch."""

    model_sha256: str
    scored_rows: int
    selected_count: int
    recorded_count: int
    skipped_existing_count: int
    order_submission_allowed: bool = False
    execution_authorization: str = "none"


async def record_frozen_top_p_shadow_signal(
    db: AsyncSession,
    *,
    experiment_id: str,
    decision_time: datetime,
    side: str,
    artifact: dict[str, Any],
    feature_vector: dict[str, float],
    recorder: ResearchShadowRecorder = research_shadow_recorder,
) -> ResearchShadowSignal:
    """Score one frozen top-p artifact and append the hypothetical signal."""
    try:
        model_sha256 = verify_frozen_top_p_policy(artifact)
        probability = predict_frozen_top_p_probability(artifact, feature_vector)
        threshold = float(artifact["probability_threshold"])
        horizon_seconds = int(artifact["horizon_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ShadowPolicyRunnerError(str(error)) from error
    return await recorder.record(
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


def score_frozen_top_p_shadow_frame(
    frame: pd.DataFrame,
    *,
    artifact: dict[str, Any],
    include_non_entries: bool = False,
) -> FrozenTopPShadowSelection:
    """Score a parquet-like frame without mutating the database.

    By default this returns only decisions that pass the frozen top-p threshold,
    because shadow performance is measured on hypothetical entries rather than
    on every no-trade polling tick.
    """
    try:
        model_sha256 = verify_frozen_top_p_policy(artifact)
        horizon_seconds = int(artifact["horizon_seconds"])
        threshold = float(artifact["probability_threshold"])
        feature_columns = tuple(str(column) for column in artifact["feature_columns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ShadowPolicyRunnerError(str(error)) from error

    required_columns = {"decision_time_ms", "side", "horizon_seconds", *feature_columns}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ShadowPolicyRunnerError(f"shadow frame is missing columns: {sorted(missing)}")

    horizon_frame = frame[frame["horizon_seconds"] == horizon_seconds].sort_values(
        ["decision_time_ms", "side"],
        kind="stable",
    )
    if horizon_frame.empty:
        raise ShadowPolicyRunnerError(f"shadow frame has no rows for horizon {horizon_seconds}")

    decisions: list[FrozenTopPShadowDecision] = []
    for row in horizon_frame.itertuples(index=False):
        row_dict = row._asdict()
        side = str(row_dict["side"]).upper()
        if side not in {"BUY", "SELL"}:
            raise ShadowPolicyRunnerError("shadow side must be BUY or SELL")
        feature_vector = {
            column: float(row_dict[column])
            for column in feature_columns
        }
        try:
            probability = predict_frozen_top_p_probability(artifact, feature_vector)
        except (KeyError, TypeError, ValueError) as error:
            raise ShadowPolicyRunnerError(str(error)) from error
        would_enter = probability >= threshold
        if not would_enter and not include_non_entries:
            continue
        decisions.append(
            FrozenTopPShadowDecision(
                decision_time=datetime.fromtimestamp(
                    int(row_dict["decision_time_ms"]) / 1000,
                    tz=UTC,
                ),
                side=side,
                horizon_seconds=horizon_seconds,
                probability=probability,
                threshold=threshold,
                would_enter=would_enter,
                model_sha256=model_sha256,
                feature_vector_sha256=_sha256(feature_vector),
                feature_vector=feature_vector,
            )
        )

    return FrozenTopPShadowSelection(
        model_sha256=model_sha256,
        scored_rows=len(horizon_frame),
        selected_count=len(decisions),
        decisions=tuple(decisions),
    )


async def record_frozen_top_p_shadow_batch(
    db: AsyncSession,
    *,
    experiment_id: str,
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    include_non_entries: bool = False,
    limit: int | None = None,
    recorder: ResearchShadowRecorder = research_shadow_recorder,
) -> ShadowPolicyBatchResult:
    """Score and append a deterministic shadow batch, skipping existing signals."""
    if limit is not None and limit <= 0:
        raise ShadowPolicyRunnerError("batch limit must be positive")
    selection = score_frozen_top_p_shadow_frame(
        frame,
        artifact=artifact,
        include_non_entries=include_non_entries,
    )
    decisions = selection.decisions[:limit] if limit is not None else selection.decisions
    existing = await _existing_shadow_keys(
        db,
        experiment_id=experiment_id,
        horizon_seconds=int(artifact["horizon_seconds"]),
    )
    recorded_count = 0
    skipped_existing_count = 0
    for decision in decisions:
        key = (
            int(decision.decision_time.timestamp() * 1000),
            decision.side,
            decision.horizon_seconds,
        )
        if key in existing:
            skipped_existing_count += 1
            continue
        await record_frozen_top_p_shadow_signal(
            db,
            experiment_id=experiment_id,
            decision_time=decision.decision_time,
            side=decision.side,
            artifact=artifact,
            feature_vector=decision.feature_vector,
            recorder=recorder,
        )
        existing.add(key)
        recorded_count += 1
    return ShadowPolicyBatchResult(
        model_sha256=selection.model_sha256,
        scored_rows=selection.scored_rows,
        selected_count=selection.selected_count,
        recorded_count=recorded_count,
        skipped_existing_count=skipped_existing_count,
    )


async def _existing_shadow_keys(
    db: AsyncSession,
    *,
    experiment_id: str,
    horizon_seconds: int,
) -> set[tuple[int, str, int]]:
    result = await db.execute(
        select(
            ResearchShadowSignal.decision_time,
            ResearchShadowSignal.side,
            ResearchShadowSignal.horizon_seconds,
        ).where(
            ResearchShadowSignal.experiment_id == experiment_id,
            ResearchShadowSignal.horizon_seconds == horizon_seconds,
        )
    )
    return {
        (
            _decision_time_key_ms(decision_time),
            side,
            signal_horizon_seconds,
        )
        for decision_time, side, signal_horizon_seconds in result.all()
    }


def _decision_time_key_ms(value: datetime) -> int:
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return int(normalized.timestamp() * 1000)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

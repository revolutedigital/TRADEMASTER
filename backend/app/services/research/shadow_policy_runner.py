"""Run frozen research policies into the shadow ledger, never into an exchange."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import ResearchShadowSignal
from app.services.research.shadow_recorder import ResearchShadowRecorder, research_shadow_recorder
from app.services.research.top_p_model import (
    predict_frozen_top_p_probability,
    verify_frozen_top_p_policy,
)


class ShadowPolicyRunnerError(ValueError):
    pass


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

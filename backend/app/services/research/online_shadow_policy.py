"""Event-driven frozen top-p scoring over causal microstructure events."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from app.schemas.microstructure import MicrostructureEvent
from app.services.research.microstructure_features import (
    TRADE_WINDOWS_SECONDS,
    CausalMicrostructureFeatureEngine,
    feature_vector_for_side,
)
from app.services.research.shadow_policy_runner import (
    FrozenTopPShadowDecision,
    FrozenTopPShadowSelection,
    ShadowPolicyRunnerError,
)
from app.services.research.top_p_model import (
    predict_frozen_top_p_probability,
    verify_frozen_top_p_policy,
)


def score_online_frozen_top_p_shadow_events(
    events: Iterable[MicrostructureEvent],
    *,
    artifact: dict[str, object],
    decision_times_ms: Sequence[int],
    sides: Sequence[str] = ("BUY", "SELL"),
    include_non_entries: bool = False,
    windows_seconds: tuple[int, ...] = TRADE_WINDOWS_SECONDS,
) -> FrozenTopPShadowSelection:
    """Score causal online snapshots without mutating the research ledger."""
    try:
        model_sha256 = verify_frozen_top_p_policy(artifact)
        horizon_seconds = int(artifact["horizon_seconds"])
        threshold = float(artifact["probability_threshold"])
        feature_columns = tuple(str(column) for column in artifact["feature_columns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ShadowPolicyRunnerError(str(error)) from error
    if any(side.upper() not in {"BUY", "SELL"} for side in sides):
        raise ShadowPolicyRunnerError("shadow side must be BUY or SELL")
    decisions_ms = tuple(int(decision_time_ms) for decision_time_ms in decision_times_ms)
    if any(later < earlier for earlier, later in zip(decisions_ms, decisions_ms[1:])):
        raise ShadowPolicyRunnerError("decision times must be monotonic")

    engine = CausalMicrostructureFeatureEngine(windows_seconds=windows_seconds)
    event_iterator = iter(events)
    next_event = next(event_iterator, None)
    scored_decisions: list[FrozenTopPShadowDecision] = []
    for decision_time_ms in decisions_ms:
        while next_event is not None and _event_time_ms(next_event) <= decision_time_ms:
            try:
                engine.consume(next_event)
            except ValueError as error:
                raise ShadowPolicyRunnerError(str(error)) from error
            next_event = next(event_iterator, None)
        snapshot = engine.snapshot(decision_time_ms)
        for side in sides:
            normalized_side = side.upper()
            try:
                feature_vector = feature_vector_for_side(
                    snapshot.values,
                    normalized_side,
                    feature_columns,
                )
                probability = predict_frozen_top_p_probability(artifact, feature_vector)
            except (KeyError, TypeError, ValueError) as error:
                raise ShadowPolicyRunnerError(str(error)) from error
            would_enter = probability >= threshold
            if not would_enter and not include_non_entries:
                continue
            scored_decisions.append(
                FrozenTopPShadowDecision(
                    decision_time=datetime.fromtimestamp(
                        snapshot.decision_time_ms / 1000,
                        tz=UTC,
                    ),
                    side=normalized_side,
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
        scored_rows=len(decisions_ms) * len(sides),
        selected_count=len(scored_decisions),
        decisions=tuple(scored_decisions),
    )


def _event_time_ms(event: MicrostructureEvent) -> int:
    return int(event.event_time.timestamp() * 1000)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

"""Deterministic settlement for research-only shadow signals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import ResearchShadowSignal
from app.services.backtest.event_replay import OrderSide
from app.services.backtest.trailing_portfolio import (
    HistoricalTrailingSimulator,
    ManagedTrade,
    TrailingPolicy,
)
from app.services.research.shadow_recorder import ResearchShadowRecorder, research_shadow_recorder


@dataclass(frozen=True)
class ShadowOutcomeSettlement:
    """Outcome evidence derived from replay, not from an exchange order."""

    signal_id: int
    would_enter: bool
    expected_net_bps: float
    stress_net_bps: float
    label_sha256: str
    policy_name: str

    def to_recorder_kwargs(self) -> dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "expected_net_bps": self.expected_net_bps,
            "stress_net_bps": self.stress_net_bps,
            "label_sha256": self.label_sha256,
        }


def settle_shadow_signal(
    signal: ResearchShadowSignal,
    *,
    simulator: HistoricalTrailingSimulator,
    policy: TrailingPolicy,
) -> ShadowOutcomeSettlement:
    """Replay one hypothetical signal and return immutable outcome evidence."""
    if not signal.would_enter:
        return _no_entry_settlement(signal, policy=policy)

    managed_trade = simulator.simulate(
        decision_time_ms=_decision_time_ms(signal),
        side=OrderSide(signal.side),
        horizon_seconds=signal.horizon_seconds,
        probability=signal.probability,
        policy=policy,
    )
    label_sha256 = _stable_sha256(
        {
            "kind": "shadow_trailing_replay_v1",
            "signal": _signal_fingerprint(signal),
            "policy": asdict(policy),
            "managed_trade": managed_trade.to_dict(),
            "safety": _safety(),
        }
    )
    return ShadowOutcomeSettlement(
        signal_id=signal.id,
        would_enter=True,
        expected_net_bps=managed_trade.expected_net_bps,
        stress_net_bps=managed_trade.stress_net_bps,
        label_sha256=label_sha256,
        policy_name=policy.name,
    )


def settle_shadow_signals(
    signals: list[ResearchShadowSignal],
    *,
    simulator: HistoricalTrailingSimulator,
    policy: TrailingPolicy,
) -> tuple[ShadowOutcomeSettlement, ...]:
    return tuple(
        settle_shadow_signal(signal, simulator=simulator, policy=policy) for signal in signals
    )


async def settle_pending_shadow_outcomes(
    db: AsyncSession,
    *,
    experiment_id: str,
    simulator: HistoricalTrailingSimulator,
    policy: TrailingPolicy,
    now: datetime | None = None,
    limit: int = 1_000,
    recorder: ResearchShadowRecorder = research_shadow_recorder,
) -> tuple[ShadowOutcomeSettlement, ...]:
    """Record replay outcomes for mature pending shadow signals only."""
    mature_signals = await list_mature_pending_shadow_signals(
        db,
        experiment_id=experiment_id,
        now=now,
        limit=limit,
    )
    settlements: list[ShadowOutcomeSettlement] = []
    for signal in mature_signals:
        settlement = settle_shadow_signal(signal, simulator=simulator, policy=policy)
        await recorder.record_outcome(db, **settlement.to_recorder_kwargs())
        settlements.append(settlement)
    return tuple(settlements)


async def list_mature_pending_shadow_signals(
    db: AsyncSession,
    *,
    experiment_id: str,
    now: datetime | None = None,
    limit: int = 1_000,
) -> tuple[ResearchShadowSignal, ...]:
    """Return pending shadow signals whose full replay horizon has elapsed."""
    if limit <= 0:
        raise ValueError("settlement limit must be positive")
    settlement_time = _normalize_utc(now or datetime.now(UTC))
    result = await db.execute(
        select(ResearchShadowSignal)
        .where(
            ResearchShadowSignal.experiment_id == experiment_id,
            ResearchShadowSignal.outcome_json.is_(None),
        )
        .order_by(ResearchShadowSignal.decision_time, ResearchShadowSignal.id)
        .limit(limit)
    )
    return tuple(
        signal for signal in result.scalars().all() if _is_mature(signal, settlement_time)
    )


def _no_entry_settlement(
    signal: ResearchShadowSignal,
    *,
    policy: TrailingPolicy,
) -> ShadowOutcomeSettlement:
    label_sha256 = _stable_sha256(
        {
            "kind": "shadow_no_entry_v1",
            "signal": _signal_fingerprint(signal),
            "policy": asdict(policy),
            "outcome": {
                "expected_net_bps": 0.0,
                "stress_net_bps": 0.0,
            },
            "safety": _safety(),
        }
    )
    return ShadowOutcomeSettlement(
        signal_id=signal.id,
        would_enter=False,
        expected_net_bps=0.0,
        stress_net_bps=0.0,
        label_sha256=label_sha256,
        policy_name=policy.name,
    )


def _signal_fingerprint(signal: ResearchShadowSignal) -> dict[str, object]:
    return {
        "experiment_id": signal.experiment_id,
        "decision_time": signal.decision_time.isoformat(),
        "side": signal.side,
        "horizon_seconds": signal.horizon_seconds,
        "probability": signal.probability,
        "threshold": signal.threshold,
        "would_enter": signal.would_enter,
        "model_sha256": signal.model_sha256,
        "feature_vector_sha256": signal.feature_vector_sha256,
    }


def _decision_time_ms(signal: ResearchShadowSignal) -> int:
    return int(_normalize_utc(signal.decision_time).timestamp() * 1000)


def _is_mature(signal: ResearchShadowSignal, now: datetime) -> bool:
    return _normalize_utc(signal.decision_time) + timedelta(seconds=signal.horizon_seconds) <= now


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _stable_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safety() -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }

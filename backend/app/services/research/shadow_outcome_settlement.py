"""Deterministic settlement for research-only shadow signals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from app.models.research_experiment import ResearchShadowSignal
from app.services.backtest.event_replay import OrderSide
from app.services.backtest.trailing_portfolio import (
    HistoricalTrailingSimulator,
    ManagedTrade,
    TrailingPolicy,
)


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
    return int(signal.decision_time.timestamp() * 1000)


def _stable_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safety() -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }

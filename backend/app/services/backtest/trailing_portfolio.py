"""Exact trade-path trailing policies with one-position portfolio constraints."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from app.services.backtest.event_replay import OrderSide


class ExitReason(StrEnum):
    INITIAL_STOP = "INITIAL_STOP"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    TRAILING_STOP = "TRAILING_STOP"
    PROFIT_TARGET = "PROFIT_TARGET"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class TrailingPolicy:
    name: str
    initial_stop_bps: float
    breakeven_activation_bps: float
    breakeven_lock_bps: float
    trailing_activation_bps: float
    trailing_distance_bps: float
    partial_take_profit_bps: float | None = None
    partial_take_fraction: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.initial_stop_bps,
            self.breakeven_activation_bps,
            self.breakeven_lock_bps,
            self.trailing_activation_bps,
            self.trailing_distance_bps,
        )
        if not self.name or not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("trailing policy values must be named, finite, and positive")
        if self.breakeven_activation_bps < self.breakeven_lock_bps:
            raise ValueError("breakeven cannot lock more than its activation")
        if self.trailing_activation_bps < self.breakeven_activation_bps:
            raise ValueError("trailing cannot activate before breakeven")
        if self.partial_take_profit_bps is None:
            if self.partial_take_fraction != 0:
                raise ValueError("partial fraction requires a take-profit threshold")
        elif self.partial_take_profit_bps <= 0 or not 0 < self.partial_take_fraction <= 1:
            raise ValueError("partial take-profit requires a positive threshold and fraction")


V1_TRAILING_POLICIES = (
    TrailingPolicy("tight", 20, 20, 12, 24, 12),
    TrailingPolicy("balanced", 20, 24, 12, 30, 16),
    TrailingPolicy("wide", 30, 24, 12, 36, 16),
    TrailingPolicy("hybrid_half", 20, 24, 12, 30, 16, 24, 0.5),
    TrailingPolicy("fixed_24", 20, 24, 12, 30, 16, 24, 1.0),
)


@dataclass(frozen=True)
class ManagedTrade:
    decision_time_ms: int
    entry_time_ms: int
    exit_time_ms: int
    side: OrderSide
    horizon_seconds: int
    probability: float
    entry_price: float
    exit_price: float
    exit_reason: ExitReason
    gross_bps: float
    expected_net_bps: float
    stress_net_bps: float
    maximum_favorable_bps: float
    maximum_adverse_bps: float
    partial_take_fraction: float

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["side"] = self.side.value
        row["exit_reason"] = self.exit_reason.value
        return row


class HistoricalTrailingSimulator:
    def __init__(
        self,
        event_times_ms: np.ndarray,
        prices: np.ndarray,
        *,
        latency_ms: int = 100,
        expected_round_trip_bps: float = 12,
        stress_round_trip_bps: float = 24,
    ) -> None:
        self._times = np.asarray(event_times_ms, dtype=np.int64)
        self._prices = np.asarray(prices, dtype=np.float64)
        if len(self._times) != len(self._prices) or len(self._times) < 2:
            raise ValueError("historical event arrays must have equal non-trivial length")
        if (np.diff(self._times) < 0).any():
            raise ValueError("historical events must be chronological")
        if not np.isfinite(self._prices).all() or (self._prices <= 0).any():
            raise ValueError("historical prices must be finite and positive")
        if latency_ms < 0:
            raise ValueError("latency cannot be negative")
        if min(expected_round_trip_bps, stress_round_trip_bps) < 0:
            raise ValueError("costs cannot be negative")
        if stress_round_trip_bps < expected_round_trip_bps:
            raise ValueError("stress cost cannot be below expected cost")
        self._latency_ms = latency_ms
        self._expected_cost = expected_round_trip_bps
        self._stress_cost = stress_round_trip_bps

    def simulate(
        self,
        *,
        decision_time_ms: int,
        side: OrderSide,
        horizon_seconds: int,
        probability: float,
        policy: TrailingPolicy,
    ) -> ManagedTrade:
        entry_index = int(
            np.searchsorted(self._times, decision_time_ms + self._latency_ms, side="left")
        )
        end_index = int(
            np.searchsorted(
                self._times,
                decision_time_ms + horizon_seconds * 1000,
                side="right",
            )
        )
        if entry_index >= len(self._times) or entry_index >= end_index:
            raise ValueError("no executable event path for signal")
        entry_price = float(self._prices[entry_index])
        direction = 1 if side == OrderSide.BUY else -1
        best_price = entry_price
        initial_stop = entry_price * math.exp(-direction * policy.initial_stop_bps / 10_000)
        active_stop = initial_stop
        stop_reason = ExitReason.INITIAL_STOP
        maximum_favorable = 0.0
        maximum_adverse = 0.0
        exit_index = end_index - 1
        exit_price = float(self._prices[exit_index])
        exit_reason = ExitReason.TIMEOUT
        realized_gross_bps = 0.0
        remaining_fraction = 1.0
        partial_taken = False

        for index in range(entry_index + 1, end_index):
            price = float(self._prices[index])
            stop_touched = price <= active_stop if side == OrderSide.BUY else price >= active_stop
            if stop_touched:
                exit_index = index
                exit_price = (
                    min(active_stop, price) if side == OrderSide.BUY else max(active_stop, price)
                )
                exit_reason = stop_reason
                break

            signed_move_bps = direction * math.log(price / entry_price) * 10_000
            maximum_favorable = max(maximum_favorable, signed_move_bps)
            maximum_adverse = min(maximum_adverse, signed_move_bps)
            if (side == OrderSide.BUY and price > best_price) or (
                side == OrderSide.SELL and price < best_price
            ):
                best_price = price

            if (
                not partial_taken
                and policy.partial_take_profit_bps is not None
                and signed_move_bps >= policy.partial_take_profit_bps
            ):
                realized_gross_bps += policy.partial_take_fraction * policy.partial_take_profit_bps
                remaining_fraction -= policy.partial_take_fraction
                partial_taken = True
                if remaining_fraction <= 1e-12:
                    exit_index = index
                    exit_price = entry_price * math.exp(
                        direction * policy.partial_take_profit_bps / 10_000
                    )
                    exit_reason = ExitReason.PROFIT_TARGET
                    break

            if maximum_favorable >= policy.breakeven_activation_bps:
                breakeven_stop = entry_price * math.exp(
                    direction * policy.breakeven_lock_bps / 10_000
                )
                active_stop = _improve_stop(active_stop, breakeven_stop, side)
                stop_reason = ExitReason.BREAKEVEN_STOP
            if maximum_favorable >= policy.trailing_activation_bps:
                trailing_stop = best_price * math.exp(
                    -direction * policy.trailing_distance_bps / 10_000
                )
                improved = _improve_stop(active_stop, trailing_stop, side)
                if improved != active_stop:
                    active_stop = improved
                    stop_reason = ExitReason.TRAILING_STOP

        final_leg_bps = direction * math.log(exit_price / entry_price) * 10_000
        gross_bps = realized_gross_bps + remaining_fraction * final_leg_bps
        maximum_favorable = max(maximum_favorable, final_leg_bps)
        maximum_adverse = min(maximum_adverse, final_leg_bps)
        return ManagedTrade(
            decision_time_ms=decision_time_ms,
            entry_time_ms=int(self._times[entry_index]),
            exit_time_ms=int(self._times[exit_index]),
            side=side,
            horizon_seconds=horizon_seconds,
            probability=probability,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            gross_bps=gross_bps,
            expected_net_bps=gross_bps - self._expected_cost,
            stress_net_bps=gross_bps - self._stress_cost,
            maximum_favorable_bps=maximum_favorable,
            maximum_adverse_bps=maximum_adverse,
            partial_take_fraction=1.0 - remaining_fraction,
        )


def replay_one_position_portfolio(
    predictions: pd.DataFrame,
    simulator: HistoricalTrailingSimulator,
    *,
    tail_fraction: float,
    policy: TrailingPolicy,
) -> pd.DataFrame:
    threshold_column = f"threshold_top_{tail_fraction:.0%}"
    required = {
        "decision_time_ms",
        "side",
        "horizon_seconds",
        "probability",
        threshold_column,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    selected = predictions[predictions["probability"] >= predictions[threshold_column]].sort_values(
        ["decision_time_ms", "probability"], ascending=[True, False], kind="stable"
    )
    selected = selected.drop_duplicates("decision_time_ms", keep="first")
    trades: list[dict[str, object]] = []
    position_exit_ms = -1
    for signal in selected.itertuples(index=False):
        decision_time_ms = int(signal.decision_time_ms)
        if decision_time_ms <= position_exit_ms:
            continue
        trade = simulator.simulate(
            decision_time_ms=decision_time_ms,
            side=OrderSide(signal.side),
            horizon_seconds=int(signal.horizon_seconds),
            probability=float(signal.probability),
            policy=policy,
        )
        trades.append(trade.to_dict())
        position_exit_ms = trade.exit_time_ms
    return pd.DataFrame(trades)


def summarize_portfolio(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trade_count": 0,
            "distinct_days": 0,
            "expected_mean_bps": 0.0,
            "stress_mean_bps": 0.0,
            "expected_total_log_return_bps": 0.0,
            "stress_total_log_return_bps": 0.0,
            "expected_win_rate": 0.0,
            "maximum_drawdown_bps": 0.0,
        }
    expected = trades["expected_net_bps"].to_numpy(dtype=np.float64)
    stress = trades["stress_net_bps"].to_numpy(dtype=np.float64)
    cumulative = np.cumsum(expected)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdowns = np.concatenate(([0.0], cumulative)) - peaks
    days = pd.to_datetime(trades["entry_time_ms"], unit="ms", utc=True).dt.date
    return {
        "trade_count": len(trades),
        "distinct_days": int(days.nunique()),
        "expected_mean_bps": float(expected.mean()),
        "stress_mean_bps": float(stress.mean()),
        "expected_total_log_return_bps": float(expected.sum()),
        "stress_total_log_return_bps": float(stress.sum()),
        "expected_win_rate": float((expected > 0).mean()),
        "maximum_drawdown_bps": float(-drawdowns.min()),
    }


def _improve_stop(current: float, candidate: float, side: OrderSide) -> float:
    return max(current, candidate) if side == OrderSide.BUY else min(current, candidate)

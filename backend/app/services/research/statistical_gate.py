"""Multiplicity-aware portfolio evidence gates for research experiments."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class GateDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class StatisticalGateResult:
    decision: GateDecision
    trade_count: int
    distinct_days: int
    expected_mean_bps: float
    stress_mean_bps: float
    adjusted_one_sided_alpha: float
    adjusted_lower_confidence_bound_bps: float | None
    probability_of_backtest_overfitting: float | None
    conditions: dict[str, bool]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["decision"] = self.decision.value
        return row


def block_bootstrap_lower_bound(
    trades: pd.DataFrame,
    *,
    value_column: str,
    alpha: float,
    samples: int = 10_000,
    seed: int = 42,
) -> float:
    """Resample complete UTC days and return a one-sided mean lower bound."""
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and one half")
    if samples < 1_000:
        raise ValueError("at least 1,000 bootstrap samples are required")
    required = {"entry_time_ms", value_column}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trade frame is missing columns: {sorted(missing)}")
    frame = trades.loc[:, ["entry_time_ms", value_column]].copy()
    frame["utc_date"] = pd.to_datetime(frame["entry_time_ms"], unit="ms", utc=True).dt.date.astype(
        str
    )
    blocks = [
        group[value_column].to_numpy(dtype=np.float64)
        for _, group in frame.groupby("utc_date", sort=True)
    ]
    if not blocks:
        raise ValueError("bootstrap needs at least one complete day")
    random = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        chosen = random.integers(0, len(blocks), size=len(blocks))
        values = np.concatenate([blocks[index] for index in chosen])
        means[sample] = values.mean()
    return float(np.quantile(means, alpha))


def probability_of_backtest_overfitting(daily_strategy_returns: pd.DataFrame) -> float | None:
    """CSCV PBO: probability the in-sample winner ranks below median OOS."""
    matrix = daily_strategy_returns.to_numpy(dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 6 or matrix.shape[1] < 2:
        return None
    if not np.isfinite(matrix).all():
        raise ValueError("daily strategy returns must be finite")
    row_count = matrix.shape[0]
    half = row_count // 2
    losses = 0
    combinations = 0
    for train_indices in itertools.combinations(range(row_count), half):
        if 0 not in train_indices:
            continue
        train_mask = np.zeros(row_count, dtype=bool)
        train_mask[list(train_indices)] = True
        test_mask = ~train_mask
        best_strategy = int(np.argmax(matrix[train_mask].mean(axis=0)))
        test_means = matrix[test_mask].mean(axis=0)
        rank = int(np.argsort(np.argsort(test_means))[best_strategy]) + 1
        normalized_rank = rank / (matrix.shape[1] + 1)
        if normalized_rank <= 0.5:
            losses += 1
        combinations += 1
    return losses / combinations if combinations else None


def evaluate_statistical_gate(
    trades: pd.DataFrame,
    *,
    attempted_hypotheses: int,
    temporal_fold_count: int,
    top_p_monotonic: bool,
    prospective_positive: bool,
    pbo: float | None,
    bootstrap_samples: int = 10_000,
) -> StatisticalGateResult:
    if attempted_hypotheses <= 0:
        raise ValueError("attempted_hypotheses must be positive")
    trade_count = len(trades)
    distinct_days = (
        int(pd.to_datetime(trades["entry_time_ms"], unit="ms", utc=True).dt.date.nunique())
        if trade_count
        else 0
    )
    expected_mean = float(trades["expected_net_bps"].mean()) if trade_count else 0.0
    stress_mean = float(trades["stress_net_bps"].mean()) if trade_count else 0.0
    adjusted_alpha = 0.05 / attempted_hypotheses
    lower_bound = None
    if distinct_days >= 2:
        lower_bound = block_bootstrap_lower_bound(
            trades,
            value_column="expected_net_bps",
            alpha=adjusted_alpha,
            samples=bootstrap_samples,
        )
    conditions = {
        "three_temporal_folds": temporal_fold_count >= 3,
        "minimum_200_trades": trade_count >= 200,
        "minimum_20_days": distinct_days >= 20,
        "positive_expected_mean": expected_mean > 0,
        "adjusted_lower_bound_positive": lower_bound is not None and lower_bound > 0,
        "positive_stress_mean": stress_mean > 0,
        "top_p_monotonic": top_p_monotonic,
        "pbo_at_most_20_percent": pbo is not None and pbo <= 0.20,
        "prospective_positive": prospective_positive,
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    economically_rejected = (
        not conditions["positive_expected_mean"] or not conditions["positive_stress_mean"]
    )
    if not failed:
        decision = GateDecision.APPROVED
    elif economically_rejected:
        decision = GateDecision.REJECTED
    else:
        decision = GateDecision.INCONCLUSIVE
    return StatisticalGateResult(
        decision=decision,
        trade_count=trade_count,
        distinct_days=distinct_days,
        expected_mean_bps=expected_mean,
        stress_mean_bps=stress_mean,
        adjusted_one_sided_alpha=adjusted_alpha,
        adjusted_lower_confidence_bound_bps=lower_bound,
        probability_of_backtest_overfitting=pbo,
        conditions=conditions,
        reasons=failed,
    )

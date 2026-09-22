"""Evidence gates reject negative economics and preserve insufficient-data states."""

import numpy as np
import pandas as pd

from app.services.research.statistical_gate import (
    GateDecision,
    block_bootstrap_lower_bound,
    evaluate_statistical_gate,
    probability_of_backtest_overfitting,
)


def trades(days: int, per_day: int, expected: float, stress: float) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for day in range(days):
        for trade in range(per_day):
            rows.append(
                {
                    "entry_time_ms": int(
                        (start + pd.Timedelta(days=day, minutes=trade)).timestamp() * 1000
                    ),
                    "expected_net_bps": expected,
                    "stress_net_bps": stress,
                }
            )
    return pd.DataFrame(rows)


def test_negative_economics_are_rejected_even_with_enough_trades() -> None:
    result = evaluate_statistical_gate(
        trades(20, 10, -1, -2),
        attempted_hypotheses=5,
        temporal_fold_count=3,
        top_p_monotonic=True,
        prospective_positive=True,
        pbo=0.1,
        bootstrap_samples=1_000,
    )
    assert result.decision == GateDecision.REJECTED
    assert not result.conditions["positive_expected_mean"]


def test_positive_but_short_history_is_inconclusive() -> None:
    result = evaluate_statistical_gate(
        trades(3, 100, 2, 1),
        attempted_hypotheses=2,
        temporal_fold_count=3,
        top_p_monotonic=True,
        prospective_positive=False,
        pbo=None,
        bootstrap_samples=1_000,
    )
    assert result.decision == GateDecision.INCONCLUSIVE
    assert not result.conditions["minimum_20_days"]


def test_top_p_monotonicity_is_required_for_approval() -> None:
    result = evaluate_statistical_gate(
        trades(20, 10, 2, 1),
        attempted_hypotheses=1,
        temporal_fold_count=3,
        top_p_monotonic=False,
        prospective_positive=True,
        pbo=0.1,
        bootstrap_samples=1_000,
    )

    assert result.decision == GateDecision.INCONCLUSIVE
    assert result.conditions["top_p_monotonic"] is False


def test_block_bootstrap_and_pbo_are_deterministic() -> None:
    frame = trades(6, 10, 2, 1)
    assert (
        block_bootstrap_lower_bound(
            frame, value_column="expected_net_bps", alpha=0.05, samples=1_000
        )
        == 2
    )
    matrix = pd.DataFrame(
        np.array(
            [
                [3, 1],
                [3, 1],
                [3, 1],
                [0, 2],
                [0, 2],
                [0, 2],
            ]
        )
    )
    pbo = probability_of_backtest_overfitting(matrix)
    assert pbo is not None and 0 <= pbo <= 1

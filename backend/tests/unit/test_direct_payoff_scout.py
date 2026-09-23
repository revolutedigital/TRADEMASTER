"""Direct-payoff scout ranks executable payoff, not binary touch events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.run_direct_payoff_scout import (
    run_direct_payoff_walk_forward,
    select_direct_payoff_candidate,
    summarize_direct_payoff_walk_forward,
)


def _result(
    *,
    selected_count: int,
    realized_mean_bps: float,
    correlation: float,
    monotonic: bool,
) -> dict[str, object]:
    return {
        "summary": {
            "feature_set": "flow",
            "horizon_seconds": 120,
            "mean_prediction_target_correlation": correlation,
            "top_p_payoff_monotonic": monotonic,
            "tails": {
                "top_1%": {
                    "selected_count": selected_count,
                    "realized_mean_bps": realized_mean_bps,
                }
            },
        }
    }


def test_select_direct_payoff_candidate_prefers_positive_monotonic_sample() -> None:
    results = [
        _result(selected_count=200, realized_mean_bps=1.0, correlation=0.4, monotonic=False),
        _result(selected_count=90, realized_mean_bps=0.5, correlation=0.3, monotonic=True),
    ]

    selection = select_direct_payoff_candidate(results, minimum_selected=50)

    assert selection["result_index"] == 1
    assert selection["selection_tier_reason"] == "positive_monotonic_minimum_selected"


def test_select_direct_payoff_candidate_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        select_direct_payoff_candidate([], minimum_selected=50)


def test_direct_payoff_walk_forward_lifts_synthetic_payoff() -> None:
    random = np.random.default_rng(42)
    rows = []
    for day in range(7):
        day_start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
        for sample in range(80):
            signal = random.normal()
            rows.append(
                {
                    "decision_time_ms": int(
                        (day_start + pd.Timedelta(minutes=sample * 10)).timestamp() * 1000
                    ),
                    "horizon_seconds": 120,
                    "side": "BUY",
                    "latency_terminal_stress_net_bps": 3.0 * signal
                    + random.normal(scale=0.5),
                    "trade_count_1s": 10.0 + abs(signal),
                    "quote_volume_1s": 100.0 + abs(signal),
                    "flow_imbalance_1s": signal,
                    "directed_flow_imbalance_1s": signal,
                    "mean_interarrival_ms_1s": 20.0,
                }
            )
    frame = pd.DataFrame(rows)

    result, _ = run_direct_payoff_walk_forward(
        frame,
        horizon_seconds=120,
        feature_set="flow",
        target_column="latency_terminal_stress_net_bps",
    )
    summary = summarize_direct_payoff_walk_forward(result)

    assert summary["mean_prediction_target_correlation"] > 0.5
    assert summary["tails"]["top_10%"]["realized_mean_bps"] > summary["base_mean_bps"]

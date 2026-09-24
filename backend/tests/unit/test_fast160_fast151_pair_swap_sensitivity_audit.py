from __future__ import annotations

import pandas as pd
import pytest

from scripts.research.run_fast160_fast151_pair_swap_sensitivity_audit import (
    apply_pair_adverse_swap_cost,
    estimate_pair_break_even_pips_per_rollover_day,
    evaluate_pair_sample,
    fast160_status,
)


def test_apply_pair_adverse_swap_cost_only_affects_target_pair() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["AUDUSD", "EURUSD", "AUDUSD"],
            "net_stress_pips": [10.0, 10.0, -5.0],
            "swap_rollover_days": [2, 5, 4],
        }
    )

    adjusted = apply_pair_adverse_swap_cost(
        trades,
        pair="AUDUSD",
        adverse_short_pips_per_rollover_day=0.5,
    )

    assert adjusted["swap_pips"].tolist() == [-1.0, 0.0, -2.0]
    assert adjusted["net_stress_pips"].tolist() == [9.0, 10.0, -7.0]


def test_estimate_pair_break_even_pips_per_rollover_day_with_zero_variance_sample() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["AUDUSD", "AUDUSD", "AUDUSD"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
                utc=True,
            ),
            "exit_stress_timestamp": pd.to_datetime(
                ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z"],
                utc=True,
            ),
            "net_stress_pips": [10.0, 10.0, 10.0],
            "swap_rollover_days": [2, 2, 2],
            "exit_stress_reason": ["hold", "hold", "hold"],
        }
    )

    assert estimate_pair_break_even_pips_per_rollover_day(trades, "AUDUSD") == pytest.approx(5.0)


def test_evaluate_pair_sample_reports_exposure_and_stress_points() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["AUDUSD", "EURUSD", "AUDUSD"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
                utc=True,
            ),
            "exit_stress_timestamp": pd.to_datetime(
                ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z"],
                utc=True,
            ),
            "net_stress_pips": [10.0, 10.0, 10.0],
            "swap_rollover_days": [2, 5, 4],
            "exit_stress_reason": ["hold", "hold", "hold"],
        }
    )

    summary = evaluate_pair_sample(trades, "AUDUSD")

    assert summary["pair_trade_count"] == 2
    assert summary["pair_total_rollover_days"] == 6
    assert summary["pair_baseline_total_stress_pips"] == 20.0
    assert len(summary["stress_evaluations"]) == 5


def test_fast160_status_flags_uniform_and_single_pair_swap_limits() -> None:
    status, reasons = fast160_status(
        pair_evaluations=[
            {
                "current": {
                    "pair_only_break_even_lcb_zero_pips_per_rollover_day": 3.5,
                }
            }
        ],
        uniform_break_even={"current_lcb_zero_pips_per_rollover_day": 0.95},
    )

    assert status == "FAST160_FAST151_PAIR_SWAP_LIMITS_IDENTIFIED_BROKER_TABLE_AND_FORWARD_REQUIRED"
    assert "uniform_current_swap_break_even_below_1_pip_per_rollover_day" in reasons
    assert "single_pair_swap_break_even_below_5_pips_per_rollover_day" in reasons
    assert "not_approved_strategy" in reasons

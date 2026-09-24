from __future__ import annotations

import pandas as pd

from scripts.research.run_fast171_fast151_pair_weight_audit import (
    apply_pair_weights,
    estimate_weighted_break_even,
    fast171_status,
    select_pre_weight_policy,
)


def _sample(*, passed: bool = True, lcb: float = 10.0, swap_break_even: float = 1.0) -> dict:
    failures = [] if passed else ["lcb_non_positive"]
    return {
        "summary": {
            "trade_count": 60,
            "mean_stress_pips": 20.0,
            "lower_stress_pips": lcb,
            "max_drawdown_pips": -300.0,
            "max_concurrent_trades": 2,
            "two_consecutive_negative_years": False,
            "positive_year_share": 0.75,
        },
        "strict_gate_passed": passed,
        "strict_gate_failures": failures,
        "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": swap_break_even,
        "weighted_rollover_day_diagnostics": {"weighted_total_rollover_days": 10.0},
    }


def _evaluation(
    weight_policy_id: str,
    *,
    pre_swap_break_even: float,
    pre_lcb: float,
    current_swap_break_even: float = 0.5,
    current_passed: bool = True,
) -> dict:
    return {
        "weight_policy_id": weight_policy_id,
        "weights": {"AUDUSD": 1.0, "EURUSD": 1.0, "GBPUSD": 1.0, "NZDUSD": 1.0},
        "pre": _sample(lcb=pre_lcb, swap_break_even=pre_swap_break_even),
        "current": _sample(passed=current_passed, lcb=5.0, swap_break_even=current_swap_break_even),
    }


def test_apply_pair_weights_scales_pips_and_weighted_rollover_days() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["AUDUSD", "GBPUSD"],
            "net_base_pips": [10.0, 20.0],
            "net_stress_pips": [8.0, 16.0],
            "raw_base_pips": [11.0, 21.0],
            "raw_stress_pips": [9.0, 17.0],
            "swap_rollover_days": [2, 4],
        }
    )

    weighted = apply_pair_weights(
        trades,
        {"AUDUSD": 0.5, "GBPUSD": 0.25, "EURUSD": 1.0, "NZDUSD": 1.0},
    )

    assert weighted["net_stress_pips"].tolist() == [4.0, 4.0]
    assert weighted["net_base_pips"].tolist() == [5.0, 5.0]
    assert weighted["weighted_swap_rollover_days"].tolist() == [1.0, 1.0]
    assert weighted["unweighted_net_stress_pips"].tolist() == [8.0, 16.0]


def test_estimate_weighted_break_even_uses_weighted_rollover_days() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["AUDUSD", "GBPUSD", "NZDUSD"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
                utc=True,
            ),
            "exit_stress_timestamp": pd.to_datetime(
                ["2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z"],
                utc=True,
            ),
            "net_stress_pips": [10.0, 10.0, 10.0],
            "exit_stress_reason": ["target", "target", "target"],
            "weighted_swap_rollover_days": [1.0, 1.0, 1.0],
        }
    )

    assert estimate_weighted_break_even(trades) == 10.0


def test_select_pre_weight_policy_ignores_current_failure_when_selecting() -> None:
    baseline = _evaluation("equal_1_00", pre_swap_break_even=1.0, pre_lcb=20.0)
    pre_best = _evaluation(
        "pair_weight:audusd_1_0:eurusd_0_5:gbpusd_1_0:nzdusd_1_0",
        pre_swap_break_even=2.0,
        pre_lcb=15.0,
        current_passed=False,
    )

    selected = select_pre_weight_policy([baseline, pre_best])

    assert selected is pre_best


def test_fast171_status_reports_pre_selected_failure_on_current_audit() -> None:
    baseline = _evaluation("equal_1_00", pre_swap_break_even=1.0, pre_lcb=20.0)
    pre_selected = _evaluation(
        "pair_weight:audusd_1_0:eurusd_0_5:gbpusd_1_0:nzdusd_1_0",
        pre_swap_break_even=2.0,
        pre_lcb=15.0,
        current_passed=False,
    )

    status, reasons = fast171_status(
        pre_selected=pre_selected,
        baseline=baseline,
        strict_survivor_count=3,
    )

    assert status == "FAST171_PRE_SELECTED_PAIR_WEIGHT_FAILS_CURRENT_AUDIT"
    assert "do_not_promote_pair_weight_policy" in reasons
    assert "not_approved_strategy" in reasons


def test_fast171_status_rejects_current_pass_without_material_swap_uplift() -> None:
    baseline = _evaluation("equal_1_00", pre_swap_break_even=1.0, pre_lcb=20.0, current_swap_break_even=1.0)
    pre_selected = _evaluation(
        "pair_weight:audusd_0_25:eurusd_1_0:gbpusd_1_0:nzdusd_1_0",
        pre_swap_break_even=2.0,
        pre_lcb=15.0,
        current_swap_break_even=1.01,
    )

    status, reasons = fast171_status(
        pre_selected=pre_selected,
        baseline=baseline,
        strict_survivor_count=3,
    )

    assert status == "FAST171_PAIR_WEIGHT_PASSES_CURRENT_BUT_NO_MATERIAL_SWAP_UPLIFT"
    assert "do_not_replace_equal_weight_fast151_policy" in reasons
    assert "not_approved_strategy" in reasons


def test_fast171_status_requires_preregistration_when_swap_uplift_is_material() -> None:
    baseline = _evaluation("equal_1_00", pre_swap_break_even=1.0, pre_lcb=20.0, current_swap_break_even=1.0)
    pre_selected = _evaluation(
        "pair_weight:audusd_0_25:eurusd_1_0:gbpusd_1_0:nzdusd_1_0",
        pre_swap_break_even=2.0,
        pre_lcb=15.0,
        current_swap_break_even=1.2,
    )

    status, reasons = fast171_status(
        pre_selected=pre_selected,
        baseline=baseline,
        strict_survivor_count=3,
    )

    assert status == "FAST171_PAIR_WEIGHT_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "pre_selected_weight_policy_passes_current_audit" in reasons
    assert "not_approved_strategy" in reasons

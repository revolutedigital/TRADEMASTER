from __future__ import annotations

from scripts.research.run_fast162_fast151_management_variant_audit import (
    fast162_status,
    select_strict_variant,
    strict_gate_failures,
)


def _summary(
    *,
    trades: int = 60,
    mean: float = 20.0,
    lcb: float = 10.0,
    drawdown: float = -500.0,
    two_bad_years: bool = False,
) -> dict:
    return {
        "trade_count": trades,
        "mean_stress_pips": mean,
        "lower_stress_pips": lcb,
        "max_drawdown_pips": drawdown,
        "max_concurrent_trades": 2,
        "two_consecutive_negative_years": two_bad_years,
        "positive_year_share": 0.8,
    }


def _sample(summary: dict, *, swap_break_even: float = 1.0) -> dict:
    failures = strict_gate_failures(summary, min_trades=50, max_drawdown_pips=1000.0)
    return {
        "summary": summary,
        "strict_gate_failures": failures,
        "strict_gate_passed": not failures,
        "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": swap_break_even,
    }


def _variant(
    risk_policy_id: str,
    *,
    pre_lcb: float,
    current_lcb: float,
    management: str = "fixed_1r",
    current_two_bad_years: bool = False,
    current_swap_break_even: float = 1.0,
) -> dict:
    return {
        "risk_policy_id": risk_policy_id,
        "management": management,
        "pre": _sample(_summary(lcb=pre_lcb)),
        "current": _sample(
            _summary(lcb=current_lcb, two_bad_years=current_two_bad_years),
            swap_break_even=current_swap_break_even,
        ),
    }


def test_strict_gate_failures_flags_bad_year_cluster() -> None:
    failures = strict_gate_failures(
        _summary(two_bad_years=True),
        min_trades=50,
        max_drawdown_pips=1000.0,
    )

    assert "two_consecutive_negative_years" in failures


def test_select_strict_variant_ignores_current_failure_and_prefers_min_lcb() -> None:
    selected = select_strict_variant(
        [
            _variant("stop2_fixed_2r", pre_lcb=30.0, current_lcb=20.0, current_two_bad_years=True),
            _variant("stop2_fixed_1r", pre_lcb=25.0, current_lcb=10.0),
            _variant("stop1.5_trail_1r_1r", pre_lcb=20.0, current_lcb=4.0, management="trail_1r_1r"),
        ]
    )

    assert selected is not None
    assert selected["risk_policy_id"] == "stop2_fixed_1r"


def test_fast162_status_reports_fixed_1r_and_trailing_comparison() -> None:
    benchmark = _variant("stop2_fixed_1r", pre_lcb=30.0, current_lcb=10.0)
    trailing = _variant("stop1.5_trail_1r_1r", pre_lcb=20.0, current_lcb=4.0, management="trail_1r_1r")

    status, reasons = fast162_status(
        strict_selected=benchmark,
        benchmark=benchmark,
        best_trailing=trailing,
    )

    assert status == "FAST162_FIXED_1R_REMAINS_PREFERRED_OVER_TRAILING"
    assert "benchmark_fixed_1r_remains_selected" in reasons
    assert "best_trailing_weaker_than_benchmark" in reasons
    assert "not_approved_strategy" in reasons

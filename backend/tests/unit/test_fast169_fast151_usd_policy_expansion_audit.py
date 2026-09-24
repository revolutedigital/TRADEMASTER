from __future__ import annotations

from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import ENTRY_POLICY_ID
from scripts.research.run_fast169_fast151_usd_policy_expansion_audit import (
    BENCHMARK_BASKET_ID,
    basket_grid,
    fast169_status,
    select_strict_survivor,
)


def _sample(*, passed: bool = True, lcb: float = 10.0, swap_break_even: float = 1.0) -> dict:
    failures = [] if passed else ["lcb_non_positive"]
    return {
        "summary": {
            "trade_count": 60,
            "mean_stress_pips": 20.0,
            "lower_stress_pips": lcb,
            "max_drawdown_pips": -300.0,
            "two_consecutive_negative_years": False,
            "positive_year_share": 0.75,
        },
        "strict_gate_passed": passed,
        "strict_gate_failures": failures,
        "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": swap_break_even,
    }


def _evaluation(
    basket_id: str,
    entry_policy_id: str,
    *,
    pre_lcb: float = 12.0,
    current_lcb: float = 10.0,
    current_swap_break_even: float = 1.0,
    pre_passed: bool = True,
    current_passed: bool = True,
) -> dict:
    return {
        "basket_id": basket_id,
        "entry_policy_id": entry_policy_id,
        "pre": _sample(passed=pre_passed, lcb=pre_lcb),
        "current": _sample(
            passed=current_passed,
            lcb=current_lcb,
            swap_break_even=current_swap_break_even,
        ),
    }


def test_basket_grid_returns_all_non_empty_subsets_in_order() -> None:
    assert basket_grid(("A", "B", "C")) == (
        ("A",),
        ("B",),
        ("C",),
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("A", "B", "C"),
    )


def test_fast169_status_reports_unique_fast151_survivor() -> None:
    benchmark = _evaluation(BENCHMARK_BASKET_ID, ENTRY_POLICY_ID)

    status, reasons = fast169_status(
        strict_survivors=[benchmark],
        strict_selected=benchmark,
        benchmark=benchmark,
    )

    assert status == "FAST169_FAST151_UNIQUE_STRICT_SURVIVOR_IN_USD_POLICY_EXPANSION"
    assert "fast151_remains_only_strict_survivor" in reasons
    assert "not_approved_strategy" in reasons


def test_fast169_status_does_not_approve_alternative_historical_selection() -> None:
    benchmark = _evaluation(BENCHMARK_BASKET_ID, ENTRY_POLICY_ID, pre_lcb=12.0, current_lcb=8.0)
    alternative = _evaluation(
        "AUDUSD+GBPUSD+NZDUSD",
        "d1_donchian_fade_ch55_hold10",
        pre_lcb=18.0,
        current_lcb=15.0,
        current_swap_break_even=2.0,
    )
    selected = select_strict_survivor([benchmark, alternative])

    status, reasons = fast169_status(
        strict_survivors=[benchmark, alternative],
        strict_selected=selected,
        benchmark=benchmark,
    )

    assert selected is alternative
    assert status == "FAST169_ALTERNATIVE_USD_POLICY_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "alternative_selected_historically_not_approved" in reasons
    assert "not_approved_strategy" in reasons

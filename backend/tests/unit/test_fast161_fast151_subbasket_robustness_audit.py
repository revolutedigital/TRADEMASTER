from __future__ import annotations

from scripts.research.run_fast161_fast151_subbasket_robustness_audit import (
    exploratory_viable,
    pair_subsets,
    select_exploratory_basket,
    select_strict_basket,
    strict_gate_failures,
)


def _summary(
    *,
    trades: int,
    mean: float,
    lcb: float,
    drawdown: float = -100.0,
    positive_year_share: float = 0.8,
    two_bad_years: bool = False,
) -> dict:
    return {
        "trade_count": trades,
        "mean_stress_pips": mean,
        "lower_stress_pips": lcb,
        "max_drawdown_pips": drawdown,
        "max_concurrent_trades": 1,
        "two_consecutive_negative_years": two_bad_years,
        "positive_year_share": positive_year_share,
    }


def _evaluation(
    basket_id: str,
    *,
    pre_lcb: float,
    current_lcb: float,
    pre_trades: int = 60,
    current_trades: int = 60,
    swap_break_even: float = 1.0,
    pair_count: int = 4,
) -> dict:
    return {
        "basket_id": basket_id,
        "pairs": basket_id.split("+"),
        "pair_count": pair_count,
        "pre": {
            "summary": _summary(trades=pre_trades, mean=20.0, lcb=pre_lcb),
            "strict_gate_failures": [] if pre_trades >= 50 and pre_lcb > 0 else ["trade_count_below_minimum"],
            "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": 1.0,
        },
        "current": {
            "summary": _summary(trades=current_trades, mean=20.0, lcb=current_lcb),
            "strict_gate_failures": [] if current_trades >= 50 and current_lcb > 0 else ["trade_count_below_minimum"],
            "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": swap_break_even,
        },
        "strict_both_gate_passed": pre_trades >= 50 and current_trades >= 50 and pre_lcb > 0 and current_lcb > 0,
    }


def test_pair_subsets_returns_all_non_empty_combinations() -> None:
    assert pair_subsets(("A", "B", "C")) == [
        ("A",),
        ("B",),
        ("C",),
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("A", "B", "C"),
    ]


def test_strict_gate_failures_reports_sample_and_lcb_failures() -> None:
    failures = strict_gate_failures(
        _summary(trades=10, mean=5.0, lcb=-1.0),
        min_trades=50,
        max_drawdown_pips=1000.0,
    )

    assert "trade_count_below_minimum" in failures
    assert "lcb_non_positive" in failures


def test_select_strict_basket_requires_both_gates_and_prefers_stronger_min_lcb() -> None:
    selected = select_strict_basket(
        [
            _evaluation("A+B", pre_lcb=5.0, current_lcb=5.0, pair_count=2),
            _evaluation("A+B+C", pre_lcb=10.0, current_lcb=8.0, pair_count=3),
            _evaluation("A", pre_lcb=50.0, current_lcb=50.0, pre_trades=20, pair_count=1),
        ]
    )

    assert selected is not None
    assert selected["basket_id"] == "A+B+C"


def test_select_exploratory_basket_allows_low_sample_but_requires_positive_lcb() -> None:
    selected = select_exploratory_basket(
        [
            _evaluation("A", pre_lcb=-1.0, current_lcb=50.0, pre_trades=10, pair_count=1),
            _evaluation("B+C", pre_lcb=20.0, current_lcb=6.0, pre_trades=20, pair_count=2),
            _evaluation("A+B+C", pre_lcb=10.0, current_lcb=10.0, pre_trades=20, pair_count=3),
        ]
    )

    assert selected is not None
    assert selected["basket_id"] == "A+B+C"
    assert exploratory_viable(selected["pre"]["summary"])

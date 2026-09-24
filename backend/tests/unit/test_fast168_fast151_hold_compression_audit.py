from __future__ import annotations

from scripts.research.run_fast168_fast151_hold_compression_audit import (
    ENTRY_POLICY_ID,
    entry_policy_passes_both,
    fast168_status,
    select_best_current_swap_candidate,
    select_strict_entry_policy,
)
from scripts.research.run_fast162_fast151_management_variant_audit import strict_gate_failures


def _summary(
    *,
    trades: int = 60,
    mean: float = 25.0,
    lcb: float = 12.0,
    drawdown: float = -400.0,
    two_bad_years: bool = False,
    positive_year_share: float = 0.75,
) -> dict:
    return {
        "trade_count": trades,
        "mean_stress_pips": mean,
        "lower_stress_pips": lcb,
        "max_drawdown_pips": drawdown,
        "max_concurrent_trades": 2,
        "two_consecutive_negative_years": two_bad_years,
        "positive_year_share": positive_year_share,
    }


def _sample(summary: dict, *, swap_break_even: float = 1.0, min_trades: int = 50) -> dict:
    failures = strict_gate_failures(summary, min_trades=min_trades, max_drawdown_pips=1000.0)
    return {
        "summary": summary,
        "strict_gate_failures": failures,
        "strict_gate_passed": not failures,
        "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": swap_break_even,
    }


def _entry_policy(
    entry_policy_id: str,
    *,
    hold_days: int,
    pre_lcb: float,
    current_lcb: float,
    current_swap_break_even: float,
    pre_two_bad_years: bool = False,
    current_two_bad_years: bool = False,
) -> dict:
    return {
        "entry_policy_id": entry_policy_id,
        "hold_days": hold_days,
        "pre": _sample(_summary(lcb=pre_lcb, two_bad_years=pre_two_bad_years)),
        "current": _sample(
            _summary(lcb=current_lcb, two_bad_years=current_two_bad_years),
            swap_break_even=current_swap_break_even,
        ),
    }


def test_select_strict_entry_policy_ignores_current_only_swap_winner() -> None:
    hold5 = _entry_policy(
        "d1_donchian_fade_ch100_hold5",
        hold_days=5,
        pre_lcb=-8.0,
        current_lcb=19.0,
        current_swap_break_even=5.6,
    )
    benchmark = _entry_policy(
        ENTRY_POLICY_ID,
        hold_days=20,
        pre_lcb=32.0,
        current_lcb=9.0,
        current_swap_break_even=0.95,
    )

    selected = select_strict_entry_policy([hold5, benchmark])
    best_current_swap = select_best_current_swap_candidate([hold5, benchmark])

    assert selected is not None
    assert selected["entry_policy_id"] == ENTRY_POLICY_ID
    assert best_current_swap is not None
    assert best_current_swap["entry_policy_id"] == "d1_donchian_fade_ch100_hold5"
    assert entry_policy_passes_both(hold5) is False


def test_fast168_status_keeps_hold20_when_compressed_policy_fails_pre_gate() -> None:
    hold5 = _entry_policy(
        "d1_donchian_fade_ch100_hold5",
        hold_days=5,
        pre_lcb=-8.0,
        current_lcb=19.0,
        current_swap_break_even=5.6,
    )
    benchmark = _entry_policy(
        ENTRY_POLICY_ID,
        hold_days=20,
        pre_lcb=32.0,
        current_lcb=9.0,
        current_swap_break_even=0.95,
    )

    status, reasons = fast168_status(
        strict_selected=benchmark,
        benchmark=benchmark,
        best_current_swap=hold5,
    )

    assert status == "FAST168_HOLD20_REMAINS_SELECTED_COMPRESSED_HOLDS_FAIL_PRE_GATE"
    assert "benchmark_hold20_remains_selected" in reasons
    assert "best_current_swap_policy_not_pre_robust" in reasons
    assert "not_approved_strategy" in reasons


def test_fast168_status_allows_future_compressed_candidate_but_does_not_approve() -> None:
    benchmark = _entry_policy(
        ENTRY_POLICY_ID,
        hold_days=20,
        pre_lcb=20.0,
        current_lcb=10.0,
        current_swap_break_even=1.0,
    )
    hold10 = _entry_policy(
        "d1_donchian_fade_ch100_hold10",
        hold_days=10,
        pre_lcb=24.0,
        current_lcb=14.0,
        current_swap_break_even=2.0,
    )

    status, reasons = fast168_status(
        strict_selected=hold10,
        benchmark=benchmark,
        best_current_swap=hold10,
    )

    assert status == "FAST168_COMPRESSED_HOLD_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "compressed_hold_selected_historically_not_approved" in reasons
    assert "not_approved_strategy" in reasons

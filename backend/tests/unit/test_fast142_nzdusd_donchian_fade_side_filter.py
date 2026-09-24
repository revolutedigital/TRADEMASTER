from __future__ import annotations

import pandas as pd

from scripts.research.run_fast142_nzdusd_donchian_fade_side_filter import (
    evaluate_filters,
    fast142_status,
    passes_audit_gate,
    passes_selection_gate,
    select_best_passed,
    summarize_sample,
)


def _trade(*, slice_name: str, side: str, year: int, day: int, stress: float) -> dict:
    timestamp = pd.Timestamp(f"{year}-01-{day:02d}T00:00:00Z")
    return {
        "slice": slice_name,
        "entry_timestamp": timestamp,
        "exit_stress_timestamp": timestamp + pd.Timedelta(days=1),
        "side": side,
        "net_stress_pips": stress,
    }


def test_summary_gates_require_positive_lcb_and_years() -> None:
    sample = pd.DataFrame(
        [
            _trade(slice_name="pre_2011_2018", side="SHORT", year=2012 + year_offset, day=day, stress=10.0)
            for year_offset in range(3)
            for day in range(1, 6)
        ]
    )

    summary = summarize_sample(sample)

    assert summary["trade_count"] == 15
    assert summary["year_count"] == 3
    assert summary["lower_stress_pips"] > 0
    assert passes_selection_gate(summary)
    assert passes_audit_gate(summary)


def test_evaluate_filters_selects_short_only_chronologically() -> None:
    rows = []
    for year_offset in range(5):
        year = 2012 + year_offset
        rows.extend(_trade(slice_name="pre_2011_2018", side="SHORT", year=year, day=day, stress=20.0) for day in range(1, 5))
        rows.extend(_trade(slice_name="pre_2011_2018", side="LONG", year=year, day=day + 10, stress=-20.0) for day in range(1, 5))
    for year_offset in range(4):
        year = 2019 + year_offset
        rows.extend(_trade(slice_name="current_2019_2026", side="SHORT", year=year, day=day, stress=15.0) for day in range(1, 5))
        rows.extend(_trade(slice_name="current_2019_2026", side="LONG", year=year, day=day + 10, stress=-10.0) for day in range(1, 5))

    evaluations = evaluate_filters(pd.DataFrame(rows))
    best = select_best_passed(evaluations)

    assert best is not None
    assert best["filter_id"] == "short_only"
    assert best["selection_passed"]
    assert best["audit_passed"]


def test_fast142_status_never_approves_strategy() -> None:
    status, reasons = fast142_status(
        {
            "filter_id": "short_only",
            "selection": {},
            "audit": {},
            "selection_passed": True,
            "audit_passed": True,
        }
    )

    assert status == "FAST142_NZDUSD_SHORT_ONLY_SIDE_FILTER_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

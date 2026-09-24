from __future__ import annotations

import pandas as pd

from scripts.research.run_fast153_fast151_robustness_audit import (
    contribution_rows,
    fast153_status,
)


def test_contribution_rows_marks_positive_group_totals() -> None:
    rows = contribution_rows(
        pd.DataFrame(
            [
                {"pair": "AUDUSD", "net_stress_pips": 10.0},
                {"pair": "AUDUSD", "net_stress_pips": 20.0},
                {"pair": "EURUSD", "net_stress_pips": -5.0},
            ]
        ),
        "pair",
    )

    by_pair = {row["pair"]: row for row in rows}
    assert by_pair["AUDUSD"]["positive_total_stress"] is True
    assert by_pair["AUDUSD"]["trade_count"] == 2
    assert by_pair["EURUSD"]["positive_total_stress"] is False


def test_fast153_status_flags_concentration_risks() -> None:
    positive_pair = [{"positive_total_stress": True}]
    clean_leave = [{"passes_positive_lcb": True, "two_consecutive_negative_years": False}]
    risky_leave = [{"passes_positive_lcb": False, "two_consecutive_negative_years": True}]

    status, reasons = fast153_status(
        pre_pair=positive_pair,
        current_pair=positive_pair,
        pre_leave_pair=clean_leave,
        current_leave_pair=clean_leave,
        pre_leave_year=clean_leave,
        current_leave_year=risky_leave,
    )

    assert status == "FAST153_FAST151_ROBUST_BUT_CONCENTRATION_RISKS_REQUIRE_PROSPECTIVE_OBSERVATION"
    assert "current_leave_one_year_lcb_not_all_positive" in reasons
    assert "not_approved_strategy" in reasons


def test_fast153_status_reports_clean_when_all_robustness_flags_pass() -> None:
    positive_pair = [{"positive_total_stress": True}]
    clean_leave = [{"passes_positive_lcb": True, "two_consecutive_negative_years": False}]

    status, reasons = fast153_status(
        pre_pair=positive_pair,
        current_pair=positive_pair,
        pre_leave_pair=clean_leave,
        current_leave_pair=clean_leave,
        pre_leave_year=clean_leave,
        current_leave_year=clean_leave,
    )

    assert status == "FAST153_FAST151_ROBUSTNESS_AUDIT_CLEAN_REQUIRES_PROSPECTIVE_OBSERVATION"
    assert "fresh_preregistered_prospective_observation_required_before_any_execution" in reasons

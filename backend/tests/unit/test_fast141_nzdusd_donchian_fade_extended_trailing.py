from __future__ import annotations

import pandas as pd

from scripts.research.run_fast141_nzdusd_donchian_fade_extended_trailing import (
    BENCHMARK_RISK_POLICY_ID,
    best_trailing_row,
    extended_risk_policy_grid,
    fast141_status,
)


def test_extended_grid_includes_half_r_trailing_and_benchmark() -> None:
    policies = extended_risk_policy_grid()
    identifiers = {policy.identifier for policy in policies}

    assert len(policies) == 45
    assert BENCHMARK_RISK_POLICY_ID in identifiers
    assert "stop2_trail_0_5r_0_5r" in identifiers
    assert "stop2_trail_0_5r_0_75r" in identifiers
    assert "stop2_trail_1r_1r" in identifiers


def test_best_trailing_row_prefers_passed_higher_stress() -> None:
    audit = pd.DataFrame(
        [
            {
                "risk_policy_id": "stop2_fixed_1r",
                "management": "fixed_1r",
                "pilot_gate": True,
                "audit_mean_stress_pips": 10.0,
                "audit_lower_stress_pips": 2.0,
                "audit_positive_stress_year_share": 1.0,
                "audit_trades": 20,
            },
            {
                "risk_policy_id": "stop2_trail_0_5r_0_5r",
                "management": "trail_0_5r_0_5r",
                "pilot_gate": False,
                "audit_mean_stress_pips": 40.0,
                "audit_lower_stress_pips": -1.0,
                "audit_positive_stress_year_share": 1.0,
                "audit_trades": 20,
            },
            {
                "risk_policy_id": "stop2_trail_1r_0_75r",
                "management": "trail_1r_0_75r",
                "pilot_gate": True,
                "audit_mean_stress_pips": 12.0,
                "audit_lower_stress_pips": 3.0,
                "audit_positive_stress_year_share": 1.0,
                "audit_trades": 20,
            },
        ]
    )

    best = best_trailing_row(audit)

    assert best is not None
    assert best["risk_policy_id"] == "stop2_trail_1r_0_75r"


def test_fast141_status_never_approves_strategy() -> None:
    status, reasons = fast141_status(
        passed_count=2,
        finalist_count=4,
        benchmark_passed=True,
        best_trailing_beats_benchmark=True,
    )

    assert status == "FAST141_EXTENDED_TRAILING_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

    status, reasons = fast141_status(
        passed_count=1,
        finalist_count=4,
        benchmark_passed=True,
        best_trailing_beats_benchmark=False,
    )

    assert status == "FAST141_EXTENDED_TRAILING_DID_NOT_BEAT_FIXED_1R_BENCHMARK"
    assert "fixed_1r_benchmark_remains_preferred" in reasons

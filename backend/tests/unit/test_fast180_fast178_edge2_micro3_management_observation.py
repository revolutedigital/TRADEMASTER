from __future__ import annotations

from scripts.research.run_fast180_fast178_edge2_micro3_management_observation import (
    PREREGISTRATION_TIMESTAMP,
    evaluate_gates,
    fast180_status,
    filter_post_preregistration_rows,
    summarize_observations,
)


def test_filter_post_preregistration_rows_is_strict() -> None:
    at_anchor = PREREGISTRATION_TIMESTAMP.isoformat()
    after_anchor = "2026-09-24T14:20:00+00:00"
    rows = [
        {"decision_time": at_anchor},
        {"decision_time": after_anchor},
    ]

    filtered = filter_post_preregistration_rows(rows)

    assert [row["decision_time"] for row in filtered] == [after_anchor]


def test_summarize_observations_uses_stress_net_and_exit_counts() -> None:
    observations = [
        {"date": "2026-09-24", "stress_net_bps": 2.0, "exit_reason": "BREAKEVEN_STOP"},
        {"date": "2026-09-24", "stress_net_bps": -1.0, "exit_reason": "INITIAL_STOP"},
    ]

    summary = summarize_observations(observations, alpha=0.05)

    assert summary["decision_count"] == 2
    assert summary["stress_net_total_bps"] == 1.0
    assert summary["stress_net_mean_bps"] == 0.5
    assert summary["exit_reason_counts"] == {"BREAKEVEN_STOP": 1, "INITIAL_STOP": 1}


def test_fast180_status_waits_for_post_anchor_rows() -> None:
    status, reasons = fast180_status(
        prospective_row_count=0,
        observation_count=0,
        gates={"early_fail": False, "shadow_candidate": False, "shadow_reasons": []},
    )

    assert status == "FAST180_WAITING_FOR_POST_PREREGISTRATION_DATA"
    assert "no_post_preregistration_decision_rows" in reasons


def test_fast180_status_reports_no_eligible_decisions() -> None:
    status, reasons = fast180_status(
        prospective_row_count=2,
        observation_count=0,
        gates={"early_fail": False, "shadow_candidate": False, "shadow_reasons": []},
    )

    assert status == "FAST180_DATA_FRESH_NO_ELIGIBLE_DECISIONS_YET"
    assert "fast178_edge_or_regime_gate_not_met" in reasons


def test_evaluate_gates_requires_sample_even_when_positive() -> None:
    gates = evaluate_gates(
        {
            "decision_count": 1,
            "day_count": 1,
            "stress_net_total_bps": 4.0,
            "stress_net_mean_bps": 4.0,
            "stress_net_lcb_bps": 4.0,
            "positive_day_share": 0.0,
            "sufficient_day_count": 0,
            "worst_sufficient_day_net_bps": 0.0,
        }
    )

    assert not gates["early_fail"]
    assert not gates["shadow_candidate"]
    assert "insufficient_decisions" in gates["shadow_reasons"]

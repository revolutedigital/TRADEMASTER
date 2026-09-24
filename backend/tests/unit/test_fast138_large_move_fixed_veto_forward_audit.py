from __future__ import annotations

from scripts.research.run_fast138_large_move_fixed_veto_forward_audit import (
    choose_best_train_policy,
    fast138_status,
)


def _policy_result(name: str, *, stress_positive: int, min_exec_windows: int, stress_net: float) -> dict:
    return {
        "policy": {"name": name},
        "aggregate": {
            "positive_window_count_by_cost_bps": {"1": stress_positive, "2": stress_positive},
            "minimum_execution_window_count": min_exec_windows,
            "net_total_by_cost_bps": {"1": stress_net, "2": stress_net},
        },
    }


def _audit_aggregate(
    *,
    stress_positive: int,
    all_stress_positive: bool,
    all_min_exec: bool,
    stress_net: float,
) -> dict:
    return {
        "positive_window_count_by_cost_bps": {"2": stress_positive},
        "all_windows_positive_at_stress_cost": all_stress_positive,
        "all_windows_meet_minimum_executions": all_min_exec,
        "net_total_by_cost_bps": {"2": stress_net},
    }


def test_choose_best_train_policy_prioritizes_stress_windows_then_net() -> None:
    best = choose_best_train_policy(
        [
            _policy_result("lower_net", stress_positive=8, min_exec_windows=8, stress_net=10.0),
            _policy_result("higher_net", stress_positive=8, min_exec_windows=8, stress_net=20.0),
            _policy_result("fewer_positive", stress_positive=7, min_exec_windows=10, stress_net=100.0),
        ],
        stress_cost_bps=2.0,
        profit_cost_bps=1.0,
    )

    assert best["policy"]["name"] == "higher_net"


def test_fast138_status_requires_train_selection_before_audit_promotion() -> None:
    status, reasons = fast138_status(
        best_train_policy=_policy_result("candidate", stress_positive=7, min_exec_windows=10, stress_net=100.0),
        audit_aggregate=_audit_aggregate(stress_positive=3, all_stress_positive=True, all_min_exec=True, stress_net=10.0),
        audit_holdout_count=3,
        minimum_train_positive_windows=8,
        stress_cost_bps=2.0,
    )

    assert status == "FAST138_LARGE_MOVE_FIXED_VETO_FAILED_TRAIN_SELECTION"
    assert "not_approved_strategy" in reasons


def test_fast138_status_never_approves_forward_candidate() -> None:
    status, reasons = fast138_status(
        best_train_policy=_policy_result("candidate", stress_positive=8, min_exec_windows=10, stress_net=100.0),
        audit_aggregate=_audit_aggregate(stress_positive=3, all_stress_positive=True, all_min_exec=True, stress_net=10.0),
        audit_holdout_count=3,
        minimum_train_positive_windows=8,
        stress_cost_bps=2.0,
    )

    assert status == "FAST138_LARGE_MOVE_FIXED_VETO_FORWARD_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons


def test_fast138_status_fails_late_audit_when_any_late_window_breaks() -> None:
    status, reasons = fast138_status(
        best_train_policy=_policy_result("candidate", stress_positive=8, min_exec_windows=10, stress_net=100.0),
        audit_aggregate=_audit_aggregate(stress_positive=2, all_stress_positive=False, all_min_exec=True, stress_net=10.0),
        audit_holdout_count=3,
        minimum_train_positive_windows=8,
        stress_cost_bps=2.0,
    )

    assert status == "FAST138_LARGE_MOVE_FIXED_VETO_FAILED_LATE_AUDIT"
    assert "do_not_promote_to_strategy" in reasons

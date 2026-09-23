import pandas as pd

from scripts.research.run_fast105_fixed_veto_prospective_observation import (
    apply_fixed_veto,
    evaluate_gates,
    fast105_status,
    post_preregistration_frames_by_label,
    preregistration_timestamp_ms,
)


def test_post_preregistration_filter_excludes_old_rows() -> None:
    cutoff = preregistration_timestamp_ms()
    frames, audit = post_preregistration_frames_by_label(
        {
            "fresh": pd.DataFrame(
                {
                    "decision_time_ms": [cutoff - 1, cutoff, cutoff + 1],
                    "gross_executable_bps": [1.0, 2.0, 3.0],
                }
            )
        }
    )

    assert frames["fresh"]["decision_time_ms"].tolist() == [cutoff + 1]
    assert audit[0]["pre_or_at_preregistration_row_count"] == 2
    assert audit[0]["post_preregistration_row_count"] == 1


def test_apply_fixed_veto_requires_non_negative_spot_return() -> None:
    rows = pd.DataFrame(
        {
            "spot_return_1s_bps": [-0.01, 0.0, 0.25, None],
            "model_score": [4.0, 3.0, 2.0, 1.0],
        }
    )

    selected = apply_fixed_veto(rows)

    assert selected["spot_return_1s_bps"].tolist() == [0.0, 0.25]


def test_fast105_status_requires_minimum_prospective_evidence() -> None:
    gates = evaluate_gates(
        aggregate={
            "executable_count": 4,
            "eligible_window_count": 1,
            "net_total_by_cost_bps": {"1": 12.0, "2": 8.0},
            "net_mean_by_cost_bps": {"1": 3.0, "2": 2.0},
            "positive_window_count_by_cost_bps": {"2": 1},
        },
        prospective_results=[
            prospective_result("fresh1", executable_count=4, gross_total_bps=16.0),
        ],
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
    )

    status, reasons = fast105_status(gates)

    assert status == "FAST105_INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    assert "gate_not_met:minimum_trades" in reasons
    assert "gate_not_met:minimum_windows" in reasons


def test_fast105_status_flags_shadow_review_candidate_without_approving_testnet() -> None:
    gates = evaluate_gates(
        aggregate={
            "executable_count": 30,
            "eligible_window_count": 3,
            "net_total_by_cost_bps": {"1": 90.0, "2": 60.0},
            "net_mean_by_cost_bps": {"1": 3.0, "2": 2.0},
            "positive_window_count_by_cost_bps": {"2": 3},
        },
        prospective_results=[
            prospective_result("fresh1", executable_count=10, gross_total_bps=40.0),
            prospective_result("fresh2", executable_count=10, gross_total_bps=40.0),
            prospective_result("fresh3", executable_count=10, gross_total_bps=40.0),
        ],
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
    )

    status, reasons = fast105_status(gates)

    assert status == "FAST105_SHADOW_OFFLINE_REVIEW_CANDIDATE_NOT_TESTNET"
    assert "do_not_promote_to_testnet" in reasons


def prospective_result(
    label: str,
    *,
    executable_count: int,
    gross_total_bps: float,
) -> dict[str, object]:
    return {
        "label": label,
        "one_position_summary": {
            "executable_count": executable_count,
            "gross_executable_total_bps": gross_total_bps,
            "cost_sensitivity": [
                {
                    "explicit_cost_bps": 1.0,
                    "net_executable_total_bps": gross_total_bps - executable_count,
                },
                {
                    "explicit_cost_bps": 2.0,
                    "net_executable_total_bps": gross_total_bps - 2 * executable_count,
                },
            ],
        },
    }

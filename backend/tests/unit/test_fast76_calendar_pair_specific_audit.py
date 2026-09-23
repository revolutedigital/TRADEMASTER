import pandas as pd

from scripts.research.run_fast76_calendar_pair_specific_audit import (
    evaluate_scope,
    fast76_status,
    lower_confidence_bound,
)


def test_lower_confidence_bound_is_below_sample_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    bound = lower_confidence_bound(values, alpha=0.05)

    assert bound < sum(values) / len(values)


def test_evaluate_scope_marks_mean_positive_but_not_significant_as_weak_survivor() -> None:
    trades = pd.DataFrame(
        {
            "event_timestamp": pd.to_datetime(
                [
                    "2023-01-01T00:00:00Z",
                    "2023-01-02T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "pair": ["EURUSD"] * 4,
            "policy_id": ["p1"] * 4,
            "event_name": ["NFP"] * 4,
            "net_base_pips": [1.0, 1.0, 1.0, 3.0],
            "net_stress_pips": [1.0, 1.0, -5.0, 9.0],
        }
    )

    result = evaluate_scope(
        trades,
        group_columns=("pair", "policy_id"),
        minimum_selection_trades=2,
        minimum_audit_trades=2,
        alpha=0.05,
    )

    assert result["summary"]["screen_candidate_count"] == 1
    assert result["summary"]["weak_survivor_count"] == 1
    assert result["summary"]["passed_candidate_count"] == 0


def test_fast76_status_rejects_weak_survivors_without_corrected_bound() -> None:
    status, reasons = fast76_status(
        pair_policy={"summary": {"passed_candidate_count": 0, "weak_survivor_count": 1}},
        pair_event_policy={"summary": {"passed_candidate_count": 0, "weak_survivor_count": 0}},
    )

    assert status == "FAST76_CALENDAR_PAIR_SPECIFIC_WEAK_SURVIVORS_FAILED_SIGNIFICANCE"
    assert "corrected_lower_bounds_non_positive" in reasons

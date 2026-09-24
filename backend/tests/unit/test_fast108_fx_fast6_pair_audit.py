from datetime import UTC, datetime

import pandas as pd

from scripts.research.run_fast108_fx_fast6_pair_audit import (
    evaluate_policy_pairs,
    fast108_status,
)


def test_evaluate_policy_pairs_summarizes_month_stability() -> None:
    trades = pd.DataFrame(
        {
            "decision_time": [
                datetime(2021, 4, 1, tzinfo=UTC),
                datetime(2021, 4, 2, tzinfo=UTC),
                datetime(2021, 5, 1, tzinfo=UTC),
            ],
            "pair": ["EURUSD", "EURUSD", "EURUSD"],
            "base_r": [1.0, -0.5, 2.0],
            "stress_r": [0.5, -0.25, 1.5],
            "month": ["2021-04", "2021-04", "2021-05"],
        }
    )

    rows = evaluate_policy_pairs(policy_name="policy-a", trades=trades)

    assert rows == [
        {
            "policy": "policy-a",
            "pair": "EURUSD",
            "trade_count": 3,
            "month_count": 2,
            "base_mean_r": (1.0 - 0.5 + 2.0) / 3,
            "stress_mean_r": (0.5 - 0.25 + 1.5) / 3,
            "base_total_r": 2.5,
            "stress_total_r": 1.75,
            "positive_base_month_count": 2,
            "positive_stress_month_count": 2,
            "min_month_stress_mean_r": 0.125,
            "min_month_stress_total_r": 0.25,
        }
    ]


def test_fast108_status_rejects_when_no_pair_policy_positive() -> None:
    status, reasons = fast108_status(candidate_count=0, weak_count=0)

    assert status == "FAST108_PAIR_SPECIFIC_FAILED_SCREEN"
    assert "no_pair_policy_positive_base_and_stress" in reasons

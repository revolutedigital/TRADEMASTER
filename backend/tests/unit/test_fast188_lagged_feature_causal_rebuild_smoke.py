from __future__ import annotations

import pandas as pd

from scripts.research.run_fast188_lagged_feature_causal_rebuild_smoke import (
    fast188_status,
    lag_feature_columns,
)


def test_lag_feature_columns_shifts_within_date_only() -> None:
    rows = pd.DataFrame(
        {
            "date": ["a", "a", "b"],
            "decision_time": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "feature": [1.0, 2.0, 3.0],
        }
    )

    lagged = lag_feature_columns(rows, ["feature"])

    assert pd.isna(lagged.iloc[0]["feature"])
    assert lagged.iloc[1]["feature"] == 1.0
    assert pd.isna(lagged.iloc[2]["feature"])


def test_fast188_reports_no_simple_survivor() -> None:
    status, reasons = fast188_status(
        [
            {
                "candidate_id": "a",
                "positive_day_count": 4,
                "worst_day_stress_total_bps": -1.0,
            }
        ]
    )

    assert status == "FAST188_LAGGED_FEATURE_CAUSAL_REBUILD_NO_SIMPLE_SURVIVOR"
    assert "no_threshold_policy_candidate_positive_on_9_of_9_days" in reasons

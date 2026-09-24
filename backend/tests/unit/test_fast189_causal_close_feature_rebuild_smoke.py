from __future__ import annotations

import pandas as pd

from scripts.research.run_fast189_causal_close_feature_rebuild_smoke import (
    FEATURE_CLOSE_DELAY_SECONDS,
    build_causal_label_rows,
    fast189_status,
)


def test_fast189_status_requires_preregistration_for_perfect_survivor() -> None:
    status, reasons = fast189_status(
        [
            {
                "positive_day_count": 9,
                "worst_day_stress_total_bps": 1.0,
                "management_policy": "micro_5_2",
                "edge_threshold_bps": 3.0,
                "start_hour_utc": 5,
                "end_hour_utc": 16,
            }
        ]
    )

    assert status == "FAST189_CAUSAL_CLOSE_FEATURE_HISTORICAL_SURVIVOR_REQUIRES_PREREGISTRATION"
    assert "fresh_future_preregistration_required" in reasons


def test_feature_close_delay_constant_is_one_minute() -> None:
    assert FEATURE_CLOSE_DELAY_SECONDS == 60

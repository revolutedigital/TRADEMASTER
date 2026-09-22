"""Top-p thresholds are fitted on calibration time, never the test block."""

import numpy as np
import pandas as pd

from app.services.research.top_p_model import (
    run_calibrated_walk_forward,
    summarize_walk_forward,
)


def test_calibrated_walk_forward_is_temporal_and_reports_all_tails() -> None:
    random = np.random.default_rng(42)
    rows = []
    for day in range(7):
        day_start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
        for sample in range(80):
            signal = random.normal()
            rows.append(
                {
                    "decision_time_ms": int(
                        (day_start + pd.Timedelta(minutes=sample * 10)).timestamp() * 1000
                    ),
                    "horizon_seconds": 120,
                    "target": int(signal + random.normal(scale=0.5) > 0),
                    "flow_imbalance_1s": signal,
                    "directed_flow_imbalance_1s": signal,
                    "trade_count_1s": 10 + abs(signal),
                    "quote_volume_1s": 100 + abs(signal),
                    "mean_interarrival_ms_1s": 20,
                }
            )
    frame = pd.DataFrame(rows)

    result = run_calibrated_walk_forward(
        frame,
        horizon_seconds=120,
        feature_set="flow",
        embargo_seconds=300,
    )
    summary = summarize_walk_forward(result)

    assert len(result.folds) == 3
    assert result.folds[0].train_end_date < result.folds[0].calibration_date
    assert result.folds[0].calibration_date < result.folds[0].test_date
    assert len(result.folds[0].tails) == 4
    assert summary["fold_count"] == 3
    assert summary["mean_roc_auc"] > 0.7

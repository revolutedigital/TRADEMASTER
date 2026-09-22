"""Top-p thresholds are fitted on calibration time, never the test block."""

import numpy as np
import pandas as pd
import pytest

from app.services.research.top_p_model import (
    feature_columns,
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


def test_book_feature_sets_require_real_book_columns() -> None:
    frame = pd.DataFrame(
        {
            "trade_count_1s": [1.0],
            "flow_imbalance_1s": [0.1],
            "directed_flow_imbalance_1s": [0.1],
        }
    )

    with pytest.raises(ValueError, match="requires book"):
        feature_columns(frame, "flow_book")


def test_book_feature_sets_include_directed_microstructure_columns() -> None:
    frame = pd.DataFrame(
        {
            "trade_count_1s": [1.0],
            "flow_imbalance_1s": [0.1],
            "directed_flow_imbalance_1s": [0.1],
            "return_1s_bps": [2.0],
            "directed_return_1s_bps": [2.0],
            "book_available": [1.0],
            "spread_bps": [1.5],
            "depth_imbalance": [0.2],
            "directed_depth_imbalance": [0.2],
            "microprice_displacement_bps": [0.4],
            "directed_microprice_displacement_bps": [0.4],
            "hour_sin": [0.0],
            "hour_cos": [1.0],
            "side_sign": [1.0],
        }
    )

    columns = feature_columns(frame, "flow_price_book_session")

    assert "directed_depth_imbalance" in columns
    assert "directed_microprice_displacement_bps" in columns
    assert "side_sign" in columns


def test_book_walk_forward_rejects_incomplete_book_rows() -> None:
    rows = []
    for day in range(7):
        day_start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
        for sample in range(40):
            signal = 1.0 if sample % 2 else -1.0
            rows.append(
                {
                    "decision_time_ms": int(
                        (day_start + pd.Timedelta(minutes=sample * 10)).timestamp() * 1000
                    ),
                    "horizon_seconds": 120,
                    "target": sample % 2,
                    "flow_imbalance_1s": signal,
                    "directed_flow_imbalance_1s": signal,
                    "trade_count_1s": 10,
                    "book_available": 0.0 if day == 3 and sample == 0 else 1.0,
                    "book_update_age_ms": 100.0,
                    "spread_bps": 2.0,
                    "depth_imbalance": signal * 0.1,
                    "directed_depth_imbalance": signal * 0.1,
                }
            )

    with pytest.raises(ValueError, match="complete book"):
        run_calibrated_walk_forward(
            pd.DataFrame(rows),
            horizon_seconds=120,
            feature_set="flow_book",
            embargo_seconds=300,
        )

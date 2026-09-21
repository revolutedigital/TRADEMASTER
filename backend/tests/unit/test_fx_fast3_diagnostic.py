"""Feature orientation and portfolio-policy tests for the round-3 diagnostic."""

import numpy as np
import pandas as pd

from app.fx import strategy as fx
from scripts.research.fx_fast3_diagnostic import apply_policy, model_matrix, select_thresholds


def test_model_matrix_orients_returns_but_preserves_spread_and_pair_identity() -> None:
    frame = pd.DataFrame({"return_3": [0.2], "spread_pips": [0.8]})

    matrix = model_matrix(frame, ["return_3", "spread_pips"], "EURUSD", fx.SHORT)

    assert matrix[0, 0] == np.float32(-0.2)
    assert matrix[0, 1] == np.float32(0.8)
    assert matrix[0, 2] == 1.0
    assert matrix[0, -1] == fx.SHORT


def test_policy_chooses_the_stronger_side_and_skips_overlapping_entries() -> None:
    times = pd.to_datetime(
        ["2022-01-03 10:00", "2022-01-03 10:00", "2022-01-03 10:05", "2022-01-03 11:00"],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "pair": ["EURUSD"] * 4,
            "side": [fx.LONG, fx.SHORT, fx.LONG, fx.LONG],
            "expected_r": [0.10, 0.20, 0.30, 0.11],
            "probability": [0.60] * 4,
            "base_r": [1.0, 2.0, 3.0, 4.0],
            "stress_r": [0.8, 1.8, 2.8, 3.8],
        },
        index=times,
    )

    trades = apply_policy(predictions, 60, 0.05, 0.55)

    assert trades["side"].tolist() == [fx.SHORT, fx.LONG]
    assert trades["base_r"].tolist() == [2.0, 4.0]


def test_no_eligible_threshold_is_a_recorded_no_trade_not_an_error() -> None:
    predictions = pd.DataFrame(
        {
            "pair": ["EURUSD"],
            "side": [fx.LONG],
            "expected_r": [-0.1],
            "probability": [0.4],
            "base_r": [1.0],
            "stress_r": [0.8],
        },
        index=pd.to_datetime(["2021-08-02"], utc=True),
    )

    assert select_thresholds(predictions, 60) is None


def test_trade_count_alone_cannot_select_a_losing_policy() -> None:
    times = pd.date_range("2021-07-01", periods=400, freq="h", tz="UTC")
    predictions = pd.DataFrame(
        {
            "pair": ["EURUSD"] * len(times),
            "side": [fx.LONG] * len(times),
            "expected_r": [0.2] * len(times),
            "probability": [0.8] * len(times),
            "base_r": [-0.2] * len(times),
            "stress_r": [-0.4] * len(times),
        },
        index=times,
    )

    assert select_thresholds(predictions, 15) is None

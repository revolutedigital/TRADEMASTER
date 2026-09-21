"""Tests for the pre-registered round-4 controls."""

import numpy as np
import pandas as pd

from scripts.research import fx_fast4_controls as controls


def test_intensity_buckets_are_fitted_only_from_supplied_training_values() -> None:
    edges = controls.intensity_edges(pd.Series(np.arange(10, dtype=float)))
    assert controls.intensity_bucket(pd.Series([0.0, 2.0, 4.0, 6.0, 9.0]), edges).tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_conditional_baseline_ignores_calibration_and_future_outcomes() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01T00:00:00Z",
            "2020-01-02T00:00:00Z",
            "2021-03-01T00:00:00Z",
            "2021-07-01T00:00:00Z",
            "2021-08-01T00:00:00Z",
        ]
    )
    features = pd.DataFrame(
        {
            "update_intensity_64": [1.0, 1.0, 1.0, 1.0, 1.0],
            "london_session": [1, 1, 1, 1, 1],
            "new_york_session": [0, 0, 0, 0, 0],
        },
        index=index,
    )
    outcomes = pd.DataFrame(
        {
            "h30_long_terminal_r_base": [0.1, 0.3, 99.0, -5.0, 7.0],
            "h30_long_terminal_r_stress": [-0.1, 0.1, 99.0, -6.0, 6.0],
        },
        index=index,
    )
    predicted = controls.conditional_predictions(
        features,
        outcomes,
        pair="EURUSD",
        side=1,
        side_name="long",
        horizon=30,
    )
    assert predicted.index.tolist() == index[3:].tolist()
    assert predicted["expected_r_base"].tolist() == [0.2, 0.2]
    assert predicted["expected_r_stress"].tolist() == [0.0, 0.0]
    assert predicted["expected_r"].tolist() == [0.0, 0.0]

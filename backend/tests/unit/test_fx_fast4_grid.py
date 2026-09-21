"""Declared grid coverage and evaluation bookkeeping."""

import pandas as pd

from scripts.research import fx_fast4_grid as grid


def test_m1_grid_contains_exactly_the_declared_eight_configurations() -> None:
    assert len(grid.M1_CONFIGS) == 8
    assert {(item.max_depth, item.learning_rate, item.n_estimators) for item in grid.M1_CONFIGS} == {
        (depth, learning_rate, estimators)
        for depth in (2, 3)
        for learning_rate in (0.03, 0.06)
        for estimators in (300, 700)
    }


def test_evaluation_counts_every_threshold_and_rejects_empty_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "pair": ["EURUSD"],
            "side": [1],
            "base_r": [0.2],
            "stress_r": [0.1],
            "expected_r_base": [-0.1],
            "expected_r_stress": [-0.2],
            "expected_r": [-0.2],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2021-07-01", tz="UTC")]),
    )
    result = grid._evaluate("test", predictions, 30)
    assert len(result["threshold_grid"]) == len(grid.EV_THRESHOLDS)
    assert not result["has_core_candidate"]

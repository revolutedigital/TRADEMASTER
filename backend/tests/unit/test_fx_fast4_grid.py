"""Declared grid coverage and evaluation bookkeeping."""

import pandas as pd
import numpy as np

from scripts.research import fx_fast4_grid as grid


def test_m1_grid_contains_exactly_the_declared_eight_configurations() -> None:
    assert len(grid.M1_CONFIGS) == 8
    assert {
        (item.max_depth, item.learning_rate, item.n_estimators) for item in grid.M1_CONFIGS
    } == {
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


def test_m1_uses_declared_early_stopping_sample(monkeypatch) -> None:
    fitted = []

    class FakeRegressor:
        def __init__(self, **kwargs) -> None:
            assert kwargs["early_stopping_rounds"] == 50
            fitted.append(self)

        def fit(self, matrix, target, *, eval_set, verbose) -> None:
            assert matrix.shape == (2, 1)
            assert target.shape == (2,)
            assert eval_set[0][0].shape == (1, 1)
            assert eval_set[0][1].shape == (1,)
            assert verbose is False

    monkeypatch.setattr(grid.xgb, "XGBRegressor", FakeRegressor)
    config = grid.M1Config(max_depth=2, learning_rate=0.03, n_estimators=300)
    grid._fit_m1(
        np.array([[1.0], [2.0]]),
        np.array([0.1, 0.2]),
        np.array([-0.1, -0.2]),
        np.array([[3.0]]),
        np.array([0.3]),
        np.array([-0.3]),
        config,
    )
    assert len(fitted) == 2

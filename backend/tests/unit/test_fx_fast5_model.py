"""Round-5 probability model boundaries and calibration."""

import numpy as np
import pandas as pd
import pytest

from scripts.research.fx_fast5_model import (
    PlattCalibrator,
    TimeWindow,
    fit_platt,
    model_matrix,
    period_mask,
)


def test_period_mask_applies_both_sides_of_embargo() -> None:
    index = pd.DatetimeIndex(
        ["2021-03-31T18:00:00Z", "2021-04-01T06:00:00Z", "2021-06-30T17:59:59Z"]
    )
    window = TimeWindow(pd.Timestamp("2021-04-01T06:00:00Z"), pd.Timestamp("2021-06-30T18:00:00Z"))
    assert period_mask(index, window).tolist() == [False, True, True]


def test_model_matrix_flips_only_directional_features() -> None:
    frame = pd.DataFrame({"mid_return_pips_16": [2.0], "spread_pips": [1.0]})
    names = ["mid_return_pips_16", "spread_pips"]
    assert model_matrix(frame, names, 1).tolist() == [[2.0, 1.0]]
    assert model_matrix(frame, names, -1).tolist() == [[-2.0, 1.0]]
    with pytest.raises(ValueError, match="side"):
        model_matrix(frame, names, 0)


def test_platt_calibrator_returns_ordered_bounded_probabilities() -> None:
    margin = np.array([-2.0, -1.0, 1.0, 2.0])
    target = np.array([0, 0, 1, 1])
    calibrator = fit_platt(margin, target)
    probability = calibrator.predict(margin)
    assert np.all(np.diff(probability) > 0)
    assert np.all((probability > 0) & (probability < 1))


def test_platt_calibrator_rejects_constant_target() -> None:
    with pytest.raises(ValueError, match="both target classes"):
        fit_platt(np.array([0.0, 1.0]), np.array([1, 1]))


def test_platt_prediction_is_numerically_stable() -> None:
    calibrator = PlattCalibrator(1.0, 0.0)
    probability = calibrator.predict(np.array([-1e6, 1e6]))
    assert probability[0] > 0
    assert probability[1] < 1

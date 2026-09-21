"""Round-5 probability model boundaries and calibration."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.research.fx_fast5_model import (
    PlattCalibrator,
    TimeWindow,
    fit_platt,
    load_pair_frame,
    model_matrix,
    period_mask,
)


def test_pair_loader_preserves_decision_index_as_metadata(tmp_path: Path) -> None:
    panel = tmp_path / "features"
    labels = tmp_path / "labels"
    panel.mkdir()
    labels.mkdir()
    timestamp = pd.Timestamp("2019-01-02T00:00:00Z")
    pd.DataFrame(
        {"decision_index": [42], "spread_pips": [1.0]},
        index=pd.DatetimeIndex([timestamp]),
    ).to_parquet(panel / "EURUSD-features.parquet")
    for year in (2019, 2020, 2021):
        year_timestamp = timestamp if year == 2019 else pd.Timestamp(f"{year}-01-02T00:00:00Z")
        if year != 2019:
            existing = pd.read_parquet(panel / "EURUSD-features.parquet")
            extra = pd.DataFrame(
                {"decision_index": [42 + year], "spread_pips": [1.0]},
                index=pd.DatetimeIndex([year_timestamp]),
            )
            pd.concat([existing, extra]).to_parquet(panel / "EURUSD-features.parquet")
        pd.DataFrame(
            {"entry_index": [43], "risk_pips": [2.0]},
            index=pd.DatetimeIndex([year_timestamp]),
        ).to_parquet(labels / f"EURUSD-{year}-entry-labels.parquet")
    frame, names = load_pair_frame(panel, labels, "EURUSD")
    assert "decision_index" in frame
    assert "decision_index" not in names


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

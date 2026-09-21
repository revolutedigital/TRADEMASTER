"""Audited local/global probability calibration for round 6."""

import numpy as np
import pandas as pd

from scripts.research.fx_fast6_model import (
    MIN_RELIABILITY_ROWS,
    ProbabilityCalibrator,
    build_reliability_map,
    fit_probability_calibrator,
    q1_absolute_floors,
    select_blend_calibration,
    strongest_direction,
    wilson_lower,
)


def test_all_calibrators_return_ordered_bounded_probabilities() -> None:
    raw = np.linspace(0.05, 0.95, 200)
    target = (raw > 0.55).astype(np.int8)
    for kind in ("platt", "isotonic", "beta"):
        calibrator = fit_probability_calibrator(kind, raw, target)
        calibrated = calibrator.predict(raw)
        assert np.all((calibrated >= 0) & (calibrated <= 1))
        assert np.all(np.diff(calibrated) >= -1e-12)


def test_wilson_and_reliability_are_conservative() -> None:
    assert 0 < wilson_lower(70, 100) < 0.70
    probability = np.concatenate(
        (
            np.linspace(0.55, 0.599, MIN_RELIABILITY_ROWS),
            np.full(MIN_RELIABILITY_ROWS - 1, 0.75),
        )
    )
    target = np.concatenate(
        (np.array([1] * 15 + [0] * (MIN_RELIABILITY_ROWS - 15)), np.ones(MIN_RELIABILITY_ROWS - 1))
    )
    reliability = build_reliability_map(probability, target)
    trusted = reliability.trusted(np.array([0.56, 0.59, 0.75]))
    assert 0 < trusted[0] < trusted[1] < 0.59
    assert trusted[2] == 0.0


def test_blend_selection_uses_global_signal_when_local_is_uninformative() -> None:
    target = np.array([0, 0, 0, 1, 1, 1] * 20, dtype=np.int8)
    good = np.where(target == 1, 0.8, 0.2).astype(float)
    bad = np.full(len(target), 0.5)
    selected, attempts = select_blend_calibration(bad, good, target, bad, good, target)
    assert selected.local_weight < 1.0
    assert selected.eligible
    assert len(attempts) == 15


def test_strongest_direction_and_q1_floor_are_global() -> None:
    index = pd.DatetimeIndex(
        ["2021-03-20T10:00:00Z", "2021-03-20T10:00:00Z", "2021-03-20T10:00:01Z"],
        name="decision_time",
    )
    frame = pd.DataFrame(
        {
            "pair": ["EURUSD", "EURUSD", "GBPUSD"],
            "side": [1, -1, 1],
            "trusted_score": [0.2, 0.4, 0.3],
            "point_score": [0.3, 0.5, 0.4],
            "model_eligible": [True, True, True],
        },
        index=index,
    )
    strongest = strongest_direction(frame)
    assert strongest["trusted_score"].tolist() == [0.4, 0.3]
    floors = q1_absolute_floors(frame)
    assert floors["0.1"] <= floors["0.0025"]


def test_calibrator_rejects_unknown_kind() -> None:
    calibrator = ProbabilityCalibrator("unknown", (), 0.0)
    try:
        calibrator.predict(np.array([0.5]))
    except ValueError as error:
        assert "unknown calibrator" in str(error)
    else:
        raise AssertionError("unknown calibrator was accepted")

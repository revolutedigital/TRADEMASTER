"""Tests for the cheap managed-payoff learnability gate."""

import numpy as np

from scripts.research.fx_fast9_pilot import (
    ContinuousCalibrator,
    audit_metrics,
    pilot_gate,
)


def test_affine_calibrator_clips_economically_extreme_predictions() -> None:
    calibrator = ContinuousCalibrator(kind="affine", slope=2.0, intercept=0.5)
    prediction = calibrator.predict(np.asarray([-10.0, 10.0]))
    assert prediction.tolist() == [-1.25, 2.0]


def test_pilot_gate_accepts_only_material_audit_improvement() -> None:
    target = np.linspace(-1.0, 1.0, 1_000)
    metrics = audit_metrics(target * 0.95, target)
    assert metrics["relative_mse_improvement"] > 0.01
    assert metrics["spearman"] > 0.02
    assert metrics["top_decile_lift"] > 0.05
    assert pilot_gate(metrics)


def test_pilot_gate_rejects_constant_prediction() -> None:
    target = np.linspace(-1.0, 1.0, 1_000)
    metrics = audit_metrics(np.zeros_like(target), target)
    assert not pilot_gate(metrics)

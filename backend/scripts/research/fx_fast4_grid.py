"""Exhaust the pre-registered B2/M1 model grid without opening 2022 or protected samples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_diagnostic import (
    CALIBRATION_END,
    DEFAULT_OUTPUT,
    DEFAULT_PANEL,
    EV_THRESHOLDS,
    SELECTION_END,
    TRAIN_END,
    _eligible,
    _outcome_columns,
    apply_policy,
    core_selection_gate,
    load_training,
    metrics,
    model_feature_names,
    model_matrix,
    read_pair_panel,
)
from scripts.research.fx_fast4_events import HORIZONS_SECONDS


@dataclass(frozen=True)
class M1Config:
    max_depth: int
    learning_rate: float
    n_estimators: int

    @property
    def identifier(self) -> str:
        learning = str(self.learning_rate).replace(".", "p")
        return f"m1-d{self.max_depth}-lr{learning}-n{self.n_estimators}"


M1_CONFIGS = tuple(
    M1Config(depth, learning_rate, estimators)
    for depth in (2, 3)
    for learning_rate in (0.03, 0.06)
    for estimators in (300, 700)
)


def load_period(
    panel: Path,
    horizon: int,
    feature_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[np.ndarray, pd.DataFrame]:
    matrices: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    columns = _outcome_columns(horizon)
    for pair in ALL_PAIRS:
        features, outcomes = read_pair_panel(panel, pair, columns)
        modeling = features[feature_names]
        for side, side_name in ((1, "long"), (-1, "short")):
            base = outcomes[f"h{horizon}_{side_name}_terminal_r_base"]
            valid = _eligible(modeling, base, start, end)
            selected = modeling.loc[valid]
            matrices.append(model_matrix(selected, feature_names, pair, side))
            metadata.append(
                pd.DataFrame(
                    {
                        "pair": pair,
                        "side": side,
                        "base_r": base.loc[valid].to_numpy(),
                        "stress_r": outcomes.loc[
                            valid, f"h{horizon}_{side_name}_terminal_r_stress"
                        ].to_numpy(),
                    },
                    index=selected.index,
                )
            )
    return np.concatenate(matrices), pd.concat(metadata).sort_index(kind="stable")


def load_selection(
    panel: Path, horizon: int, feature_names: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    return load_period(panel, horizon, feature_names, CALIBRATION_END, SELECTION_END)


def _prediction_frame(
    metadata: pd.DataFrame, expected_base: np.ndarray, expected_stress: np.ndarray
) -> pd.DataFrame:
    predictions = metadata.copy()
    predictions["expected_r_base"] = expected_base
    predictions["expected_r_stress"] = expected_stress
    predictions["expected_r"] = np.minimum(expected_base, expected_stress)
    return predictions


def _evaluate(identifier: str, predictions: pd.DataFrame, horizon: int) -> dict[str, object]:
    threshold_grid = []
    for threshold in EV_THRESHOLDS:
        result = metrics(apply_policy(predictions, horizon, threshold))
        threshold_grid.append(
            {"expected_r": threshold, **asdict(result), "core_gate": core_selection_gate(result)}
        )
    return {
        "model": identifier,
        "score_quantiles": {
            str(quantile): float(predictions["expected_r"].quantile(quantile))
            for quantile in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "threshold_grid": threshold_grid,
        "has_core_candidate": any(item["core_gate"] for item in threshold_grid),
    }


def _fit_m1(
    matrix: np.ndarray,
    base_target: np.ndarray,
    stress_target: np.ndarray,
    calibration: np.ndarray,
    calibration_base: np.ndarray,
    calibration_stress: np.ndarray,
    config: M1Config,
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    common = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 100,
        "reg_lambda": 10.0,
        "tree_method": "hist",
        "n_jobs": 10,
        "random_state": 20260921,
        "early_stopping_rounds": 50,
    }
    base = xgb.XGBRegressor(objective="reg:squarederror", **common)
    stress = xgb.XGBRegressor(objective="reg:squarederror", **common)
    base.fit(matrix, base_target, eval_set=[(calibration, calibration_base)], verbose=False)
    stress.fit(matrix, stress_target, eval_set=[(calibration, calibration_stress)], verbose=False)
    return base, stress


def run_grid(panel: Path, output: Path, horizon: int, stride: int) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    training, base_target, stress_target, feature_names = load_training(panel, horizon, stride)
    calibration, calibration_metadata = load_period(
        panel, horizon, feature_names, TRAIN_END, CALIBRATION_END
    )
    selection, metadata = load_selection(panel, horizon, feature_names)
    attempts: list[dict[str, object]] = []

    ridge_base = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr", tol=1e-4))
    ridge_stress = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr", tol=1e-4))
    ridge_base.fit(training, base_target)
    ridge_stress.fit(training, stress_target)
    attempts.append(
        _evaluate(
            "b2-ridge-alpha10",
            _prediction_frame(
                metadata, ridge_base.predict(selection), ridge_stress.predict(selection)
            ),
            horizon,
        )
    )
    sys.stdout.write(f"h{horizon} b2-ridge-alpha10 complete\n")
    sys.stdout.flush()

    for config in M1_CONFIGS:
        base_model, stress_model = _fit_m1(
            training,
            base_target,
            stress_target,
            calibration,
            calibration_metadata["base_r"].to_numpy(),
            calibration_metadata["stress_r"].to_numpy(),
            config,
        )
        attempt = _evaluate(
            config.identifier,
            _prediction_frame(
                metadata, base_model.predict(selection), stress_model.predict(selection)
            ),
            horizon,
        )
        attempt["best_iteration_base"] = int(base_model.best_iteration)
        attempt["best_iteration_stress"] = int(stress_model.best_iteration)
        attempts.append(attempt)
        sys.stdout.write(f"h{horizon} {config.identifier} complete\n")
        sys.stdout.flush()

    core_candidates = [attempt for attempt in attempts if attempt["has_core_candidate"]]
    report: dict[str, object] = {
        "kind": "exhaustive_declared_grid_diagnostic",
        "horizon_seconds": horizon,
        "training_rows": len(training),
        "calibration_rows": len(calibration),
        "selection_rows": len(selection),
        "training_stride": stride,
        "attempts": attempts,
        "attempt_count": len(attempts) * len(EV_THRESHOLDS),
        "core_candidate_count": len(core_candidates),
        "status": "core_candidate_requires_full_selection"
        if core_candidates
        else "no_core_candidate",
        "year_2022_opened": False,
        "protected_samples_opened": False,
        "pbo_dsr": "not computed because no core candidate" if not core_candidates else "required",
    }
    (output / f"h{horizon}-grid-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, choices=HORIZONS_SECONDS, required=True)
    parser.add_argument("--stride", type=int, default=8)
    arguments = parser.parse_args(argv)
    if arguments.stride < 1:
        parser.error("--stride must be positive")
    report = run_grid(arguments.panel, arguments.output, arguments.horizon, arguments.stride)
    sys.stdout.write(
        json.dumps(
            {
                "horizon_seconds": report["horizon_seconds"],
                "attempt_count": report["attempt_count"],
                "core_candidate_count": report["core_candidate_count"],
                "status": report["status"],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

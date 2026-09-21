"""Development-only XGBoost diagnostic for the round-3 conditional signal hypothesis.

This is deliberately not the pre-registered model-selection run: it uses a single fixed model and
chronological 2021 calibration/selection plus 2022 evaluation to verify that the panel contains a
learnable conditional edge before spending the full nested-walk-forward budget. It cannot open D3.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from app.fx import strategy as fx
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast3_events import DIRECTIONAL_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = REPO_ROOT / "backend" / "data" / "lab_fast3" / "development"
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast3" / "diagnostics"
TRAIN_END = pd.Timestamp("2021-01-01", tz="UTC")
CALIBRATION_END = pd.Timestamp("2021-07-01", tz="UTC")
SELECTION_END = pd.Timestamp("2022-01-01", tz="UTC")
TEST_END = pd.Timestamp("2023-01-01", tz="UTC")
EV_THRESHOLDS = (0.03, 0.05, 0.08, 0.12, 0.18)
PROBABILITY_THRESHOLDS = (0.52, 0.55, 0.58, 0.62)
EXCLUDED_FEATURES = frozenset({"atr_price", "history_contiguous"})


@dataclass(frozen=True)
class PolicyMetrics:
    trades: int
    mean_base_r: float
    mean_stress_r: float
    positive_pairs: int
    positive_month_fraction: float


def model_feature_names(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name not in EXCLUDED_FEATURES]


def model_matrix(frame: pd.DataFrame, feature_names: list[str], pair: str, side: int) -> np.ndarray:
    """Orient signed features and append pair/side identity without changing input."""
    matrix = frame[feature_names].to_numpy(dtype=np.float32, copy=True)
    for column, name in enumerate(feature_names):
        if name in DIRECTIONAL_FEATURES:
            matrix[:, column] *= side
    pair_columns = np.zeros((len(frame), len(ALL_PAIRS)), dtype=np.float32)
    pair_columns[:, list(ALL_PAIRS).index(pair)] = 1.0
    side_column = np.full((len(frame), 1), side, dtype=np.float32)
    return np.concatenate((matrix, pair_columns, side_column), axis=1)


def expanded_feature_names(feature_names: list[str]) -> list[str]:
    return [*feature_names, *(f"pair_{pair}" for pair in ALL_PAIRS), "side"]


def _eligible(features: pd.DataFrame, outcome: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp) -> np.ndarray:
    feature_values = features.to_numpy(dtype=np.float32, copy=False)
    valid = np.isfinite(feature_values).all(axis=1) & outcome.notna().to_numpy()
    valid &= features.index < end
    if start is not None:
        valid &= features.index >= start
    return valid


def load_training(panel: Path, horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    matrices: list[np.ndarray] = []
    regression_targets: list[np.ndarray] = []
    classification_targets: list[np.ndarray] = []
    names: list[str] | None = None
    for pair_index, pair in enumerate(ALL_PAIRS):
        features = pd.read_parquet(panel / f"{pair}-features.parquet")
        names = names or model_feature_names(features)
        modeling = features[names]
        outcomes = pd.read_parquet(
            panel / f"{pair}-outcomes.parquet",
            columns=[f"h{horizon}_long_terminal_r_base", f"h{horizon}_short_terminal_r_base"],
        )
        for side, side_name in ((fx.LONG, "long"), (fx.SHORT, "short")):
            target = outcomes[f"h{horizon}_{side_name}_terminal_r_base"]
            valid = _eligible(modeling, target, None, TRAIN_END)
            positions = np.flatnonzero(valid)
            # Deterministic time/pair/side offset; selection never reads the target value.
            offset = (2 * pair_index + (0 if side == fx.LONG else 1)) % stride
            positions = positions[offset::stride]
            selected = modeling.iloc[positions]
            y = target.iloc[positions].to_numpy(dtype=np.float32)
            matrices.append(model_matrix(selected, names, pair, side))
            regression_targets.append(y)
            classification_targets.append((y > 0).astype(np.int8))
    if names is None:
        raise ValueError("the panel contains no pairs")
    return (
        np.concatenate(matrices),
        np.concatenate(regression_targets),
        np.concatenate(classification_targets),
        names,
    )


def fit_models(
    matrix: np.ndarray, regression_target: np.ndarray, classification_target: np.ndarray
) -> tuple[xgb.XGBRegressor, xgb.XGBClassifier]:
    common = {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_lambda": 10.0,
        "tree_method": "hist",
        "n_jobs": 10,
        "random_state": 20260921,
    }
    regressor = xgb.XGBRegressor(objective="reg:squarederror", **common)
    classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    regressor.fit(matrix, regression_target, verbose=False)
    classifier.fit(matrix, classification_target, verbose=False)
    return regressor, classifier


def predict_period(
    panel: Path,
    horizon: int,
    regressor: xgb.XGBRegressor,
    classifier: xgb.XGBClassifier,
    feature_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for pair in ALL_PAIRS:
        features = pd.read_parquet(panel / f"{pair}-features.parquet")
        modeling = features[feature_names]
        outcomes = pd.read_parquet(
            panel / f"{pair}-outcomes.parquet",
            columns=[
                f"h{horizon}_long_terminal_r_base",
                f"h{horizon}_long_terminal_r_stress",
                f"h{horizon}_short_terminal_r_base",
                f"h{horizon}_short_terminal_r_stress",
            ],
        )
        for side, side_name in ((fx.LONG, "long"), (fx.SHORT, "short")):
            base = outcomes[f"h{horizon}_{side_name}_terminal_r_base"]
            valid = _eligible(modeling, base, start, end)
            selected = modeling.loc[valid]
            matrix = model_matrix(selected, feature_names, pair, side)
            parts.append(
                pd.DataFrame(
                    {
                        "pair": pair,
                        "side": side,
                        "expected_r": regressor.predict(matrix),
                        "raw_probability": classifier.predict_proba(matrix)[:, 1],
                        "base_r": base.loc[valid].to_numpy(),
                        "stress_r": outcomes.loc[
                            valid, f"h{horizon}_{side_name}_terminal_r_stress"
                        ].to_numpy(),
                    },
                    index=selected.index,
                )
            )
    return pd.concat(parts).sort_index()


def apply_policy(
    predictions: pd.DataFrame,
    horizon: int,
    ev_threshold: float,
    probability_threshold: float,
) -> pd.DataFrame:
    """Choose the stronger side and enforce one non-overlapping position per pair."""
    selected_parts: list[pd.DataFrame] = []
    holding = pd.Timedelta(minutes=horizon)
    for pair, pair_frame in predictions.groupby("pair", sort=False):
        candidates = pair_frame[
            (pair_frame["expected_r"] >= ev_threshold)
            & (pair_frame["probability"] >= probability_threshold)
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values("expected_r").groupby(level=0).tail(1).sort_index()
        keep = np.zeros(len(candidates), dtype=bool)
        available_at = pd.Timestamp.min.tz_localize("UTC")
        for position, timestamp in enumerate(candidates.index):
            if timestamp >= available_at:
                keep[position] = True
                available_at = timestamp + holding
        chosen = candidates.iloc[keep].copy()
        chosen["pair"] = pair
        selected_parts.append(chosen)
    return pd.concat(selected_parts).sort_index() if selected_parts else predictions.iloc[:0].copy()


def metrics(trades: pd.DataFrame) -> PolicyMetrics:
    if trades.empty:
        return PolicyMetrics(0, float("nan"), float("nan"), 0, 0.0)
    pair_means = trades.groupby("pair")["base_r"].mean()
    month = trades.index.tz_localize(None).to_period("M")
    month_means = trades.groupby(month)["base_r"].mean()
    return PolicyMetrics(
        trades=len(trades),
        mean_base_r=float(trades["base_r"].mean()),
        mean_stress_r=float(trades["stress_r"].mean()),
        positive_pairs=int((pair_means > 0).sum()),
        positive_month_fraction=float((month_means > 0).mean()),
    )


def threshold_grid(predictions: pd.DataFrame, horizon: int) -> list[tuple[float, float, PolicyMetrics]]:
    """Evaluate the declared grid, including cells with too few trades for selection."""
    candidates: list[tuple[float, float, PolicyMetrics]] = []
    for ev_threshold in EV_THRESHOLDS:
        for probability_threshold in PROBABILITY_THRESHOLDS:
            result = metrics(apply_policy(predictions, horizon, ev_threshold, probability_threshold))
            candidates.append((ev_threshold, probability_threshold, result))
    return candidates


def select_thresholds(
    predictions: pd.DataFrame, horizon: int
) -> tuple[float, float, PolicyMetrics] | None:
    eligible = [candidate for candidate in threshold_grid(predictions, horizon) if candidate[2].trades >= 300]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[2].mean_stress_r, item[2].mean_base_r))


def run_diagnostic(panel: Path, output: Path, horizon: int, stride: int) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    matrix, regression_target, classification_target, names = load_training(panel, horizon, stride)
    regressor, classifier = fit_models(matrix, regression_target, classification_target)
    validation = predict_period(
        panel, horizon, regressor, classifier, names, TRAIN_END, SELECTION_END
    )
    calibration = validation.index < CALIBRATION_END
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(validation.loc[calibration, "raw_probability"], (validation.loc[calibration, "base_r"] > 0).astype(int))
    validation["probability"] = calibrator.predict(validation["raw_probability"])
    selection = validation.loc[~calibration]
    grid = threshold_grid(selection, horizon)
    chosen = select_thresholds(selection, horizon)
    importance = sorted(
        zip(expanded_feature_names(names), regressor.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    report: dict[str, object] = {
        "kind": "development_diagnostic_not_candidate_selection",
        "horizon_minutes": horizon,
        "training_stride": stride,
        "training_rows": len(matrix),
        "selection_score_quantiles": {
            "expected_r": {
                str(quantile): float(selection["expected_r"].quantile(quantile))
                for quantile in (0.5, 0.9, 0.99, 0.999, 1.0)
            },
            "probability": {
                str(quantile): float(selection["probability"].quantile(quantile))
                for quantile in (0.5, 0.9, 0.99, 0.999, 1.0)
            },
        },
        "threshold_grid": [
            {
                "expected_r": ev,
                "probability": probability,
                **asdict(grid_metrics),
            }
            for ev, probability, grid_metrics in grid
        ],
        "top_regression_features": [{"feature": name, "gain": float(gain)} for name, gain in importance],
    }
    if chosen is None:
        report["status"] = "no_policy_with_300_validation_trades"
        report["thresholds"] = None
        report["selection_2021_h2"] = None
        report["test_2022"] = "not_opened"
    else:
        ev_threshold, probability_threshold, selection_metrics = chosen
        test = predict_period(panel, horizon, regressor, classifier, names, SELECTION_END, TEST_END)
        test["probability"] = calibrator.predict(test["raw_probability"])
        test_trades = apply_policy(test, horizon, ev_threshold, probability_threshold)
        report["status"] = "policy_selected"
        report["thresholds"] = {
            "expected_r": ev_threshold,
            "probability": probability_threshold,
        }
        report["selection_2021_h2"] = asdict(selection_metrics)
        report["test_2022"] = asdict(metrics(test_trades))
    regressor.save_model(output / f"h{horizon}-regressor.json")
    classifier.save_model(output / f"h{horizon}-classifier.json")
    (output / f"h{horizon}-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, choices=(15, 60, 180, 360), default=60)
    parser.add_argument("--stride", type=int, default=8)
    arguments = parser.parse_args(argv)
    if arguments.stride < 1:
        parser.error("--stride must be positive")
    report = run_diagnostic(arguments.panel, arguments.output, arguments.horizon, arguments.stride)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the cheap round-9 learnability pilot before any Q2 portfolio simulation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_diagnostic import model_feature_names, model_matrix
from scripts.research.fx_fast4_materialize import _sha256
from scripts.research.fx_fast5_materialize import DEFAULT_FEATURE_PANEL
from scripts.research.fx_fast5_model import _read_features
from scripts.research.fx_fast9_materialize import DEFAULT_OUTPUT as PAYOFF_ROOT

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast9" / "pilot"
RANDOM_SEED = 20260921
PURGE = pd.Timedelta(hours=6)
PREDICTION_MIN_R = -1.25
PREDICTION_MAX_R = 2.0
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CALIBRATORS = ("identity", "affine", "isotonic")


@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp | None
    end: pd.Timestamp


WINDOWS = {
    "train": TimeWindow(None, pd.Timestamp("2021-01-01", tz="UTC") - PURGE),
    "early_stop": TimeWindow(
        pd.Timestamp("2021-01-01", tz="UTC") + PURGE,
        pd.Timestamp("2021-02-15", tz="UTC") - PURGE,
    ),
    "calibration": TimeWindow(
        pd.Timestamp("2021-02-15", tz="UTC") + PURGE,
        pd.Timestamp("2021-03-15", tz="UTC") - PURGE,
    ),
    "audit": TimeWindow(
        pd.Timestamp("2021-03-15", tz="UTC") + PURGE,
        pd.Timestamp("2021-04-01", tz="UTC") - PURGE,
    ),
}


@dataclass(frozen=True)
class ContinuousCalibrator:
    kind: str
    slope: float = 1.0
    intercept: float = 0.0
    x_thresholds: tuple[float, ...] = ()
    y_thresholds: tuple[float, ...] = ()

    def predict(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64)
        if self.kind == "identity":
            calibrated = values
        elif self.kind == "affine":
            calibrated = self.slope * values + self.intercept
        elif self.kind == "isotonic":
            calibrated = np.interp(
                values,
                np.asarray(self.x_thresholds),
                np.asarray(self.y_thresholds),
            )
        else:
            raise ValueError(f"unknown calibrator: {self.kind}")
        return np.clip(calibrated, PREDICTION_MIN_R, PREDICTION_MAX_R)


def fit_calibrator(kind: str, raw: np.ndarray, target: np.ndarray) -> ContinuousCalibrator:
    if kind == "identity":
        return ContinuousCalibrator(kind="identity")
    if kind == "affine":
        model = LinearRegression().fit(raw.reshape(-1, 1), target)
        return ContinuousCalibrator(
            kind="affine", slope=float(model.coef_[0]), intercept=float(model.intercept_)
        )
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=PREDICTION_MIN_R, y_max=2.0)
        model.fit(raw, target)
        return ContinuousCalibrator(
            kind="isotonic",
            x_thresholds=tuple(float(value) for value in model.X_thresholds_),
            y_thresholds=tuple(float(value) for value in model.y_thresholds_),
        )
    raise ValueError(f"unknown calibrator: {kind}")


def audit_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    realized = np.asarray(target, dtype=np.float64)
    baseline = np.full(len(realized), realized.mean())
    baseline_mse = float(mean_squared_error(realized, baseline))
    mse = float(mean_squared_error(realized, predicted))
    rank_correlation = (
        float(pd.Series(predicted).corr(pd.Series(realized), method="spearman"))
        if np.unique(predicted).size > 1
        else 0.0
    )
    cutoff = float(np.quantile(predicted, 0.9))
    top = realized[predicted >= cutoff]
    overall_mean = float(realized.mean())
    top_mean = float(top.mean())
    return {
        "rows": len(realized),
        "mse": mse,
        "mae": float(mean_absolute_error(realized, predicted)),
        "baseline_mse": baseline_mse,
        "relative_mse_improvement": (baseline_mse - mse) / baseline_mse,
        "mean_prediction": float(predicted.mean()),
        "mean_realized": overall_mean,
        "absolute_mean_error": abs(float(predicted.mean()) - overall_mean),
        "spearman": rank_correlation,
        "top_decile_realized": top_mean,
        "top_decile_lift": top_mean - overall_mean,
    }


def pilot_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["relative_mse_improvement"] >= 0.01
        and metrics["spearman"] > 0.02
        and metrics["absolute_mean_error"] <= 0.05
        and metrics["top_decile_lift"] >= 0.05
    )


def _regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=700,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=100,
        reg_lambda=10.0,
        tree_method="hist",
        n_jobs=10,
        random_state=RANDOM_SEED,
        early_stopping_rounds=50,
    )


def _classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=700,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=100,
        reg_lambda=10.0,
        tree_method="hist",
        n_jobs=10,
        random_state=RANDOM_SEED,
        early_stopping_rounds=50,
    )


def _window_mask(index: pd.DatetimeIndex, window: TimeWindow) -> np.ndarray:
    mask = np.asarray(index < window.end)
    if window.start is not None:
        mask &= np.asarray(index >= window.start)
    return mask


def load_pair_dataset(
    pair: str, feature_panel: Path, payoff_root: Path
) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, np.ndarray], list[str]]:
    features = _read_features(feature_panel, pair)
    payoff_paths = [
        payoff_root / f"{pair}-{year}-managed-payoffs.parquet" for year in (2019, 2020, 2021)
    ]
    missing = [path.name for path in payoff_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"pilot payoff partitions missing for {pair}: {missing}")
    payoffs = pd.concat(pd.read_parquet(path) for path in payoff_paths).sort_index()
    names = model_feature_names(features)
    matrices: list[np.ndarray] = []
    indexes: list[pd.DatetimeIndex] = []
    columns: dict[str, list[np.ndarray]] = {
        f"{kind}_{scenario}": []
        for kind in ("return", "activated")
        for scenario in ("base", "stress")
    }
    for side, side_name in ((1, "long"), (-1, "short")):
        side_targets = [
            f"{side_name}_{kind}_{scenario}"
            for kind in ("result_r", "activated")
            for scenario in ("base", "stress")
        ]
        valid = payoffs[side_targets].notna().all(axis=1)
        selected = payoffs.loc[valid]
        selected_features = features.loc[selected.index]
        matrix = model_matrix(selected_features, names, pair, side)
        pair_columns = np.zeros((len(matrix), len(ALL_PAIRS)), dtype=np.float32)
        pair_columns[:, list(ALL_PAIRS).index(pair)] = 1.0
        matrices.append(
            np.concatenate(
                (matrix, pair_columns, np.full((len(matrix), 1), side, dtype=np.float32)), axis=1
            )
        )
        indexes.append(pd.DatetimeIndex(selected.index))
        for scenario in ("base", "stress"):
            columns[f"return_{scenario}"].append(
                selected[f"{side_name}_result_r_{scenario}"].to_numpy(dtype=np.float32)
            )
            columns[f"activated_{scenario}"].append(
                selected[f"{side_name}_activated_{scenario}"].to_numpy(dtype=np.int8)
            )
    matrix = np.concatenate(matrices)
    index = indexes[0].append(indexes[1:])
    targets = {name: np.concatenate(parts) for name, parts in columns.items()}
    order = np.argsort(index.asi8, kind="stable")
    return matrix[order], index[order], {name: value[order] for name, value in targets.items()}, names


def _fit_scenario(
    matrix: np.ndarray,
    index: pd.DatetimeIndex,
    target: np.ndarray,
    activated: np.ndarray,
    scenario: str,
    output: Path,
) -> dict[str, object]:
    masks = {name: _window_mask(index, window) for name, window in WINDOWS.items()}
    direct = _regressor()
    direct.fit(
        matrix[masks["train"]],
        target[masks["train"]],
        eval_set=[(matrix[masks["early_stop"]], target[masks["early_stop"]])],
        verbose=False,
    )
    activation = _classifier()
    activation.fit(
        matrix[masks["train"]],
        activated[masks["train"]],
        eval_set=[(matrix[masks["early_stop"]], activated[masks["early_stop"]])],
        verbose=False,
    )
    conditional_models: dict[int, xgb.XGBRegressor] = {}
    for state in (0, 1):
        train_mask = masks["train"] & (activated == state)
        stop_mask = masks["early_stop"] & (activated == state)
        model = _regressor()
        model.fit(
            matrix[train_mask],
            target[train_mask],
            eval_set=[(matrix[stop_mask], target[stop_mask])],
            verbose=False,
        )
        conditional_models[state] = model

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for window_name in ("calibration", "audit"):
        selected = matrix[masks[window_name]]
        direct_prediction = direct.predict(selected)
        probability = activation.predict_proba(selected)[:, 1]
        failure = conditional_models[0].predict(selected)
        runner = conditional_models[1].predict(selected)
        predictions[window_name] = {
            "direct": direct_prediction,
            "hurdle": probability * runner + (1 - probability) * failure,
        }

    calibration_target = target[masks["calibration"]]
    audit_target = target[masks["audit"]]
    attempts: list[dict[str, object]] = []
    candidates: list[tuple[float, float, float, ContinuousCalibrator, dict[str, float]]] = []
    for hurdle_weight in BLEND_WEIGHTS:
        calibration_raw = (
            hurdle_weight * predictions["calibration"]["hurdle"]
            + (1 - hurdle_weight) * predictions["calibration"]["direct"]
        )
        audit_raw = (
            hurdle_weight * predictions["audit"]["hurdle"]
            + (1 - hurdle_weight) * predictions["audit"]["direct"]
        )
        for kind in CALIBRATORS:
            calibrator = fit_calibrator(kind, calibration_raw, calibration_target)
            metrics = audit_metrics(calibrator.predict(audit_raw), audit_target)
            attempts.append(
                {"hurdle_weight": hurdle_weight, "calibrator": kind, **metrics}
            )
            candidates.append(
                (metrics["mse"], metrics["mae"], -hurdle_weight, calibrator, metrics)
            )
    mse, mae, negative_weight, calibrator, selected_metrics = min(candidates)
    hurdle_weight = -negative_weight
    model_paths: dict[str, str] = {}
    models = {
        "direct": direct,
        "activation": activation,
        "failure": conditional_models[0],
        "runner": conditional_models[1],
    }
    for name, model in models.items():
        path = output / f"{scenario}-{name}.json"
        model.save_model(path)
        model_paths[name] = _sha256(path)
    return {
        "scenario": scenario,
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "best_iterations": {name: int(model.best_iteration) for name, model in models.items()},
        "model_sha256": model_paths,
        "selected": {
            "hurdle_weight": hurdle_weight,
            "calibrator": asdict(calibrator),
            **selected_metrics,
            "pilot_gate": pilot_gate(selected_metrics),
        },
        "attempts": attempts,
    }


def run_pilot(
    pair: str = "EURUSD",
    feature_panel: Path = DEFAULT_FEATURE_PANEL,
    payoff_root: Path = PAYOFF_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    if pair not in ALL_PAIRS:
        raise ValueError(f"unknown pair: {pair}")
    output = output / pair
    output.mkdir(parents=True, exist_ok=True)
    matrix, index, targets, names = load_pair_dataset(pair, feature_panel, payoff_root)
    scenarios = []
    for scenario in ("base", "stress"):
        report = _fit_scenario(
            matrix,
            index,
            targets[f"return_{scenario}"],
            targets[f"activated_{scenario}"],
            scenario,
            output,
        )
        scenarios.append(report)
        selected = report["selected"]
        sys.stdout.write(
            f"{pair} {scenario}: improvement={selected['relative_mse_improvement']:.4f}, "
            f"spearman={selected['spearman']:.4f}, lift={selected['top_decile_lift']:.4f}, "
            f"gate={selected['pilot_gate']}\n"
        )
        sys.stdout.flush()
        gc.collect()
    passed = all(bool(report["selected"]["pilot_gate"]) for report in scenarios)
    final = {
        "kind": "round9_p0_managed_payoff_learnability",
        "pair": pair,
        "protected_samples_opened": False,
        "q2_opened": False,
        "features": names,
        "directional_rows": len(matrix),
        "scenarios": scenarios,
        "status": "p0_pass" if passed else "p0_rejected",
    }
    report_path = output / "pilot-report.json"
    report_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"round9 P0: {final['status']}, sha256={_sha256(report_path)}\n")
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--feature-panel", type=Path, default=DEFAULT_FEATURE_PANEL)
    parser.add_argument("--payoff-root", type=Path, default=PAYOFF_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    run_pilot(arguments.pair.upper(), arguments.feature_panel, arguments.payoff_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

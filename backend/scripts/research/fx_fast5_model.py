"""Train per-pair directional probability models for the round-5 runner study."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_diagnostic import model_feature_names
from scripts.research.fx_fast4_events import DIRECTIONAL_FEATURES
from scripts.research.fx_fast4_materialize import _sha256
from scripts.research.fx_fast5_materialize import (
    DEFAULT_FEATURE_PANEL,
    DEFAULT_OUTPUT as LABEL_ROOT,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast5" / "models"
RANDOM_SEED = 20260921
PURGE = pd.Timedelta(hours=6)
TRAIN_END = pd.Timestamp("2021-01-01", tz="UTC")
CALIBRATION_END = pd.Timestamp("2021-04-01", tz="UTC")
SELECTION_END = pd.Timestamp("2021-07-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2022-01-01", tz="UTC")


@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp | None
    end: pd.Timestamp


WINDOWS = {
    "train": TimeWindow(None, TRAIN_END - PURGE),
    "calibration": TimeWindow(TRAIN_END + PURGE, CALIBRATION_END - PURGE),
    "selection": TimeWindow(CALIBRATION_END + PURGE, SELECTION_END - PURGE),
    "validation": TimeWindow(SELECTION_END + PURGE, VALIDATION_END - PURGE),
}


@dataclass(frozen=True)
class PlattCalibrator:
    coefficient: float
    intercept: float

    def predict(self, margin: np.ndarray) -> np.ndarray:
        linear = np.clip(self.coefficient * margin + self.intercept, -36.0, 36.0)
        return 1.0 / (1.0 + np.exp(-linear))


def _read_features(panel: Path, pair: str) -> pd.DataFrame:
    whole = panel / f"{pair}-features.parquet"
    if whole.exists():
        frame = pd.read_parquet(whole)
    else:
        paths = sorted(panel.glob(f"{pair}-????-features.parquet"))
        if not paths:
            raise FileNotFoundError(f"round-4 features not found for {pair}")
        frame = pd.concat(pd.read_parquet(path) for path in paths).sort_index()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"invalid feature index for {pair}")
    return frame.loc[frame.index < VALIDATION_END]


def _read_labels(label_root: Path, pair: str) -> pd.DataFrame:
    paths = [label_root / f"{pair}-{year}-entry-labels.parquet" for year in (2019, 2020, 2021)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"round-5 labels missing for {pair}: {missing}")
    frame = pd.concat(pd.read_parquet(path) for path in paths).sort_index()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"invalid label index for {pair}")
    return frame


def load_pair_frame(
    feature_panel: Path, label_root: Path, pair: str
) -> tuple[pd.DataFrame, list[str]]:
    features = _read_features(feature_panel, pair)
    labels = _read_labels(label_root, pair)
    if not labels.index.isin(features.index).all():
        raise ValueError(f"labels do not align with features for {pair}")
    names = model_feature_names(features)
    combined = features.loc[labels.index, ["decision_index", *names]].join(
        labels, how="inner", validate="one_to_one"
    )
    if len(combined) != len(labels):
        raise ValueError(f"feature/label join lost rows for {pair}")
    return combined, names


def period_mask(index: pd.DatetimeIndex, window: TimeWindow) -> np.ndarray:
    mask = index < window.end
    if window.start is not None:
        mask &= index >= window.start
    return np.asarray(mask)


def model_matrix(frame: pd.DataFrame, names: list[str], side: int) -> np.ndarray:
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    matrix = frame[names].to_numpy(dtype=np.float32, copy=True)
    for column, name in enumerate(names):
        if name in DIRECTIONAL_FEATURES:
            matrix[:, column] *= side
    return matrix


def eligible_mask(
    frame: pd.DataFrame, names: list[str], target: str, window: TimeWindow
) -> np.ndarray:
    finite = np.isfinite(frame[names].to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    finite &= frame[target].notna().to_numpy()
    finite &= period_mask(frame.index, window)
    return finite


def fit_platt(margin: np.ndarray, target: np.ndarray) -> PlattCalibrator:
    if len(np.unique(target)) != 2:
        raise ValueError("Platt calibration requires both target classes")
    logistic = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000, random_state=RANDOM_SEED)
    logistic.fit(margin.reshape(-1, 1), target)
    return PlattCalibrator(float(logistic.coef_[0, 0]), float(logistic.intercept_[0]))


def fit_classifier(
    training: np.ndarray,
    training_target: np.ndarray,
    calibration: np.ndarray,
    calibration_target: np.ndarray,
) -> tuple[xgb.XGBClassifier, PlattCalibrator]:
    model = xgb.XGBClassifier(
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
    model.fit(
        training,
        training_target,
        eval_set=[(calibration, calibration_target)],
        verbose=False,
    )
    margin = model.predict(calibration, output_margin=True)
    return model, fit_platt(margin, calibration_target)


def calibrated_probability(
    model: xgb.XGBClassifier, calibrator: PlattCalibrator, matrix: np.ndarray
) -> np.ndarray:
    return calibrator.predict(model.predict(matrix, output_margin=True))


def _calibration_report(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(len(target)),
        "positive_rate": float(np.mean(target)),
        "mean_probability": float(np.mean(probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
    }


def train_pair(
    pair: str,
    feature_panel: Path,
    label_root: Path,
    output: Path,
) -> dict[str, object]:
    frame, names = load_pair_frame(feature_panel, label_root, pair)
    output.mkdir(parents=True, exist_ok=True)
    prediction_parts: list[pd.DataFrame] = []
    model_reports: list[dict[str, object]] = []
    for side, side_name in ((1, "long"), (-1, "short")):
        scenario_probabilities: dict[str, np.ndarray] = {}
        selection_index: pd.DatetimeIndex | None = None
        for scenario in ("base", "stress"):
            target_name = f"{side_name}_hit_{scenario}"
            train_mask = eligible_mask(frame, names, target_name, WINDOWS["train"])
            calibration_mask = eligible_mask(frame, names, target_name, WINDOWS["calibration"])
            selection_mask = eligible_mask(frame, names, target_name, WINDOWS["selection"])
            training = model_matrix(frame.loc[train_mask], names, side)
            calibration = model_matrix(frame.loc[calibration_mask], names, side)
            selection = model_matrix(frame.loc[selection_mask], names, side)
            training_target = frame.loc[train_mask, target_name].to_numpy(dtype=np.int8)
            calibration_target = frame.loc[calibration_mask, target_name].to_numpy(dtype=np.int8)
            model, calibrator = fit_classifier(
                training, training_target, calibration, calibration_target
            )
            calibration_probability = calibrated_probability(model, calibrator, calibration)
            selection_probability = calibrated_probability(model, calibrator, selection)
            current_index = frame.index[selection_mask]
            if selection_index is not None and not current_index.equals(selection_index):
                raise ValueError(f"base/stress selection rows differ for {pair} {side_name}")
            selection_index = current_index
            scenario_probabilities[scenario] = selection_probability
            artifact = f"{pair}-{side_name}-{scenario}"
            model_path = output / f"{artifact}.json"
            model.save_model(model_path)
            model_reports.append(
                {
                    "pair": pair,
                    "side": side_name,
                    "scenario": scenario,
                    "training_rows": len(training),
                    "training_positive_rate": float(np.mean(training_target)),
                    "best_iteration": int(model.best_iteration),
                    "calibrator": asdict(calibrator),
                    "calibration": _calibration_report(calibration_probability, calibration_target),
                    "selection_rows": len(selection),
                    "model_sha256": _sha256(model_path),
                }
            )
            del training, calibration, selection, model
            gc.collect()
        if selection_index is None:
            raise ValueError(f"no selection rows for {pair} {side_name}")
        side_frame = frame.loc[
            selection_index,
            ["decision_index", "entry_index", "risk_pips", "mid_range_pips_256"],
        ].copy()
        side_frame["pair"] = pair
        side_frame["side"] = side
        side_frame["probability_base"] = scenario_probabilities["base"]
        side_frame["probability_stress"] = scenario_probabilities["stress"]
        side_frame["score"] = np.minimum(
            scenario_probabilities["base"], scenario_probabilities["stress"]
        )
        prediction_parts.append(side_frame)
    predictions = pd.concat(prediction_parts).sort_index(kind="stable")
    prediction_path = output / f"{pair}-q2-probabilities.parquet"
    predictions.to_parquet(prediction_path, compression="zstd", index=True)
    report: dict[str, object] = {
        "pair": pair,
        "features": names,
        "feature_count": len(names),
        "models": model_reports,
        "selection_rows": len(predictions),
        "score_quantiles": {
            str(quantile): float(predictions["score"].quantile(quantile))
            for quantile in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
        },
        "threshold_counts": {
            str(threshold): int((predictions["score"] >= threshold).sum())
            for threshold in (0.55, 0.60, 0.65, 0.70)
        },
        "predictions_sha256": _sha256(prediction_path),
    }
    report_path = output / f"{pair}-model-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_sha256"] = _sha256(report_path)
    return report


def verified_existing_report(pair: str, output: Path) -> dict[str, object] | None:
    report_path = output / f"{pair}-model-report.json"
    prediction_path = output / f"{pair}-q2-probabilities.parquet"
    if not report_path.exists() or not prediction_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("pair") != pair or report.get("predictions_sha256") != _sha256(prediction_path):
        return None
    models = report.get("models")
    if not isinstance(models, list) or len(models) != 4:
        return None
    for item in models:
        if not isinstance(item, dict):
            return None
        artifact = f"{pair}-{item.get('side')}-{item.get('scenario')}.json"
        model_path = output / artifact
        if not model_path.exists() or item.get("model_sha256") != _sha256(model_path):
            return None
    report["report_sha256"] = _sha256(report_path)
    return report


def train_all(
    pairs: tuple[str, ...],
    feature_panel: Path = DEFAULT_FEATURE_PANEL,
    label_root: Path = LABEL_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    if set(pairs) - set(ALL_PAIRS):
        raise ValueError("unknown pair")
    reports = []
    for pair in pairs:
        report = verified_existing_report(pair, output)
        if report is None:
            report = train_pair(pair, feature_panel, label_root, output)
            status = "trained"
        else:
            status = "verified existing"
        reports.append(report)
        sys.stdout.write(
            f"{pair}: {status}, q2={report['selection_rows']:,}, "
            f"candidates={report['threshold_counts']}\n"
        )
        sys.stdout.flush()
    manifest = {
        "kind": "round5_probability_models_q2_only",
        "protected_samples_opened": False,
        "year_2022_opened": False,
        "windows": {
            name: {
                "start": window.start.isoformat() if window.start is not None else None,
                "end": window.end.isoformat(),
            }
            for name, window in WINDOWS.items()
        },
        "pairs": reports,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(payload, encoding="utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=",".join(ALL_PAIRS))
    parser.add_argument("--feature-panel", type=Path, default=DEFAULT_FEATURE_PANEL)
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    pairs = tuple(value.strip().upper() for value in arguments.pairs.split(",") if value.strip())
    train_all(pairs, arguments.feature_panel, arguments.label_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

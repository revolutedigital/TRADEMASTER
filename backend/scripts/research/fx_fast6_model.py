"""Train audited local/global probability models for the global Top-P runner study."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_diagnostic import model_matrix as global_model_matrix
from scripts.research.fx_fast4_materialize import _sha256
from scripts.research.fx_fast5_materialize import (
    DEFAULT_FEATURE_PANEL,
    DEFAULT_OUTPUT as LABEL_ROOT,
)
from scripts.research.fx_fast5_model import eligible_mask, load_pair_frame, model_matrix

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast6" / "models"
RANDOM_SEED = 20260921
PURGE = pd.Timedelta(hours=6)
GLOBAL_STRIDE = 4
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CALIBRATOR_KINDS = ("platt", "isotonic", "beta")
TOP_FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
RELIABILITY_BINS = 20
MIN_RELIABILITY_ROWS = 30
WILSON_Z = 1.6448536269514722


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
    "reference": TimeWindow(
        pd.Timestamp("2021-01-01", tz="UTC") + PURGE,
        pd.Timestamp("2021-07-01", tz="UTC") - PURGE,
    ),
}


@dataclass(frozen=True)
class ProbabilityCalibrator:
    kind: str
    coefficients: tuple[float, ...]
    intercept: float
    x_thresholds: tuple[float, ...] = ()
    y_thresholds: tuple[float, ...] = ()

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(raw_probability, dtype=np.float64), 1e-6, 1 - 1e-6)
        if self.kind == "platt":
            feature = np.log(probability / (1 - probability))
            linear = self.coefficients[0] * feature + self.intercept
            return _sigmoid(linear)
        if self.kind == "beta":
            linear = (
                self.coefficients[0] * np.log(probability)
                + self.coefficients[1] * -np.log1p(-probability)
                + self.intercept
            )
            return _sigmoid(linear)
        if self.kind == "isotonic":
            return np.interp(
                probability,
                np.asarray(self.x_thresholds),
                np.asarray(self.y_thresholds),
            )
        raise ValueError(f"unknown calibrator kind: {self.kind}")


@dataclass(frozen=True)
class ReliabilityMap:
    lower_bounds: tuple[float, ...]
    adjustment_factors: tuple[float, ...]
    counts: tuple[int, ...]

    def trusted(self, probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
        bins = np.minimum((values * RELIABILITY_BINS).astype(np.int64), RELIABILITY_BINS - 1)
        factors = np.asarray(self.adjustment_factors)[bins]
        return values * factors


@dataclass(frozen=True)
class BlendCalibration:
    local_weight: float
    calibrator: ProbabilityCalibrator
    reliability: ReliabilityMap
    audit_brier: float
    audit_log_loss: float
    baseline_brier: float
    eligible: bool


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -36.0, 36.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _calibration_features(kind: str, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    if kind == "platt":
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if kind == "beta":
        return np.column_stack((np.log(clipped), -np.log1p(-clipped)))
    raise ValueError(f"unsupported logistic calibrator: {kind}")


def fit_probability_calibrator(
    kind: str, raw_probability: np.ndarray, target: np.ndarray
) -> ProbabilityCalibrator:
    if len(np.unique(target)) != 2:
        raise ValueError("probability calibration requires both target classes")
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(raw_probability, target)
        return ProbabilityCalibrator(
            kind="isotonic",
            coefficients=(),
            intercept=0.0,
            x_thresholds=tuple(float(value) for value in model.X_thresholds_),
            y_thresholds=tuple(float(value) for value in model.y_thresholds_),
        )
    features = _calibration_features(kind, raw_probability)
    logistic = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000, random_state=RANDOM_SEED)
    logistic.fit(features, target)
    return ProbabilityCalibrator(
        kind=kind,
        coefficients=tuple(float(value) for value in logistic.coef_[0]),
        intercept=float(logistic.intercept_[0]),
    )


def wilson_lower(successes: int, rows: int, z: float = WILSON_Z) -> float:
    if rows <= 0:
        return 0.0
    proportion = successes / rows
    denominator = 1 + z * z / rows
    centre = proportion + z * z / (2 * rows)
    spread = z * math.sqrt(proportion * (1 - proportion) / rows + z * z / (4 * rows * rows))
    return max(0.0, (centre - spread) / denominator)


def build_reliability_map(probability: np.ndarray, target: np.ndarray) -> ReliabilityMap:
    values = np.clip(np.asarray(probability), 0.0, 1.0)
    bins = np.minimum((values * RELIABILITY_BINS).astype(np.int64), RELIABILITY_BINS - 1)
    lower: list[float] = []
    factors: list[float] = []
    counts: list[int] = []
    for bin_index in range(RELIABILITY_BINS):
        selected_mask = bins == bin_index
        selected = target[selected_mask]
        rows = len(selected)
        counts.append(rows)
        bound = wilson_lower(int(selected.sum()), rows) if rows >= MIN_RELIABILITY_ROWS else 0.0
        lower.append(bound)
        mean_probability = float(values[selected_mask].mean()) if rows else 0.0
        factors.append(min(1.0, bound / mean_probability) if mean_probability > 0 else 0.0)
    return ReliabilityMap(tuple(lower), tuple(factors), tuple(counts))


def select_blend_calibration(
    local_calibration: np.ndarray,
    global_calibration: np.ndarray,
    calibration_target: np.ndarray,
    local_audit: np.ndarray,
    global_audit: np.ndarray,
    audit_target: np.ndarray,
) -> tuple[BlendCalibration, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    candidates: list[tuple[float, float, float, ProbabilityCalibrator, np.ndarray]] = []
    for local_weight in BLEND_WEIGHTS:
        fit_probability = local_weight * local_calibration + (1 - local_weight) * global_calibration
        audit_probability = local_weight * local_audit + (1 - local_weight) * global_audit
        for kind in CALIBRATOR_KINDS:
            calibrator = fit_probability_calibrator(kind, fit_probability, calibration_target)
            calibrated = np.clip(calibrator.predict(audit_probability), 1e-6, 1 - 1e-6)
            brier = float(brier_score_loss(audit_target, calibrated))
            loss = float(log_loss(audit_target, calibrated, labels=[0, 1]))
            attempts.append(
                {
                    "local_weight": local_weight,
                    "calibrator": kind,
                    "audit_brier": brier,
                    "audit_log_loss": loss,
                }
            )
            candidates.append((brier, loss, local_weight, calibrator, calibrated))
    brier, loss, local_weight, calibrator, calibrated = min(
        candidates, key=lambda item: (item[0], item[1], -item[2])
    )
    base_rate = float(np.mean(audit_target))
    baseline_brier = float(brier_score_loss(audit_target, np.full(len(audit_target), base_rate)))
    selected = BlendCalibration(
        local_weight=local_weight,
        calibrator=calibrator,
        reliability=build_reliability_map(calibrated, audit_target),
        audit_brier=brier,
        audit_log_loss=loss,
        baseline_brier=baseline_brier,
        eligible=brier < baseline_brier,
    )
    return selected, attempts


def fit_classifier(
    training: np.ndarray,
    training_target: np.ndarray,
    early_stop: np.ndarray,
    early_stop_target: np.ndarray,
) -> xgb.XGBClassifier:
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
        eval_set=[(early_stop, early_stop_target)],
        verbose=False,
    )
    return model


def _raw_probability(model: xgb.XGBClassifier, matrix: np.ndarray) -> np.ndarray:
    return model.predict_proba(matrix)[:, 1].astype(np.float64)


def _scenario_target(side_name: str, scenario: str) -> str:
    return f"{side_name}_hit_{scenario}"


def _window_data(
    frame: pd.DataFrame,
    names: list[str],
    target_name: str,
    window_name: str,
    pair: str,
    side: int,
    *,
    global_matrix: bool,
    stride: int = 1,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    mask = eligible_mask(frame, names, target_name, WINDOWS[window_name])
    positions = np.flatnonzero(mask)[offset::stride]
    selected = frame.iloc[positions]
    matrix = (
        global_model_matrix(selected, names, pair, side)
        if global_matrix
        else model_matrix(selected, names, side)
    )
    target = selected[target_name].to_numpy(dtype=np.int8)
    return matrix, target, selected.index


def load_frames(
    feature_panel: Path, label_root: Path, pairs: tuple[str, ...]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    names: list[str] | None = None
    for pair in pairs:
        frame, pair_names = load_pair_frame(feature_panel, label_root, pair)
        if names is None:
            names = pair_names
        elif pair_names != names:
            raise ValueError(f"feature columns differ for {pair}")
        frames[pair] = frame
    if names is None:
        raise ValueError("no pair frames loaded")
    return frames, names


def train_global_models(
    frames: dict[str, pd.DataFrame], names: list[str], output: Path
) -> dict[str, xgb.XGBClassifier]:
    models: dict[str, xgb.XGBClassifier] = {}
    for scenario in ("base", "stress"):
        training_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        stop_parts: list[np.ndarray] = []
        stop_target_parts: list[np.ndarray] = []
        for pair_index, (pair, frame) in enumerate(frames.items()):
            for side_index, (side, side_name) in enumerate(((1, "long"), (-1, "short"))):
                target_name = _scenario_target(side_name, scenario)
                offset = (2 * pair_index + side_index) % GLOBAL_STRIDE
                training, target, _ = _window_data(
                    frame,
                    names,
                    target_name,
                    "train",
                    pair,
                    side,
                    global_matrix=True,
                    stride=GLOBAL_STRIDE,
                    offset=offset,
                )
                early_stop, stop_target, _ = _window_data(
                    frame,
                    names,
                    target_name,
                    "early_stop",
                    pair,
                    side,
                    global_matrix=True,
                    stride=GLOBAL_STRIDE,
                    offset=offset,
                )
                training_parts.append(training)
                target_parts.append(target)
                stop_parts.append(early_stop)
                stop_target_parts.append(stop_target)
        model = fit_classifier(
            np.concatenate(training_parts),
            np.concatenate(target_parts),
            np.concatenate(stop_parts),
            np.concatenate(stop_target_parts),
        )
        model_path = output / f"global-{scenario}.json"
        model.save_model(model_path)
        models[scenario] = model
        del training_parts, target_parts, stop_parts, stop_target_parts
        gc.collect()
    return models


def _serialize_blend(blend: BlendCalibration) -> dict[str, object]:
    return {
        "local_weight": blend.local_weight,
        "calibrator": asdict(blend.calibrator),
        "reliability": asdict(blend.reliability),
        "audit_brier": blend.audit_brier,
        "audit_log_loss": blend.audit_log_loss,
        "baseline_brier": blend.baseline_brier,
        "eligible": blend.eligible,
    }


def train_pair(
    pair: str,
    frame: pd.DataFrame,
    names: list[str],
    global_models: dict[str, xgb.XGBClassifier],
    output: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_parts: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    for side, side_name in ((1, "long"), (-1, "short")):
        scenario_predictions: dict[str, np.ndarray] = {}
        scenario_trusted: dict[str, np.ndarray] = {}
        reference_index: pd.DatetimeIndex | None = None
        side_eligible = True
        for scenario in ("base", "stress"):
            target_name = _scenario_target(side_name, scenario)
            matrices: dict[str, np.ndarray] = {}
            targets: dict[str, np.ndarray] = {}
            indexes: dict[str, pd.DatetimeIndex] = {}
            for window_name in ("train", "early_stop", "calibration", "audit", "reference"):
                matrix, target, index = _window_data(
                    frame,
                    names,
                    target_name,
                    window_name,
                    pair,
                    side,
                    global_matrix=False,
                )
                matrices[window_name] = matrix
                targets[window_name] = target
                indexes[window_name] = index
            local_model = fit_classifier(
                matrices["train"],
                targets["train"],
                matrices["early_stop"],
                targets["early_stop"],
            )
            local_path = output / f"{pair}-{side_name}-{scenario}.json"
            local_model.save_model(local_path)
            local_calibration = _raw_probability(local_model, matrices["calibration"])
            local_audit = _raw_probability(local_model, matrices["audit"])

            global_calibration_matrix = global_model_matrix(
                frame.loc[indexes["calibration"]], names, pair, side
            )
            global_audit_matrix = global_model_matrix(
                frame.loc[indexes["audit"]], names, pair, side
            )
            blend, attempts = select_blend_calibration(
                local_calibration,
                _raw_probability(global_models[scenario], global_calibration_matrix),
                targets["calibration"],
                local_audit,
                _raw_probability(global_models[scenario], global_audit_matrix),
                targets["audit"],
            )
            reference_global = global_model_matrix(
                frame.loc[indexes["reference"]], names, pair, side
            )
            reference_raw = blend.local_weight * _raw_probability(
                local_model, matrices["reference"]
            ) + (1 - blend.local_weight) * _raw_probability(
                global_models[scenario], reference_global
            )
            probability = blend.calibrator.predict(reference_raw)
            scenario_predictions[scenario] = probability
            scenario_trusted[scenario] = blend.reliability.trusted(probability)
            if reference_index is not None and not indexes["reference"].equals(reference_index):
                raise ValueError(f"scenario rows differ for {pair} {side_name}")
            reference_index = indexes["reference"]
            side_eligible &= blend.eligible
            reports.append(
                {
                    "pair": pair,
                    "side": side_name,
                    "scenario": scenario,
                    "training_rows": len(matrices["train"]),
                    "early_stop_rows": len(matrices["early_stop"]),
                    "calibration_rows": len(matrices["calibration"]),
                    "audit_rows": len(matrices["audit"]),
                    "best_iteration": int(local_model.best_iteration),
                    "local_model_sha256": _sha256(local_path),
                    "selected": _serialize_blend(blend),
                    "calibration_attempts": attempts,
                }
            )
            del matrices, targets, local_model
            gc.collect()
        if reference_index is None:
            raise ValueError(f"no reference rows for {pair} {side_name}")
        selected = frame.loc[
            reference_index,
            ["decision_index", "entry_index", "risk_pips", "mid_range_pips_256"],
        ].copy()
        selected["pair"] = pair
        selected["side"] = side
        selected["probability_base"] = scenario_predictions["base"]
        selected["probability_stress"] = scenario_predictions["stress"]
        selected["trusted_base"] = scenario_trusted["base"]
        selected["trusted_stress"] = scenario_trusted["stress"]
        selected["point_score"] = np.minimum(
            scenario_predictions["base"], scenario_predictions["stress"]
        )
        selected["trusted_score"] = np.minimum(scenario_trusted["base"], scenario_trusted["stress"])
        selected["model_eligible"] = side_eligible
        prediction_parts.append(selected)
    predictions = pd.concat(prediction_parts).sort_index(kind="stable")
    return predictions, {"pair": pair, "models": reports}


def strongest_direction(predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = predictions.loc[predictions["model_eligible"]].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.assign(_decision_time_key=eligible.index)
        .sort_values(["trusted_score", "point_score"], kind="stable")
        .groupby(["_decision_time_key", "pair"], sort=False)
        .tail(1)
        .drop(columns="_decision_time_key")
        .sort_index(kind="stable")
    )


def q1_absolute_floors(predictions: pd.DataFrame) -> dict[str, float]:
    q1 = strongest_direction(
        predictions.loc[predictions.index < pd.Timestamp("2021-04-01", tz="UTC")]
    )
    if q1.empty:
        raise ValueError("no eligible Q1 predictions for absolute floors")
    return {
        str(fraction): float(q1["trusted_score"].quantile(1 - fraction))
        for fraction in TOP_FRACTIONS
    }


def train_all(
    pairs: tuple[str, ...],
    feature_panel: Path = DEFAULT_FEATURE_PANEL,
    label_root: Path = LABEL_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    if set(pairs) - set(ALL_PAIRS):
        raise ValueError("unknown pair")
    if set(pairs) != set(ALL_PAIRS):
        raise ValueError("global round-6 calibration requires all declared pairs")
    output.mkdir(parents=True, exist_ok=True)
    frames, names = load_frames(feature_panel, label_root, pairs)
    global_models = train_global_models(frames, names, output)
    prediction_parts: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    for pair in pairs:
        predictions, report = train_pair(pair, frames[pair], names, global_models, output)
        prediction_parts.append(predictions)
        reports.append(report)
        eligible = int(predictions["model_eligible"].sum())
        sys.stdout.write(f"{pair}: {eligible:,}/{len(predictions):,} eligible directional rows\n")
        sys.stdout.flush()
        del frames[pair]
        gc.collect()
    predictions = pd.concat(prediction_parts).sort_index(kind="stable")
    prediction_path = output / "reference-probabilities.parquet"
    predictions.to_parquet(prediction_path, compression="zstd", index=True)
    floors = q1_absolute_floors(predictions)
    manifest = {
        "kind": "round6_audited_local_global_probabilities",
        "protected_samples_opened": False,
        "year_2022_opened": False,
        "features": names,
        "windows": {
            name: {
                "start": window.start.isoformat() if window.start is not None else None,
                "end": window.end.isoformat(),
            }
            for name, window in WINDOWS.items()
        },
        "global_models": {
            scenario: {
                "best_iteration": int(model.best_iteration),
                "sha256": _sha256(output / f"global-{scenario}.json"),
            }
            for scenario, model in global_models.items()
        },
        "absolute_floors": floors,
        "pairs": reports,
        "predictions_sha256": _sha256(prediction_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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

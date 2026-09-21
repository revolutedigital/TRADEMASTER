"""Development-only learnability diagnostic for round-4 quote events.

This fixed XGBoost probe reads only training through 2020 and 2021 calibration/selection. It never
opens 2022, D4, S3, or the broker week and cannot lock a policy.
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

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_events import DIRECTIONAL_FEATURES, HORIZONS_SECONDS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = REPO_ROOT / "backend" / "data" / "lab_fast4" / "development"
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast4" / "diagnostics"
TRAIN_END = pd.Timestamp("2021-01-01", tz="UTC")
CALIBRATION_END = pd.Timestamp("2021-07-01", tz="UTC")
SELECTION_END = pd.Timestamp("2022-01-01", tz="UTC")
EV_THRESHOLDS = (0.03, 0.05, 0.08, 0.12)
EXCLUDED_FEATURES = frozenset({"decision_index"})


@dataclass(frozen=True)
class PolicyMetrics:
    trades: int
    mean_base_r: float
    mean_stress_r: float
    lower_95_base_r: float
    positive_pairs: int
    positive_month_fraction: float
    maximum_pair_profit_fraction: float


def model_feature_names(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name not in EXCLUDED_FEATURES]


def model_matrix(frame: pd.DataFrame, feature_names: list[str], pair: str, side: int) -> np.ndarray:
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


def _eligible(
    features: pd.DataFrame,
    outcome: pd.Series,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> np.ndarray:
    valid = np.isfinite(features.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    valid &= outcome.notna().to_numpy()
    valid &= features.index < end
    if start is not None:
        valid &= features.index >= start
    return valid


def _outcome_columns(horizon: int) -> list[str]:
    return [
        f"h{horizon}_{side}_{cost}"
        for side in ("long_terminal_r", "short_terminal_r")
        for cost in ("base", "stress")
    ]


def load_training(
    panel: Path, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    matrices: list[np.ndarray] = []
    base_targets: list[np.ndarray] = []
    stress_targets: list[np.ndarray] = []
    names: list[str] | None = None
    for pair_index, pair in enumerate(ALL_PAIRS):
        features = pd.read_parquet(panel / f"{pair}-features.parquet")
        names = names or model_feature_names(features)
        modeling = features[names]
        outcomes = pd.read_parquet(panel / f"{pair}-outcomes.parquet", columns=_outcome_columns(horizon))
        for side, side_name in ((1, "long"), (-1, "short")):
            base = outcomes[f"h{horizon}_{side_name}_terminal_r_base"]
            valid = _eligible(modeling, base, None, TRAIN_END)
            positions = np.flatnonzero(valid)
            offset = (2 * pair_index + (0 if side == 1 else 1)) % stride
            positions = positions[offset::stride]
            selected = modeling.iloc[positions]
            matrices.append(model_matrix(selected, names, pair, side))
            base_targets.append(base.iloc[positions].to_numpy(dtype=np.float32))
            stress_targets.append(
                outcomes[f"h{horizon}_{side_name}_terminal_r_stress"]
                .iloc[positions]
                .to_numpy(dtype=np.float32)
            )
    if names is None:
        raise ValueError("the panel contains no pairs")
    return np.concatenate(matrices), np.concatenate(base_targets), np.concatenate(stress_targets), names


def fit_models(
    matrix: np.ndarray, base_target: np.ndarray, stress_target: np.ndarray
) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    common = {
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 100,
        "reg_lambda": 10.0,
        "tree_method": "hist",
        "n_jobs": 10,
        "random_state": 20260921,
    }
    base_model = xgb.XGBRegressor(objective="reg:squarederror", **common)
    stress_model = xgb.XGBRegressor(objective="reg:squarederror", **common)
    base_model.fit(matrix, base_target, verbose=False)
    stress_model.fit(matrix, stress_target, verbose=False)
    return base_model, stress_model


def predict_period(
    panel: Path,
    horizon: int,
    base_model: xgb.XGBRegressor,
    stress_model: xgb.XGBRegressor,
    feature_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for pair in ALL_PAIRS:
        features = pd.read_parquet(panel / f"{pair}-features.parquet")
        modeling = features[feature_names]
        outcomes = pd.read_parquet(panel / f"{pair}-outcomes.parquet", columns=_outcome_columns(horizon))
        for side, side_name in ((1, "long"), (-1, "short")):
            base = outcomes[f"h{horizon}_{side_name}_terminal_r_base"]
            valid = _eligible(modeling, base, start, end)
            selected = modeling.loc[valid]
            matrix = model_matrix(selected, feature_names, pair, side)
            expected_base = base_model.predict(matrix)
            expected_stress = stress_model.predict(matrix)
            parts.append(
                pd.DataFrame(
                    {
                        "pair": pair,
                        "side": side,
                        "expected_r": np.minimum(expected_base, expected_stress),
                        "expected_r_base": expected_base,
                        "expected_r_stress": expected_stress,
                        "base_r": base.loc[valid].to_numpy(),
                        "stress_r": outcomes.loc[
                            valid, f"h{horizon}_{side_name}_terminal_r_stress"
                        ].to_numpy(),
                    },
                    index=selected.index,
                )
            )
    return pd.concat(parts).sort_index()


def apply_policy(predictions: pd.DataFrame, horizon: int, ev_threshold: float) -> pd.DataFrame:
    """Choose the stronger side and enforce one non-overlapping position per pair."""
    selected_parts: list[pd.DataFrame] = []
    holding = pd.Timedelta(seconds=horizon)
    for pair, pair_frame in predictions.groupby("pair", sort=False):
        candidates = pair_frame[pair_frame["expected_r"] >= ev_threshold].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values("expected_r").groupby(level=0).tail(1).sort_index()
        keep = np.zeros(len(candidates), dtype=bool)
        available_at = pd.Timestamp.min.tz_localize("UTC")
        for position, timestamp in enumerate(candidates.index):
            if timestamp >= available_at:
                keep[position] = True
                available_at = timestamp + holding
        selected_parts.append(candidates.iloc[keep])
    return pd.concat(selected_parts).sort_index() if selected_parts else predictions.iloc[:0].copy()


def _fx_day(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert("America/New_York") - pd.Timedelta(hours=17)
    return local.tz_localize(None).normalize().to_numpy()


def stationary_bootstrap_lower_bound(trades: pd.DataFrame, *, samples: int = 2_000) -> float:
    if trades.empty:
        return float("nan")
    daily = trades.assign(fx_day=_fx_day(trades.index)).groupby("fx_day")["base_r"].agg(["sum", "count"])
    sums = daily["sum"].to_numpy(dtype=np.float64)
    counts = daily["count"].to_numpy(dtype=np.float64)
    if len(sums) < 2:
        return float("nan")
    rng = np.random.default_rng(20260921)
    means = np.empty(samples, dtype=np.float64)
    restart_probability = 0.2
    for sample in range(samples):
        position = int(rng.integers(len(sums)))
        total, count = 0.0, 0.0
        for _ in range(len(sums)):
            if rng.random() < restart_probability:
                position = int(rng.integers(len(sums)))
            total += sums[position]
            count += counts[position]
            position = (position + 1) % len(sums)
        means[sample] = total / count
    return float(np.quantile(means, 0.05))


def metrics(trades: pd.DataFrame) -> PolicyMetrics:
    if trades.empty:
        return PolicyMetrics(0, *(float("nan"),) * 3, 0, 0.0, float("nan"))
    pair_profit = trades.groupby("pair")["base_r"].sum()
    total_profit = float(trades["base_r"].sum())
    concentration = (
        float(pair_profit.clip(lower=0).max() / total_profit) if total_profit > 0 else float("inf")
    )
    month = trades.index.tz_localize(None).to_period("M")
    return PolicyMetrics(
        trades=len(trades),
        mean_base_r=float(trades["base_r"].mean()),
        mean_stress_r=float(trades["stress_r"].mean()),
        lower_95_base_r=stationary_bootstrap_lower_bound(trades),
        positive_pairs=int((trades.groupby("pair")["base_r"].mean() > 0).sum()),
        positive_month_fraction=float((trades.groupby(month)["base_r"].mean() > 0).mean()),
        maximum_pair_profit_fraction=concentration,
    )


def core_selection_gate(result: PolicyMetrics) -> bool:
    return (
        result.trades >= 1_500
        and result.mean_base_r > 0
        and result.mean_stress_r > 0
        and result.lower_95_base_r > 0
        and result.positive_pairs >= 7
        and result.positive_month_fraction >= 0.6
        and result.maximum_pair_profit_fraction <= 0.35
    )


def run_diagnostic(panel: Path, output: Path, horizon: int, stride: int) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    matrix, base_target, stress_target, names = load_training(panel, horizon, stride)
    base_model, stress_model = fit_models(matrix, base_target, stress_target)
    validation = predict_period(
        panel, horizon, base_model, stress_model, names, TRAIN_END, SELECTION_END
    )
    selection = validation.loc[validation.index >= CALIBRATION_END]
    grid = []
    candidates = []
    for threshold in EV_THRESHOLDS:
        result = metrics(apply_policy(selection, horizon, threshold))
        item = {"expected_r": threshold, **asdict(result), "core_gate": core_selection_gate(result)}
        grid.append(item)
        if item["core_gate"]:
            candidates.append(item)
    importance = sorted(
        zip(expanded_feature_names(names), stress_model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    report: dict[str, object] = {
        "kind": "development_diagnostic_not_candidate_selection",
        "protected_samples_opened": False,
        "year_2022_opened": False,
        "horizon_seconds": horizon,
        "training_stride": stride,
        "training_rows": len(matrix),
        "selection_score_quantiles": {
            str(quantile): float(selection["expected_r"].quantile(quantile))
            for quantile in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "threshold_grid": grid,
        "status": "candidate_for_full_selection" if candidates else "no_core_candidate",
        "top_stress_features": [
            {"feature": name, "gain": float(gain)} for name, gain in importance
        ],
    }
    artifact = f"h{horizon}"
    base_model.save_model(output / f"{artifact}-base-regressor.json")
    stress_model.save_model(output / f"{artifact}-stress-regressor.json")
    (output / f"{artifact}-report.json").write_text(
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
    report = run_diagnostic(arguments.panel, arguments.output, arguments.horizon, arguments.stride)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


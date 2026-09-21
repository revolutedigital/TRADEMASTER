"""Evaluate the pre-registered B0/B1 controls without opening 2022 or protected samples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_diagnostic import (
    CALIBRATION_END,
    DEFAULT_OUTPUT,
    DEFAULT_PANEL,
    EV_THRESHOLDS,
    SELECTION_END,
    TRAIN_END,
    _outcome_columns,
    apply_policy,
    core_selection_gate,
    metrics,
)
from scripts.research.fx_fast4_events import HORIZONS_SECONDS

CONTROL_FEATURES = ("update_intensity_64", "london_session", "new_york_session")


def intensity_edges(values: pd.Series) -> np.ndarray:
    """Return deterministic training-only cut points for five intensity buckets."""
    finite = values[np.isfinite(values.to_numpy(dtype=np.float64, copy=False))]
    if finite.empty:
        raise ValueError("B1 intensity has no finite training observations")
    return np.quantile(finite.to_numpy(dtype=np.float64), (0.2, 0.4, 0.6, 0.8))


def intensity_bucket(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values.to_numpy(dtype=np.float64), side="right").astype(np.int8)


def session_code(features: pd.DataFrame) -> np.ndarray:
    london = features["london_session"].to_numpy(dtype=np.int8, copy=False)
    new_york = features["new_york_session"].to_numpy(dtype=np.int8, copy=False)
    return london + 2 * new_york


def _read_columns(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("no panel partitions found")
    return pd.concat(pd.read_parquet(path, columns=columns) for path in paths).sort_index(
        kind="stable"
    )


def read_control_panel(panel: Path, pair: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = panel / f"{pair}-features.parquet"
    outcome_path = panel / f"{pair}-outcomes.parquet"
    if feature_path.exists() and outcome_path.exists():
        features = pd.read_parquet(feature_path, columns=list(CONTROL_FEATURES))
        outcomes = pd.read_parquet(outcome_path, columns=_outcome_columns(horizon))
    else:
        feature_paths = sorted(panel.glob(f"{pair}-????-features.parquet"))
        outcome_paths = sorted(panel.glob(f"{pair}-????-outcomes.parquet"))
        if len(feature_paths) != len(outcome_paths):
            raise FileNotFoundError(f"incomplete panel partitions for {pair} in {panel}")
        features = _read_columns(feature_paths, list(CONTROL_FEATURES))
        outcomes = _read_columns(outcome_paths, _outcome_columns(horizon))
    if not features.index.equals(outcomes.index):
        raise ValueError(f"feature/outcome indexes differ for {pair}")
    return features, outcomes


def conditional_predictions(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    pair: str,
    side: int,
    side_name: str,
    horizon: int,
) -> pd.DataFrame:
    """Fit B1 on 2019-2020 and return predictions only for 2021-H2."""
    base_name = f"h{horizon}_{side_name}_terminal_r_base"
    stress_name = f"h{horizon}_{side_name}_terminal_r_stress"
    finite_features = np.isfinite(features.to_numpy(dtype=np.float64, copy=False)).all(axis=1)
    valid_outcomes = outcomes[[base_name, stress_name]].notna().all(axis=1).to_numpy()
    training_mask = finite_features & valid_outcomes & (features.index < TRAIN_END)
    selection_mask = (
        finite_features
        & valid_outcomes
        & (features.index >= CALIBRATION_END)
        & (features.index < SELECTION_END)
    )
    training_features = features.loc[training_mask]
    selection_features = features.loc[selection_mask]
    edges = intensity_edges(training_features["update_intensity_64"])

    training_keys = pd.MultiIndex.from_arrays(
        [
            session_code(training_features),
            intensity_bucket(training_features["update_intensity_64"], edges),
        ],
        names=("session", "intensity_quintile"),
    )
    training_targets = outcomes.loc[training_mask, [base_name, stress_name]].copy()
    training_targets.index = training_keys
    conditional_means = training_targets.groupby(level=[0, 1]).mean()

    selection_keys = pd.MultiIndex.from_arrays(
        [
            session_code(selection_features),
            intensity_bucket(selection_features["update_intensity_64"], edges),
        ],
        names=("session", "intensity_quintile"),
    )
    expected = conditional_means.reindex(selection_keys)
    actual = outcomes.loc[selection_mask, [base_name, stress_name]]
    return pd.DataFrame(
        {
            "pair": pair,
            "side": side,
            "expected_r_base": expected[base_name].to_numpy(),
            "expected_r_stress": expected[stress_name].to_numpy(),
            "expected_r": np.minimum(
                expected[base_name].to_numpy(), expected[stress_name].to_numpy()
            ),
            "base_r": actual[base_name].to_numpy(),
            "stress_r": actual[stress_name].to_numpy(),
        },
        index=selection_features.index,
    )


def run_controls(panel: Path, output: Path, horizon: int) -> dict[str, object]:
    parts: list[pd.DataFrame] = []
    for pair in ALL_PAIRS:
        features, outcomes = read_control_panel(panel, pair, horizon)
        for side, side_name in ((1, "long"), (-1, "short")):
            parts.append(
                conditional_predictions(
                    features,
                    outcomes,
                    pair=pair,
                    side=side,
                    side_name=side_name,
                    horizon=horizon,
                )
            )
    predictions = pd.concat(parts).sort_index(kind="stable")
    threshold_grid = []
    for threshold in EV_THRESHOLDS:
        result = metrics(apply_policy(predictions, horizon, threshold))
        threshold_grid.append(
            {"expected_r": threshold, **asdict(result), "core_gate": core_selection_gate(result)}
        )
    report: dict[str, object] = {
        "kind": "preregistered_controls",
        "horizon_seconds": horizon,
        "b0": {"policy": "never trade", "trades": 0},
        "b1": {
            "policy": "training mean by pair, side, session and intensity quintile",
            "prediction_rows": len(predictions),
            "score_quantiles": {
                str(quantile): float(predictions["expected_r"].quantile(quantile))
                for quantile in (0.5, 0.9, 0.99, 0.999, 1.0)
            },
            "threshold_grid": threshold_grid,
            "has_core_candidate": any(item["core_gate"] for item in threshold_grid),
        },
        "year_2022_opened": False,
        "protected_samples_opened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"h{horizon}-controls-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizon", type=int, choices=HORIZONS_SECONDS, required=True)
    arguments = parser.parse_args(argv)
    report = run_controls(arguments.panel, arguments.output, arguments.horizon)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

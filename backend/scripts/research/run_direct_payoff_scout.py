"""Scout direct executable-payoff ranking instead of touch-probability ranking.

This development-only experiment asks a narrower question than approval:
if the model ranks by latency-adjusted terminal net payoff, do the selected
signals survive the same one-position trailing replay better than the binary
"touched target before stop" objective?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.top_p_model import (  # noqa: E402
    LOG1P_NONNEGATIVE_PREFIXES,
    TOP_P_TAILS,
    feature_columns,
)
from scripts.research.run_latency_adjusted_top_p_scout import (  # noqa: E402
    apply_latency_adjusted_labels,
    available_feature_sets,
    load_research_dataset,
    run_portfolio_replay,
)


@dataclass(frozen=True)
class PayoffTailMetrics:
    tail_fraction: float
    score_threshold_bps: float
    selected_count: int
    realized_mean_bps: float
    realized_median_bps: float
    positive_rate: float
    lift_over_base_bps: float


@dataclass(frozen=True)
class PayoffFoldResult:
    fold: int
    train_end_date: str
    calibration_date: str
    test_date: str
    test_count: int
    base_mean_bps: float
    base_median_bps: float
    base_positive_rate: float
    prediction_target_correlation: float
    tails: tuple[PayoffTailMetrics, ...]


@dataclass(frozen=True)
class PayoffWalkForwardResult:
    horizon_seconds: int
    feature_set: str
    target_column: str
    feature_columns: tuple[str, ...]
    folds: tuple[PayoffFoldResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "feature_set": self.feature_set,
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "folds": [
                {
                    **asdict(fold),
                    "tails": [asdict(tail) for tail in fold.tails],
                }
                for fold in self.folds
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "research-v1",
    )
    parser.add_argument(
        "--trade-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "normalized"
        / "aggTrades",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "direct-payoff-scout-stress",
    )
    parser.add_argument("--target", choices=("expected", "stress"), default="stress")
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--minimum-selected", type=int, default=50)
    parser.add_argument("--policy-set", choices=("v1", "low_cost"), default="v1")
    arguments = parser.parse_args()
    if arguments.latency_ms < 0:
        parser.error("--latency-ms cannot be negative")
    if arguments.minimum_selected < 1:
        parser.error("--minimum-selected must be positive")

    dataset_frame, opened_dates, label_config, label_config_source = load_research_dataset(
        arguments.dataset_root
    )
    latency_frame = apply_latency_adjusted_labels(
        dataset_frame,
        trade_root=arguments.trade_root,
        latency_ms=arguments.latency_ms,
        label_config=label_config,
    )
    target_column = f"latency_terminal_{arguments.target}_net_bps"
    cost_bps = (
        label_config.expected_cost_bps
        if arguments.target == "expected"
        else label_config.stress_cost_bps
    )
    latency_frame[target_column] = latency_frame["latency_terminal_bps"] - cost_bps

    scout_results: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for horizon_seconds in label_config.horizons_seconds:
        for feature_set in available_feature_sets(latency_frame):
            result, predictions = run_direct_payoff_walk_forward(
                latency_frame,
                horizon_seconds=horizon_seconds,
                feature_set=feature_set,
                target_column=target_column,
            )
            scout_results.append(
                {
                    "summary": summarize_direct_payoff_walk_forward(result),
                    "detail": result.to_dict(),
                }
            )
            predictions["candidate_result_index"] = len(scout_results) - 1
            prediction_frames.append(predictions)

    selection = select_direct_payoff_candidate(
        scout_results,
        minimum_selected=arguments.minimum_selected,
    )
    selected_predictions = next(
        frame
        for frame in prediction_frames
        if int(frame["candidate_result_index"].iloc[0]) == selection["result_index"]
    ).drop(columns=["candidate_result_index"])
    portfolio_results = run_portfolio_replay(
        selected_predictions,
        trade_root=arguments.trade_root,
        latency_ms=arguments.latency_ms,
        expected_round_trip_bps=label_config.expected_cost_bps,
        stress_round_trip_bps=label_config.stress_cost_bps,
        policies=_latency_policies(arguments.policy_set),
        output_root=arguments.output_root,
    )
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "development_direct_payoff_scout",
        "warning": (
            "Development scout on a short local dataset. This cannot approve Testnet, Demo, "
            "or LIVE trading."
        ),
        "opened_dates": opened_dates,
        "latency_ms": arguments.latency_ms,
        "policy_set": arguments.policy_set,
        "label_config_source": label_config_source,
        "target_column": target_column,
        "label_config": {
            "horizons_seconds": list(label_config.horizons_seconds),
            "expected_cost_bps": label_config.expected_cost_bps,
            "stress_cost_bps": label_config.stress_cost_bps,
            "initial_stop_bps": label_config.initial_stop_bps,
        },
        "hypothesis_attempt_count": len(scout_results),
        "candidate_selection": selection,
        "selected_candidate_summary": scout_results[selection["result_index"]]["summary"],
        "direct_payoff_results": scout_results,
        "portfolio_hypothesis_attempt_count": len(portfolio_results),
        "portfolio_results": portfolio_results,
    }
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    output = arguments.output_root / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_console_summary(report, output), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_direct_payoff_walk_forward(
    frame: pd.DataFrame,
    *,
    horizon_seconds: int,
    feature_set: str,
    target_column: str,
    minimum_train_days: int = 3,
    embargo_seconds: int = 300,
) -> tuple[PayoffWalkForwardResult, pd.DataFrame]:
    required = {"decision_time_ms", "horizon_seconds", "side", target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"payoff frame is missing columns: {sorted(missing)}")
    if minimum_train_days < 2:
        raise ValueError("minimum_train_days must be at least two")
    if embargo_seconds < horizon_seconds:
        raise ValueError("embargo must be at least the label horizon")
    horizon_frame = frame[frame["horizon_seconds"] == horizon_seconds].copy()
    horizon_frame["utc_date"] = pd.to_datetime(
        horizon_frame["decision_time_ms"], unit="ms", utc=True
    ).dt.date.astype(str)
    horizon_frame = horizon_frame.sort_values("decision_time_ms", kind="stable")
    columns = feature_columns(horizon_frame, feature_set)
    dates = tuple(horizon_frame["utc_date"].drop_duplicates())
    if len(dates) < minimum_train_days + 2:
        raise ValueError("not enough UTC days for train, calibration, and test")

    folds: list[PayoffFoldResult] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_position in range(minimum_train_days + 1, len(dates)):
        calibration_date = dates[test_position - 1]
        test_date = dates[test_position]
        calibration_start_ms = _date_start_ms(calibration_date)
        test_start_ms = _date_start_ms(test_date)
        train = horizon_frame[
            horizon_frame["decision_time_ms"] < calibration_start_ms - embargo_seconds * 1000
        ]
        calibration = horizon_frame[
            (horizon_frame["utc_date"] == calibration_date)
            & (horizon_frame["decision_time_ms"] < test_start_ms - embargo_seconds * 1000)
        ]
        test = horizon_frame[horizon_frame["utc_date"] == test_date]
        if train.empty or calibration.empty or test.empty:
            raise ValueError("training, calibration, and test splits must be non-empty")

        model = _new_payoff_model()
        model.fit(_matrix(train, columns), train[target_column].to_numpy(dtype=np.float64))
        calibration_scores = model.predict(_matrix(calibration, columns))
        test_scores = model.predict(_matrix(test, columns))
        test_targets = test[target_column].to_numpy(dtype=np.float64)
        tails = _payoff_tail_metrics(calibration_scores, test_scores, test_targets)
        predictions = test.loc[
            :,
            ["decision_time_ms", "side", "horizon_seconds", target_column],
        ].copy()
        predictions["probability"] = test_scores
        predictions["predicted_payoff_bps"] = test_scores
        predictions["fold"] = len(folds) + 1
        predictions["test_date"] = test_date
        for tail in tails:
            predictions[f"threshold_top_{tail.tail_fraction:.0%}"] = tail.score_threshold_bps
        prediction_frames.append(predictions)
        folds.append(
            PayoffFoldResult(
                fold=len(folds) + 1,
                train_end_date=dates[test_position - 2],
                calibration_date=calibration_date,
                test_date=test_date,
                test_count=len(test),
                base_mean_bps=float(np.mean(test_targets)),
                base_median_bps=float(np.median(test_targets)),
                base_positive_rate=float(np.mean(test_targets > 0)),
                prediction_target_correlation=_finite_correlation(test_scores, test_targets),
                tails=tails,
            )
        )
    return (
        PayoffWalkForwardResult(
            horizon_seconds=horizon_seconds,
            feature_set=feature_set,
            target_column=target_column,
            feature_columns=columns,
            folds=tuple(folds),
        ),
        pd.concat(prediction_frames, ignore_index=True),
    )


def summarize_direct_payoff_walk_forward(result: PayoffWalkForwardResult) -> dict[str, object]:
    folds = result.folds
    tail_summary: dict[str, dict[str, float]] = {}
    for tail_fraction in TOP_P_TAILS:
        tail_rows = [
            metric
            for fold in folds
            for metric in fold.tails
            if metric.tail_fraction == tail_fraction
        ]
        selected = sum(metric.selected_count for metric in tail_rows)
        realized_mean = (
            sum(metric.realized_mean_bps * metric.selected_count for metric in tail_rows)
            / selected
            if selected
            else 0.0
        )
        positive_rate = (
            sum(metric.positive_rate * metric.selected_count for metric in tail_rows) / selected
            if selected
            else 0.0
        )
        tail_summary[f"top_{tail_fraction:.0%}"] = {
            "selected_count": selected,
            "realized_mean_bps": realized_mean,
            "positive_rate": positive_rate,
            "mean_lift_over_base_bps": float(np.mean([row.lift_over_base_bps for row in tail_rows])),
        }
    return {
        "horizon_seconds": result.horizon_seconds,
        "feature_set": result.feature_set,
        "target_column": result.target_column,
        "fold_count": len(folds),
        "base_mean_bps": float(np.mean([fold.base_mean_bps for fold in folds])),
        "base_positive_rate": float(np.mean([fold.base_positive_rate for fold in folds])),
        "mean_prediction_target_correlation": float(
            np.mean([fold.prediction_target_correlation for fold in folds])
        ),
        "tails": tail_summary,
        "top_p_payoff_monotonic": _payoff_monotonic(tail_summary),
    }


def select_direct_payoff_candidate(
    results: list[dict[str, object]],
    *,
    minimum_selected: int,
) -> dict[str, object]:
    if not results:
        raise ValueError("results cannot be empty")
    ranked: list[tuple[tuple[float, float, float, float], dict[str, object]]] = []
    for index, result in enumerate(results):
        summary = _dict(result["summary"])
        top_1 = _dict(_dict(summary["tails"])["top_1%"])
        selected_count = int(top_1["selected_count"])
        realized_mean_bps = float(top_1["realized_mean_bps"])
        monotonic = bool(summary["top_p_payoff_monotonic"])
        enough_rows = selected_count >= minimum_selected
        positive = realized_mean_bps > 0
        tier = 3 if monotonic and enough_rows and positive else 2 if enough_rows and positive else 1 if enough_rows else 0
        key = (
            float(tier),
            realized_mean_bps,
            float(summary["mean_prediction_target_correlation"]),
            -float(index),
        )
        ranked.append(
            (
                key,
                {
                    "result_index": index,
                    "selection_tier": tier,
                    "selection_tier_reason": (
                        "positive_monotonic_minimum_selected"
                        if tier == 3
                        else "positive_minimum_selected"
                        if tier == 2
                        else "minimum_selected_only"
                        if tier == 1
                        else "fallback_below_minimum_selected"
                    ),
                    "minimum_selected": minimum_selected,
                    "selected_count_top_1pct": selected_count,
                    "realized_mean_bps_top_1pct": realized_mean_bps,
                    "top_p_payoff_monotonic": monotonic,
                    "horizon_seconds": int(summary["horizon_seconds"]),
                    "feature_set": str(summary["feature_set"]),
                    "mean_prediction_target_correlation": float(
                        summary["mean_prediction_target_correlation"]
                    ),
                },
            )
        )
    return max(ranked, key=lambda row: row[0])[1]


def _payoff_tail_metrics(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    test_targets: np.ndarray,
) -> tuple[PayoffTailMetrics, ...]:
    base_mean = float(np.mean(test_targets))
    rows: list[PayoffTailMetrics] = []
    for tail_fraction in TOP_P_TAILS:
        threshold = float(np.quantile(calibration_scores, 1 - tail_fraction))
        selected_mask = test_scores >= threshold
        selected_count = int(selected_mask.sum())
        selected_targets = test_targets[selected_mask]
        realized_mean = float(np.mean(selected_targets)) if selected_count else 0.0
        rows.append(
            PayoffTailMetrics(
                tail_fraction=tail_fraction,
                score_threshold_bps=threshold,
                selected_count=selected_count,
                realized_mean_bps=realized_mean,
                realized_median_bps=float(np.median(selected_targets)) if selected_count else 0.0,
                positive_rate=float(np.mean(selected_targets > 0)) if selected_count else 0.0,
                lift_over_base_bps=realized_mean - base_mean,
            )
        )
    return tuple(rows)


def _payoff_monotonic(tail_summary: dict[str, dict[str, float]]) -> bool:
    return (
        tail_summary["top_1%"]["realized_mean_bps"]
        >= tail_summary["top_5%"]["realized_mean_bps"]
        >= tail_summary["top_10%"]["realized_mean_bps"]
    )


def _new_payoff_model() -> Pipeline:
    return Pipeline(
        steps=(
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        )
    )


def _matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    matrix = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("model features must be finite")
    for index, column in enumerate(columns):
        if column.startswith(LOG1P_NONNEGATIVE_PREFIXES):
            matrix[:, index] = np.log1p(np.maximum(matrix[:, index], 0))
    return matrix


def _date_start_ms(utc_date: str) -> int:
    return int(datetime.fromisoformat(utc_date).replace(tzinfo=UTC).timestamp() * 1000)


def _finite_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    if len(predictions) < 2 or len(targets) < 2:
        return 0.0
    if float(np.std(predictions)) == 0.0 or float(np.std(targets)) == 0.0:
        return 0.0
    correlation = float(np.corrcoef(predictions, targets)[0, 1])
    return correlation if math.isfinite(correlation) else 0.0


def _console_summary(report: dict[str, Any], output: Path) -> dict[str, Any]:
    portfolio_rows = [
        {
            "policy": result["policy"],
            "tail_fraction": result["tail_fraction"],
            **_dict(result["summary"]),
        }
        for result in report["portfolio_results"]
    ]
    best_expected = max(
        portfolio_rows,
        key=lambda row: float(row["expected_mean_bps"]),
    )
    best_stress = max(
        portfolio_rows,
        key=lambda row: float(row["stress_mean_bps"]),
    )
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "opened_dates": report["opened_dates"],
        "target_column": report["target_column"],
        "latency_ms": report["latency_ms"],
        "policy_set": report["policy_set"],
        "candidate_selection": report["candidate_selection"],
        "best_expected_portfolio": best_expected,
        "best_stress_portfolio": best_stress,
        "report_path": str(output),
    }


def _dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected dict")
    return value


def _latency_policies(policy_set: str) -> tuple[Any, ...]:
    from scripts.research.run_latency_adjusted_top_p_scout import _policies

    return _policies(policy_set)


if __name__ == "__main__":
    raise SystemExit(main())

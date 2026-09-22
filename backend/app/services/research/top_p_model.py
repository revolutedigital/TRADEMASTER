"""Purged walk-forward calibrated probability models and top-p policies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TOP_P_TAILS = (0.01, 0.02, 0.05, 0.10)
BOOK_FEATURE_SETS = (
    "flow_book",
    "flow_price_book",
    "flow_price_book_session",
)


@dataclass(frozen=True)
class TopPTailMetrics:
    tail_fraction: float
    probability_threshold: float
    selected_count: int
    target_rate: float
    lift_over_base: float


@dataclass(frozen=True)
class WalkForwardFoldResult:
    fold: int
    train_end_date: str
    calibration_date: str
    test_date: str
    test_count: int
    base_rate: float
    brier_score: float
    log_loss: float
    roc_auc: float
    calibration_mean_error: float
    tails: tuple[TopPTailMetrics, ...]


@dataclass(frozen=True)
class WalkForwardResult:
    horizon_seconds: int
    feature_set: str
    feature_columns: tuple[str, ...]
    folds: tuple[WalkForwardFoldResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon_seconds": self.horizon_seconds,
            "feature_set": self.feature_set,
            "feature_columns": list(self.feature_columns),
            "folds": [
                {
                    **asdict(fold),
                    "tails": [asdict(tail) for tail in fold.tails],
                }
                for fold in self.folds
            ],
        }


def feature_columns(frame: pd.DataFrame, feature_set: str) -> tuple[str, ...]:
    flow = tuple(
        column
        for column in frame.columns
        if column.startswith(
            (
                "trade_count_",
                "quote_volume_",
                "flow_imbalance_",
                "directed_flow_imbalance_",
                "mean_interarrival_ms_",
            )
        )
    )
    price = tuple(
        column
        for column in frame.columns
        if column.startswith(("return_", "directed_return_", "realized_vol_"))
    )
    session = tuple(
        column for column in ("hour_sin", "hour_cos", "side_sign") if column in frame.columns
    )
    book = tuple(
        column
        for column in frame.columns
        if column.startswith(
            (
                "book_available",
                "book_update_age_ms",
                "spread_bps",
                "depth_imbalance",
                "directed_depth_imbalance",
                "microprice_displacement_bps",
                "directed_microprice_displacement_bps",
            )
        )
    )
    available = {
        "flow": flow,
        "flow_price": flow + price,
        "flow_price_session": flow + price + session,
        "flow_book": flow + book,
        "flow_price_book": flow + price + book,
        "flow_price_book_session": flow + price + book + session,
    }
    if feature_set not in available:
        raise ValueError(f"unknown feature set: {feature_set}")
    if feature_set in BOOK_FEATURE_SETS and not book:
        raise ValueError(f"feature set {feature_set} requires book feature columns")
    columns = tuple(dict.fromkeys(available[feature_set]))
    if not columns:
        raise ValueError(f"feature set {feature_set} has no available columns")
    return columns


def run_calibrated_walk_forward(
    frame: pd.DataFrame,
    *,
    horizon_seconds: int,
    feature_set: str,
    minimum_train_days: int = 3,
    embargo_seconds: int = 300,
) -> WalkForwardResult:
    result, _ = _run_calibrated_walk_forward(
        frame,
        horizon_seconds=horizon_seconds,
        feature_set=feature_set,
        minimum_train_days=minimum_train_days,
        embargo_seconds=embargo_seconds,
        collect_predictions=False,
    )
    return result


def run_calibrated_walk_forward_with_predictions(
    frame: pd.DataFrame,
    *,
    horizon_seconds: int,
    feature_set: str,
    minimum_train_days: int = 3,
    embargo_seconds: int = 300,
) -> tuple[WalkForwardResult, pd.DataFrame]:
    return _run_calibrated_walk_forward(
        frame,
        horizon_seconds=horizon_seconds,
        feature_set=feature_set,
        minimum_train_days=minimum_train_days,
        embargo_seconds=embargo_seconds,
        collect_predictions=True,
    )


def _run_calibrated_walk_forward(
    frame: pd.DataFrame,
    *,
    horizon_seconds: int,
    feature_set: str,
    minimum_train_days: int,
    embargo_seconds: int,
    collect_predictions: bool,
) -> tuple[WalkForwardResult, pd.DataFrame]:
    required = {"decision_time_ms", "horizon_seconds", "target"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"model frame is missing columns: {sorted(missing)}")
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
    if feature_set in BOOK_FEATURE_SETS:
        _require_complete_book_frame(horizon_frame)
    dates = tuple(horizon_frame["utc_date"].drop_duplicates())
    if len(dates) < minimum_train_days + 2:
        raise ValueError("not enough UTC days for train, calibration, and test")

    folds: list[WalkForwardFoldResult] = []
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
        _require_binary(train["target"], "training")
        _require_binary(calibration["target"], "calibration")
        _require_binary(test["target"], "test")

        base_model = Pipeline(
            steps=(
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=0.1,
                        l1_ratio=0,
                        solver="lbfgs",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            )
        )
        base_model.fit(_matrix(train, columns), train["target"].to_numpy())
        calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
        calibrated.fit(_matrix(calibration, columns), calibration["target"].to_numpy())
        calibration_probabilities = calibrated.predict_proba(_matrix(calibration, columns))[:, 1]
        test_probabilities = calibrated.predict_proba(_matrix(test, columns))[:, 1]
        test_targets = test["target"].to_numpy(dtype=np.int8)
        base_rate = float(test_targets.mean())
        tails = _top_p_metrics(
            calibration_probabilities, test_probabilities, test_targets, base_rate
        )
        if collect_predictions:
            predictions = test.loc[
                :,
                [
                    column
                    for column in (
                        "decision_time_ms",
                        "side",
                        "horizon_seconds",
                        "target",
                    )
                    if column in test.columns
                ],
            ].copy()
            predictions["probability"] = test_probabilities
            predictions["fold"] = len(folds) + 1
            predictions["test_date"] = test_date
            for tail in tails:
                predictions[f"threshold_top_{tail.tail_fraction:.0%}"] = tail.probability_threshold
            prediction_frames.append(predictions)
        folds.append(
            WalkForwardFoldResult(
                fold=len(folds) + 1,
                train_end_date=dates[test_position - 2],
                calibration_date=calibration_date,
                test_date=test_date,
                test_count=len(test),
                base_rate=base_rate,
                brier_score=float(brier_score_loss(test_targets, test_probabilities)),
                log_loss=float(log_loss(test_targets, test_probabilities)),
                roc_auc=float(roc_auc_score(test_targets, test_probabilities)),
                calibration_mean_error=float(test_probabilities.mean() - base_rate),
                tails=tails,
            )
        )
    result = WalkForwardResult(
        horizon_seconds=horizon_seconds,
        feature_set=feature_set,
        feature_columns=columns,
        folds=tuple(folds),
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    return result, predictions


def summarize_walk_forward(result: WalkForwardResult) -> dict[str, object]:
    folds = result.folds
    tail_summary: dict[str, dict[str, float]] = {}
    for tail in TOP_P_TAILS:
        tail_rows = [
            metric for fold in folds for metric in fold.tails if metric.tail_fraction == tail
        ]
        selected = sum(metric.selected_count for metric in tail_rows)
        weighted_rate = (
            sum(metric.target_rate * metric.selected_count for metric in tail_rows) / selected
            if selected
            else 0.0
        )
        tail_summary[f"top_{tail:.0%}"] = {
            "selected_count": selected,
            "target_rate": weighted_rate,
            "mean_lift": float(np.mean([row.lift_over_base for row in tail_rows])),
        }
    return {
        "horizon_seconds": result.horizon_seconds,
        "feature_set": result.feature_set,
        "fold_count": len(folds),
        "mean_brier_score": float(np.mean([fold.brier_score for fold in folds])),
        "mean_roc_auc": float(np.mean([fold.roc_auc for fold in folds])),
        "mean_calibration_error": float(np.mean([fold.calibration_mean_error for fold in folds])),
        "tails": tail_summary,
    }


def _top_p_metrics(
    calibration_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    test_targets: np.ndarray,
    base_rate: float,
) -> tuple[TopPTailMetrics, ...]:
    rows: list[TopPTailMetrics] = []
    for tail in TOP_P_TAILS:
        threshold = float(np.quantile(calibration_probabilities, 1 - tail))
        selected_mask = test_probabilities >= threshold
        selected_count = int(selected_mask.sum())
        target_rate = float(test_targets[selected_mask].mean()) if selected_count else 0.0
        lift = target_rate / base_rate if base_rate > 0 else 0.0
        rows.append(
            TopPTailMetrics(
                tail_fraction=tail,
                probability_threshold=threshold,
                selected_count=selected_count,
                target_rate=target_rate,
                lift_over_base=lift,
            )
        )
    return tuple(rows)


def _matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    matrix = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("model features must be finite")
    for index, column in enumerate(columns):
        if column.startswith(("trade_count_", "quote_volume_", "mean_interarrival_ms_")):
            matrix[:, index] = np.log1p(np.maximum(matrix[:, index], 0))
    return matrix


def _require_complete_book_frame(frame: pd.DataFrame) -> None:
    required = {"book_available", "book_update_age_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"book model frame is missing columns: {sorted(missing)}")
    unavailable_count = int((frame["book_available"].to_numpy(dtype=np.float64) < 1).sum())
    if unavailable_count:
        raise ValueError(
            f"book feature set requires complete book features; "
            f"{unavailable_count} rows are unavailable"
        )
    if (frame["book_update_age_ms"].to_numpy(dtype=np.float64) < 0).any():
        raise ValueError("book feature set contains quotes from the future")


def _require_binary(target: pd.Series, split: str) -> None:
    if len(target) == 0 or target.nunique() < 2:
        raise ValueError(f"{split} split must contain both target classes")


def _date_start_ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)

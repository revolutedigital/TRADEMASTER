"""Purged walk-forward calibrated probability models and top-p policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

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
    "flow_book_aux",
    "flow_price_book_aux",
    "flow_price_book_aux_session",
)
AUXILIARY_FEATURE_SETS = (
    "flow_aux",
    "flow_price_aux",
    "flow_price_aux_session",
    "flow_book_aux",
    "flow_price_book_aux",
    "flow_price_book_aux_session",
)
SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class TopPTailMetrics:
    tail_fraction: float
    probability_threshold: float
    selected_count: int
    target_rate: float
    lift_over_base: float


@dataclass(frozen=True)
class TopPMonotonicityComparison:
    narrower_tail_fraction: float
    wider_tail_fraction: float
    narrower_selected_count: int
    wider_selected_count: int
    narrower_target_rate: float
    wider_target_rate: float
    one_sided_z_margin: float
    passed: bool
    reason: str | None


@dataclass(frozen=True)
class TopPMonotonicityReport:
    passed: bool
    comparisons: tuple[TopPMonotonicityComparison, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "comparisons": [asdict(comparison) for comparison in self.comparisons],
            "reasons": list(self.reasons),
        }


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


@dataclass(frozen=True)
class FrozenTopPPolicy:
    """Serializable research-only policy artifact for prospective shadow decisions."""

    payload: dict[str, object]
    model_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            **self.payload,
            "model_sha256": self.model_sha256,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()) + "\n"


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
                "book_event_count_",
                "book_bid_replenishment_qty_",
                "book_ask_replenishment_qty_",
                "book_bid_liquidity_removed_qty_",
                "book_ask_liquidity_removed_qty_",
                "spread_bps",
                "depth_imbalance",
                "directed_depth_imbalance",
                "microprice_displacement_bps",
                "directed_microprice_displacement_bps",
                "book_pressure_imbalance_",
                "directed_book_pressure_imbalance_",
                "book_spread_widening_bps_",
                "book_spread_recovery_bps_",
                "book_depth_imbalance_change_",
                "directed_book_depth_imbalance_change_",
                "book_microprice_displacement_change_bps_",
                "directed_book_microprice_displacement_change_bps_",
            )
        )
    )
    auxiliary = tuple(
        column
        for column in frame.columns
        if column.startswith(
            (
                "mark_available",
                "mark_update_age_ms",
                "mark_index_basis_bps",
                "directed_mark_index_basis_bps",
                "funding_rate",
                "directed_funding_rate",
                "liquidation_count_",
                "liquidation_net_qty_",
                "directed_liquidation_net_qty_",
                "liquidation_abs_qty_",
                "liquidation_net_notional_",
                "directed_liquidation_net_notional_",
                "liquidation_abs_notional_",
            )
        )
    )
    available = {
        "flow": flow,
        "flow_price": flow + price,
        "flow_price_session": flow + price + session,
        "flow_aux": flow + auxiliary,
        "flow_price_aux": flow + price + auxiliary,
        "flow_price_aux_session": flow + price + auxiliary + session,
        "flow_book": flow + book,
        "flow_price_book": flow + price + book,
        "flow_price_book_session": flow + price + book + session,
        "flow_book_aux": flow + book + auxiliary,
        "flow_price_book_aux": flow + price + book + auxiliary,
        "flow_price_book_aux_session": flow + price + book + auxiliary + session,
    }
    if feature_set not in available:
        raise ValueError(f"unknown feature set: {feature_set}")
    if feature_set in BOOK_FEATURE_SETS and not book:
        raise ValueError(f"feature set {feature_set} requires book feature columns")
    if feature_set in AUXILIARY_FEATURE_SETS and not auxiliary:
        raise ValueError(f"feature set {feature_set} requires auxiliary feature columns")
    columns = tuple(dict.fromkeys(available[feature_set]))
    if not columns:
        raise ValueError(f"feature set {feature_set} has no available columns")
    return columns


def freeze_top_p_policy(
    frame: pd.DataFrame,
    *,
    horizon_seconds: int,
    feature_set: str,
    tail_fraction: float,
    calibration_date: str,
    dataset_manifest_sha256: str,
    target_column: str = "target",
    embargo_seconds: int = 300,
) -> FrozenTopPPolicy:
    """Fit and serialize the frozen probability + top-p rule for shadow use only."""
    if tail_fraction <= 0 or tail_fraction >= 1:
        raise ValueError("tail_fraction must be in (0, 1)")
    if len(dataset_manifest_sha256) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in dataset_manifest_sha256
    ):
        raise ValueError("dataset_manifest_sha256 must be a lowercase SHA-256")
    if target_column not in frame.columns:
        raise ValueError(f"target column is missing: {target_column}")
    required = {"decision_time_ms", "horizon_seconds", target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"model frame is missing columns: {sorted(missing)}")
    if embargo_seconds < horizon_seconds:
        raise ValueError("embargo must be at least the label horizon")

    horizon_frame = frame[frame["horizon_seconds"] == horizon_seconds].copy()
    if horizon_frame.empty:
        raise ValueError(f"no rows for horizon {horizon_seconds}")
    horizon_frame["target"] = horizon_frame[target_column].astype("int8")
    horizon_frame["utc_date"] = pd.to_datetime(
        horizon_frame["decision_time_ms"], unit="ms", utc=True
    ).dt.date.astype(str)
    horizon_frame = horizon_frame.sort_values("decision_time_ms", kind="stable")
    columns = feature_columns(horizon_frame, feature_set)
    if feature_set in BOOK_FEATURE_SETS:
        _require_complete_book_frame(horizon_frame)

    calibration_start_ms = _date_start_ms(calibration_date)
    train = horizon_frame[
        horizon_frame["decision_time_ms"] < calibration_start_ms - embargo_seconds * 1000
    ]
    calibration = horizon_frame[horizon_frame["utc_date"] == calibration_date]
    if train.empty or calibration.empty:
        raise ValueError("training and calibration splits must be non-empty")
    _require_binary(train["target"], "training")
    _require_binary(calibration["target"], "calibration")

    base_model = _new_base_model()
    base_model.fit(_matrix(train, columns), train["target"].to_numpy())
    calibration_scores = _decision_scores(base_model, calibration, columns)
    calibrator = LogisticRegression(
        C=1_000_000,
        solver="lbfgs",
        max_iter=500,
        random_state=42,
    )
    calibrator.fit(calibration_scores.reshape(-1, 1), calibration["target"].to_numpy())
    calibration_probabilities = calibrator.predict_proba(calibration_scores.reshape(-1, 1))[:, 1]
    probability_threshold = float(np.quantile(calibration_probabilities, 1 - tail_fraction))
    selected_mask = calibration_probabilities >= probability_threshold
    selected_count = int(selected_mask.sum())

    scaler = base_model.named_steps["scale"]
    logistic = base_model.named_steps["logistic"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "research_top_p_shadow_policy",
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "model_family": "standardized_logistic_with_platt_sigmoid",
        "horizon_seconds": horizon_seconds,
        "feature_set": feature_set,
        "feature_columns": list(columns),
        "target_column": target_column,
        "tail_fraction": tail_fraction,
        "probability_threshold": probability_threshold,
        "embargo_seconds": embargo_seconds,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training": {
            "row_count": int(len(train)),
            "start_decision_time_ms": int(train["decision_time_ms"].min()),
            "end_decision_time_ms": int(train["decision_time_ms"].max()),
            "target_rate": float(train["target"].mean()),
        },
        "calibration": {
            "utc_date": calibration_date,
            "row_count": int(len(calibration)),
            "target_rate": float(calibration["target"].mean()),
            "selected_count": selected_count,
            "selected_target_rate": (
                float(calibration.loc[selected_mask, "target"].mean()) if selected_count else 0.0
            ),
        },
        "transform": {
            "log1p_nonnegative_prefixes": [
                "trade_count_",
                "quote_volume_",
                "mean_interarrival_ms_",
                "book_event_count_",
                "book_bid_replenishment_qty_",
                "book_ask_replenishment_qty_",
                "book_bid_liquidity_removed_qty_",
                "book_ask_liquidity_removed_qty_",
                "book_spread_widening_bps_",
                "book_spread_recovery_bps_",
                "mark_update_age_ms",
                "liquidation_count_",
                "liquidation_abs_qty_",
                "liquidation_abs_notional_",
            ],
            "scaler_mean": _float_list(scaler.mean_),
            "scaler_scale": _float_list(scaler.scale_),
        },
        "base_model": {
            "intercept": float(logistic.intercept_[0]),
            "coefficients": _float_list(logistic.coef_[0]),
            "regularization": {
                "class": "LogisticRegression",
                "C": 0.1,
                "solver": "lbfgs",
                "max_iter": 500,
                "random_state": 42,
            },
        },
        "calibrator": {
            "method": "logistic_sigmoid_on_base_logit",
            "intercept": float(calibrator.intercept_[0]),
            "coefficient": float(calibrator.coef_[0][0]),
        },
    }
    return FrozenTopPPolicy(payload=payload, model_sha256=_stable_sha256(payload))


def predict_frozen_top_p_probability(
    artifact: dict[str, Any],
    feature_vector: dict[str, float],
) -> float:
    """Score one feature vector from a frozen JSON policy artifact."""
    columns = tuple(str(column) for column in artifact["feature_columns"])
    raw_values = np.array([float(feature_vector[column]) for column in columns], dtype=np.float64)
    if not np.isfinite(raw_values).all():
        raise ValueError("feature vector must contain finite values")
    transformed = raw_values.copy()
    for index, column in enumerate(columns):
        if column.startswith(("trade_count_", "quote_volume_", "mean_interarrival_ms_")):
            transformed[index] = math.log1p(max(transformed[index], 0.0))
    transform = artifact["transform"]
    scaler_mean = np.asarray(transform["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(transform["scaler_scale"], dtype=np.float64)
    if len(scaler_mean) != len(columns) or len(scaler_scale) != len(columns):
        raise ValueError("artifact scaler shape does not match feature columns")
    scaled = (transformed - scaler_mean) / scaler_scale
    base_model = artifact["base_model"]
    coefficients = np.asarray(base_model["coefficients"], dtype=np.float64)
    if len(coefficients) != len(columns):
        raise ValueError("artifact coefficient shape does not match feature columns")
    base_score = float(base_model["intercept"]) + float(np.dot(coefficients, scaled))
    calibrator = artifact["calibrator"]
    calibrated_score = float(calibrator["intercept"]) + float(calibrator["coefficient"]) * base_score
    return _sigmoid(calibrated_score)


def frozen_top_p_would_enter(artifact: dict[str, Any], feature_vector: dict[str, float]) -> bool:
    probability = predict_frozen_top_p_probability(artifact, feature_vector)
    return probability >= float(artifact["probability_threshold"])


def verify_frozen_top_p_policy(artifact: dict[str, Any]) -> str:
    """Return the model hash when a frozen top-p artifact is intact and research-only."""
    model_sha256 = str(artifact.get("model_sha256", ""))
    payload = {key: value for key, value in artifact.items() if key != "model_sha256"}
    if model_sha256 != _stable_sha256(payload):
        raise ValueError("frozen top-p policy artifact hash does not match its payload")
    if payload.get("artifact_kind") != "research_top_p_shadow_policy":
        raise ValueError("unexpected frozen top-p policy artifact kind")
    if payload.get("research_only") is not True:
        raise ValueError("frozen top-p policy must be research-only")
    if payload.get("order_submission_allowed") is not False:
        raise ValueError("frozen top-p policy cannot allow order submission")
    if payload.get("execution_authorization") != "none":
        raise ValueError("frozen top-p policy cannot carry execution authorization")
    return model_sha256


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

        base_model = _new_base_model()
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
    monotonicity = evaluate_top_p_monotonicity(result)
    return {
        "horizon_seconds": result.horizon_seconds,
        "feature_set": result.feature_set,
        "fold_count": len(folds),
        "mean_brier_score": float(np.mean([fold.brier_score for fold in folds])),
        "mean_roc_auc": float(np.mean([fold.roc_auc for fold in folds])),
        "mean_calibration_error": float(np.mean([fold.calibration_mean_error for fold in folds])),
        "tails": tail_summary,
        "top_p_monotonic": monotonicity.passed,
        "top_p_monotonicity": monotonicity.to_dict(),
    }


def evaluate_top_p_monotonicity(
    result: WalkForwardResult,
    *,
    confidence_z: float = 1.6448536269514722,
) -> TopPMonotonicityReport:
    """Check the preregistered tail ordering without pretending tiny samples are certain."""
    if confidence_z < 0:
        raise ValueError("confidence_z cannot be negative")
    aggregate = _aggregate_tail_metrics(result)
    comparisons = tuple(
        _compare_top_p_tails(
            aggregate,
            narrower_tail_fraction=narrower,
            wider_tail_fraction=wider,
            confidence_z=confidence_z,
        )
        for narrower, wider in ((0.01, 0.05), (0.05, 0.10))
    )
    reasons = tuple(comparison.reason for comparison in comparisons if comparison.reason)
    return TopPMonotonicityReport(
        passed=not reasons,
        comparisons=comparisons,
        reasons=reasons,
    )


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


def _aggregate_tail_metrics(
    result: WalkForwardResult,
) -> dict[float, tuple[int, float, float]]:
    aggregate: dict[float, tuple[int, float, float]] = {}
    for tail in TOP_P_TAILS:
        metrics = [
            metric
            for fold in result.folds
            for metric in fold.tails
            if metric.tail_fraction == tail
        ]
        selected_count = sum(metric.selected_count for metric in metrics)
        success_estimate = sum(metric.target_rate * metric.selected_count for metric in metrics)
        target_rate = success_estimate / selected_count if selected_count else 0.0
        aggregate[tail] = (selected_count, success_estimate, target_rate)
    return aggregate


def _compare_top_p_tails(
    aggregate: dict[float, tuple[int, float, float]],
    *,
    narrower_tail_fraction: float,
    wider_tail_fraction: float,
    confidence_z: float,
) -> TopPMonotonicityComparison:
    narrower_count, _, narrower_rate = aggregate[narrower_tail_fraction]
    wider_count, _, wider_rate = aggregate[wider_tail_fraction]
    reason = None
    if narrower_count == 0:
        reason = f"top_{narrower_tail_fraction:.0%}_selected_no_rows"
    elif wider_count == 0:
        reason = f"top_{wider_tail_fraction:.0%}_selected_no_rows"
    standard_error = (
        math.sqrt(
            narrower_rate * (1 - narrower_rate) / max(narrower_count, 1)
            + wider_rate * (1 - wider_rate) / max(wider_count, 1)
        )
        if reason is None
        else 0.0
    )
    one_sided_margin = confidence_z * standard_error
    passed = reason is None and narrower_rate + one_sided_margin >= wider_rate
    if reason is None and not passed:
        reason = (
            f"top_{narrower_tail_fraction:.0%}_underperforms_top_"
            f"{wider_tail_fraction:.0%}"
        )
    return TopPMonotonicityComparison(
        narrower_tail_fraction=narrower_tail_fraction,
        wider_tail_fraction=wider_tail_fraction,
        narrower_selected_count=narrower_count,
        wider_selected_count=wider_count,
        narrower_target_rate=narrower_rate,
        wider_target_rate=wider_rate,
        one_sided_z_margin=one_sided_margin,
        passed=passed,
        reason=reason,
    )


def _matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    matrix = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("model features must be finite")
    for index, column in enumerate(columns):
        if column.startswith(
            (
                "trade_count_",
                "quote_volume_",
                "mean_interarrival_ms_",
                "book_event_count_",
                "book_bid_replenishment_qty_",
                "book_ask_replenishment_qty_",
                "book_bid_liquidity_removed_qty_",
                "book_ask_liquidity_removed_qty_",
                "book_spread_widening_bps_",
                "book_spread_recovery_bps_",
                "mark_update_age_ms",
                "liquidation_count_",
                "liquidation_abs_qty_",
                "liquidation_abs_notional_",
            )
        ):
            matrix[:, index] = np.log1p(np.maximum(matrix[:, index], 0))
    return matrix


def _new_base_model() -> Pipeline:
    return Pipeline(
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


def _decision_scores(model: Pipeline, frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.asarray(model.decision_function(_matrix(frame, columns)), dtype=np.float64)


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


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)

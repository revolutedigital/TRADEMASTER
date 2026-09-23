"""Frozen latency-adjusted top-p holdout replay.

This script takes a candidate selected during development, freezes the
probability model and top-p threshold on a training/calibration dataset, then
applies that exact artifact to a separate holdout dataset. It is research-only:
it writes evidence, never sends orders, and never approves a strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.trailing_portfolio import (  # noqa: E402
    HistoricalTrailingSimulator,
    replay_one_position_portfolio,
    summarize_portfolio,
)
from app.services.research.research_dataset import load_trade_interval  # noqa: E402
from app.services.research.top_p_model import (  # noqa: E402
    LOG1P_NONNEGATIVE_PREFIXES,
    freeze_top_p_policy,
    verify_frozen_top_p_policy,
)
from scripts.research.run_latency_adjusted_top_p_scout import (  # noqa: E402
    _policies,
    apply_latency_adjusted_labels,
    load_research_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset-root", type=Path, required=True)
    parser.add_argument("--holdout-dataset-root", type=Path, required=True)
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
        / "latency-adjusted-frozen-top-p-holdout",
    )
    parser.add_argument("--target", choices=("expected", "stress"), default="stress")
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--horizon-seconds", type=int, required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--tail-fraction", type=float, required=True)
    parser.add_argument("--calibration-date", required=True)
    parser.add_argument("--policy-set", choices=("v1", "low_cost"), default="low_cost")
    parser.add_argument("--policy-name", required=True)
    arguments = parser.parse_args()
    if arguments.latency_ms < 0:
        parser.error("--latency-ms cannot be negative")
    if arguments.horizon_seconds <= 0:
        parser.error("--horizon-seconds must be positive")
    if not 0 < arguments.tail_fraction < 1:
        parser.error("--tail-fraction must be between 0 and 1")

    report = run_frozen_holdout(
        training_dataset_root=arguments.training_dataset_root,
        holdout_dataset_root=arguments.holdout_dataset_root,
        trade_root=arguments.trade_root,
        output_root=arguments.output_root,
        target=arguments.target,
        latency_ms=arguments.latency_ms,
        horizon_seconds=arguments.horizon_seconds,
        feature_set=arguments.feature_set,
        tail_fraction=arguments.tail_fraction,
        calibration_date=arguments.calibration_date,
        policy_set=arguments.policy_set,
        policy_name=arguments.policy_name,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_frozen_holdout(
    *,
    training_dataset_root: Path,
    holdout_dataset_root: Path,
    trade_root: Path,
    output_root: Path,
    target: str,
    latency_ms: int,
    horizon_seconds: int,
    feature_set: str,
    tail_fraction: float,
    calibration_date: str,
    policy_set: str,
    policy_name: str,
) -> dict[str, object]:
    train_frame, train_dates, train_config, train_config_source = load_research_dataset(
        training_dataset_root
    )
    holdout_frame, holdout_dates, holdout_config, holdout_config_source = load_research_dataset(
        holdout_dataset_root
    )
    if train_config != holdout_config:
        raise ValueError("training and holdout label configs must match")
    target_column = f"latency_paid_{target}_before_stop"
    train_latency = apply_latency_adjusted_labels(
        train_frame,
        trade_root=trade_root,
        latency_ms=latency_ms,
        label_config=train_config,
    )
    holdout_latency = apply_latency_adjusted_labels(
        holdout_frame,
        trade_root=trade_root,
        latency_ms=latency_ms,
        label_config=train_config,
    )
    artifact = freeze_top_p_policy(
        train_latency,
        horizon_seconds=horizon_seconds,
        feature_set=feature_set,
        tail_fraction=tail_fraction,
        calibration_date=calibration_date,
        dataset_manifest_sha256=_manifest_bundle_sha256(training_dataset_root),
        target_column=target_column,
        embargo_seconds=max(300, horizon_seconds),
    ).to_dict()
    model_sha256 = verify_frozen_top_p_policy(artifact)
    horizon_holdout = holdout_latency[
        holdout_latency["horizon_seconds"] == horizon_seconds
    ].copy()
    if horizon_holdout.empty:
        raise ValueError(f"holdout has no rows for horizon {horizon_seconds}")
    probabilities = _predict_artifact_probabilities(artifact, horizon_holdout)
    predictions = horizon_holdout.loc[
        :,
        ["decision_time_ms", "side", "horizon_seconds", target_column],
    ].copy()
    predictions["target"] = predictions[target_column].astype("int8")
    predictions["probability"] = probabilities
    threshold_column = f"threshold_top_{tail_fraction:.0%}"
    predictions[threshold_column] = float(artifact["probability_threshold"])
    selected = predictions[predictions["probability"] >= predictions[threshold_column]]
    policy = _policy_by_name(policy_set, policy_name)
    trades = _replay_frozen_policy(
        predictions,
        trade_root=trade_root,
        latency_ms=latency_ms,
        expected_round_trip_bps=train_config.expected_cost_bps,
        stress_round_trip_bps=train_config.stress_cost_bps,
        tail_fraction=tail_fraction,
        policy=policy,
    )
    summary = summarize_portfolio(trades)
    status, status_reasons = _status(summary)
    report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "frozen_holdout_research_only",
            "single_holdout_block_is_not_testnet_authorization",
            "top_p_model_and_trailing_replay_not_exchange_execution",
        ],
        "experiment_stage": "latency_adjusted_frozen_top_p_holdout",
        "status": status,
        "status_reasons": status_reasons,
        "target_column": target_column,
        "latency_ms": latency_ms,
        "policy_set": policy_set,
        "policy_name": policy_name,
        "tail_fraction": tail_fraction,
        "horizon_seconds": horizon_seconds,
        "feature_set": feature_set,
        "training_dates": train_dates,
        "holdout_dates": holdout_dates,
        "label_config": {
            "horizons_seconds": list(train_config.horizons_seconds),
            "expected_cost_bps": train_config.expected_cost_bps,
            "stress_cost_bps": train_config.stress_cost_bps,
            "initial_stop_bps": train_config.initial_stop_bps,
        },
        "config_sources": {
            "training": train_config_source,
            "holdout": holdout_config_source,
        },
        "frozen_artifact": artifact,
        "model_sha256": model_sha256,
        "holdout_selection": {
            "row_count": int(len(predictions)),
            "base_target_rate": float(predictions["target"].mean()),
            "selected_count": int(len(selected)),
            "selected_target_rate": float(selected["target"].mean()) if len(selected) else 0.0,
            "probability_threshold": float(artifact["probability_threshold"]),
        },
        "portfolio_summary": summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    trades_path = output_root / "frozen_holdout_trades.parquet"
    trades.to_parquet(trades_path, index=False)
    report["trades_path"] = str(trades_path)
    report_path = output_root / "report.json"
    payload = {
        **report,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _replay_frozen_policy(
    predictions: pd.DataFrame,
    *,
    trade_root: Path,
    latency_ms: int,
    expected_round_trip_bps: float,
    stress_round_trip_bps: float,
    tail_fraction: float,
    policy: Any,
) -> pd.DataFrame:
    start = datetime.fromtimestamp(predictions["decision_time_ms"].min() / 1000, tz=UTC) - timedelta(
        seconds=1
    )
    end = datetime.fromtimestamp(
        (predictions["decision_time_ms"] + predictions["horizon_seconds"] * 1000).max() / 1000,
        tz=UTC,
    ) + timedelta(seconds=1)
    trades = load_trade_interval(trade_root, start, end, expected_product="usdm_perpetual")
    simulator = HistoricalTrailingSimulator(
        trades["event_time_ms"].to_numpy(),
        trades["price"].to_numpy(),
        latency_ms=latency_ms,
        expected_round_trip_bps=expected_round_trip_bps,
        stress_round_trip_bps=stress_round_trip_bps,
    )
    return replay_one_position_portfolio(
        predictions,
        simulator,
        tail_fraction=tail_fraction,
        policy=policy,
    )


def _predict_artifact_probabilities(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    columns = tuple(str(column) for column in artifact["feature_columns"])
    matrix = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("holdout feature matrix must be finite")
    log1p_prefixes = tuple(artifact["transform"].get("log1p_nonnegative_prefixes", ()))
    for index, column in enumerate(columns):
        if column.startswith(log1p_prefixes or LOG1P_NONNEGATIVE_PREFIXES):
            matrix[:, index] = np.log1p(np.maximum(matrix[:, index], 0))
    scaler_mean = np.asarray(artifact["transform"]["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(artifact["transform"]["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(artifact["base_model"]["coefficients"], dtype=np.float64)
    scaled = (matrix - scaler_mean) / scaler_scale
    base_scores = float(artifact["base_model"]["intercept"]) + scaled @ coefficients
    calibrated_scores = (
        float(artifact["calibrator"]["intercept"])
        + float(artifact["calibrator"]["coefficient"]) * base_scores
    )
    return 1 / (1 + np.exp(-calibrated_scores))


def _policy_by_name(policy_set: str, policy_name: str) -> Any:
    for policy in _policies(policy_set):
        if policy.name == policy_name:
            return policy
    raise ValueError(f"policy {policy_name!r} not found in policy set {policy_set!r}")


def _manifest_bundle_sha256(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(dataset_root.glob("date=*/research_rows.manifest.json"))
    if not paths:
        raise ValueError(f"no manifests found in {dataset_root}")
    for path in paths:
        digest.update(path.relative_to(dataset_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _status(summary: dict[str, float | int]) -> tuple[str, list[str]]:
    expected_mean = float(summary["expected_mean_bps"])
    stress_mean = float(summary["stress_mean_bps"])
    if stress_mean > 0:
        return (
            "FROZEN_HOLDOUT_STRESS_POSITIVE_RESEARCH_ONLY",
            [
                "stress_cost_positive_but_not_strategy_approval",
                "needs_additional_untouched_blocks_and_execution_cost_validation",
            ],
        )
    if expected_mean > 0:
        return (
            "FROZEN_HOLDOUT_EXPECTED_POSITIVE_STRESS_NEGATIVE",
            [
                "expected_cost_positive",
                "stress_cost_negative",
                "do_not_promote_to_strategy",
            ],
        )
    return (
        "FROZEN_HOLDOUT_FAILED_REPLICATION",
        ["expected_and_stress_cost_negative", "do_not_promote_to_strategy"],
    )


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "holdout_selection": report["holdout_selection"],
        "portfolio_summary": report["portfolio_summary"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

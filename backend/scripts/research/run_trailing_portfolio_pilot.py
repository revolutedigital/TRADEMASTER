"""Replay calibrated OOS signals through bounded trailing policies."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.trailing_portfolio import (
    V1_TRAILING_POLICIES,
    HistoricalTrailingSimulator,
    replay_one_position_portfolio,
    summarize_portfolio,
)
from app.services.research.research_dataset import load_trade_interval
from app.services.research.top_p_model import (
    TOP_P_TAILS,
    run_calibrated_walk_forward_with_predictions,
)


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
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "normalized" / "aggTrades",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "trailing-pilot-stress-target",
    )
    arguments = parser.parse_args()
    paths = sorted(arguments.dataset_root.glob("date=*/research_rows.parquet"))
    if not paths:
        parser.error("no research dataset partitions found")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    frame["target"] = frame["paid_stress_before_stop"].astype("int8")
    model_feature_set = (
        "flow_price_book_session" if _has_book_features(frame) else "flow_price_session"
    )
    prediction_frames = []
    model_folds = []
    for horizon in (120, 300):
        result, predictions = run_calibrated_walk_forward_with_predictions(
            frame,
            horizon_seconds=horizon,
            feature_set=model_feature_set,
        )
        prediction_frames.append(predictions)
        model_folds.extend(fold.test_date for fold in result.folds)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    start = datetime.fromtimestamp(
        predictions["decision_time_ms"].min() / 1000, tz=UTC
    ) - timedelta(seconds=1)
    end = datetime.fromtimestamp(predictions["decision_time_ms"].max() / 1000, tz=UTC) + timedelta(
        seconds=301
    )
    trades = load_trade_interval(arguments.trade_root, start, end)
    simulator = HistoricalTrailingSimulator(
        trades["event_time_ms"].to_numpy(),
        trades["price"].to_numpy(),
    )
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for policy in V1_TRAILING_POLICIES:
        for tail in TOP_P_TAILS:
            managed = replay_one_position_portfolio(
                predictions,
                simulator,
                tail_fraction=tail,
                policy=policy,
            )
            filename = f"policy={policy.name}-top={tail:.0%}.parquet"
            managed.to_parquet(arguments.output_root / filename, index=False)
            results.append(
                {
                    "policy": policy.name,
                    "tail_fraction": tail,
                    "policy_parameters": policy.__dict__,
                    "summary": summarize_portfolio(managed),
                    "trades_path": filename,
                }
            )
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "experiment_stage": "development_pilot",
        "model_target": "paid_stress_before_stop",
        "model_feature_set": model_feature_set,
        "warning": (
            "Feature set and management policy are being compared on development folds. "
            "These results are not an untouched audit and cannot approve trading."
        ),
        "model_test_dates": sorted(set(model_folds)),
        "portfolio_hypothesis_attempt_count": len(results),
        "results": results,
    }
    (arguments.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(  # noqa: T201
        json.dumps(
            [
                {
                    "policy": result["policy"],
                    "top": result["tail_fraction"],
                    **result["summary"],
                }
                for result in results
            ],
            indent=2,
        )
    )
    return 0


def _has_book_features(frame: pd.DataFrame) -> bool:
    return any(
        column in frame.columns
        for column in (
            "book_available",
            "spread_bps",
            "depth_imbalance",
            "microprice_displacement_bps",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

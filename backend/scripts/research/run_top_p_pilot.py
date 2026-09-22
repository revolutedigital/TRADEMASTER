"""Run the bounded linear-model/top-p development pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.top_p_model import (
    run_calibrated_walk_forward,
    summarize_walk_forward,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "research-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument("--target", choices=("expected", "stress"), default="expected")
    arguments = parser.parse_args()
    paths = sorted(arguments.dataset_root.glob("date=*/research_rows.parquet"))
    if not paths:
        parser.error("no research dataset partitions found")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    target_column = (
        "paid_expected_before_stop" if arguments.target == "expected" else "paid_stress_before_stop"
    )
    frame["target"] = frame[target_column].astype("int8")
    results = []
    feature_sets = ["flow", "flow_price", "flow_price_session"]
    if _has_auxiliary_features(frame):
        feature_sets.extend(["flow_aux", "flow_price_aux", "flow_price_aux_session"])
    if _has_book_features(frame):
        feature_sets.extend(["flow_book", "flow_price_book", "flow_price_book_session"])
        if _has_auxiliary_features(frame):
            feature_sets.extend(
                ["flow_book_aux", "flow_price_book_aux", "flow_price_book_aux_session"]
            )
    for horizon in (120, 300):
        for feature_set in feature_sets:
            result = run_calibrated_walk_forward(
                frame,
                horizon_seconds=horizon,
                feature_set=feature_set,
            )
            results.append(
                {
                    "summary": summarize_walk_forward(result),
                    "detail": result.to_dict(),
                }
            )
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "experiment_stage": "development_pilot",
        "target": target_column,
        "opened_dates": [path.parent.name.removeprefix("date=") for path in paths],
        "hypothesis_attempt_count": len(results),
        "results": results,
    }
    output = arguments.output or (
        REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / f"top-p-pilot-{arguments.target}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps([result["summary"] for result in results], indent=2))  # noqa: T201
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


def _has_auxiliary_features(frame: pd.DataFrame) -> bool:
    return any(
        any(column.startswith(prefix) for column in frame.columns)
        for prefix in (
            "mark_available",
            "mark_index_basis_bps",
            "funding_rate",
            "liquidation_net_qty_1s",
            "spot_available",
            "spot_perp_basis_bps",
            "spot_perp_return_gap_",
            "spot_perp_flow_gap_",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

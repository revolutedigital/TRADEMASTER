"""Run the bounded linear-model/top-p development pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.services.research.top_p_model import (
    run_calibrated_walk_forward,
    summarize_walk_forward,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


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
    for horizon in (120, 300):
        for feature_set in ("flow", "flow_price", "flow_price_session"):
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


if __name__ == "__main__":
    raise SystemExit(main())

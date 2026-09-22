"""Build immutable causal feature/label partitions from normalized trades."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from app.services.research.research_dataset import build_research_partition


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "normalized" / "aggTrades",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "research-v1",
    )
    arguments = parser.parse_args()
    if arguments.end_date < arguments.start_date:
        parser.error("--end-date cannot precede --start-date")
    cursor = arguments.start_date
    while cursor <= arguments.end_date:
        result = build_research_partition(
            source_root=arguments.source_root,
            output_root=arguments.output_root,
            utc_date=cursor,
        )
        print(json.dumps(result.__dict__, sort_keys=True), flush=True)  # noqa: T201
        cursor += timedelta(days=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

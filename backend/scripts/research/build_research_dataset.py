"""Build immutable causal feature/label partitions from normalized trades."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import ResearchDatasetConfig, build_research_partition


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
    parser.add_argument(
        "--book-source-root",
        type=Path,
        help=(
            "Optional top-of-book root with date=*/events.parquet or date=*/events.jsonl.gz "
            "partitions, for example the prospective WAL depth directory."
        ),
    )
    parser.add_argument(
        "--mark-source-root",
        type=Path,
        help=(
            "Optional mark-price root with date=*/events.parquet or date=*/events.jsonl.gz "
            "partitions, for example the prospective WAL mark_price directory."
        ),
    )
    parser.add_argument(
        "--liquidation-source-root",
        type=Path,
        help=(
            "Optional liquidation root with date=*/events.parquet or date=*/events.jsonl.gz "
            "partitions, for example the prospective WAL liquidation directory."
        ),
    )
    parser.add_argument(
        "--require-book-features",
        action="store_true",
        help="Fail closed unless every decision has fresh book features.",
    )
    parser.add_argument(
        "--max-book-staleness-ms",
        type=int,
        default=1_000,
        help="Maximum quote age when --require-book-features is set.",
    )
    arguments = parser.parse_args()
    if arguments.end_date < arguments.start_date:
        parser.error("--end-date cannot precede --start-date")
    if arguments.require_book_features and arguments.book_source_root is None:
        parser.error("--require-book-features requires --book-source-root")
    config = ResearchDatasetConfig(
        require_book_features=arguments.require_book_features,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
    )
    cursor = arguments.start_date
    while cursor <= arguments.end_date:
        result = build_research_partition(
            source_root=arguments.source_root,
            output_root=arguments.output_root,
            utc_date=cursor,
            config=config,
            book_source_root=arguments.book_source_root,
            mark_source_root=arguments.mark_source_root,
            liquidation_source_root=arguments.liquidation_source_root,
        )
        print(json.dumps(result.__dict__, sort_keys=True), flush=True)  # noqa: T201
        cursor += timedelta(days=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

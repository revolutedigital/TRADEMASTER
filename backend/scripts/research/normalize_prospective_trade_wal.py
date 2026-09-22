"""Normalize prospective recorder TRADE WAL files into replay parquet partitions."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.market.microstructure_dataset import normalize_trade_wal_jsonl


DEFAULT_WAL_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1" / "prospective-wal"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1" / "normalized" / "trades"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--wal-root", type=Path, default=DEFAULT_WAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    arguments = parser.parse_args()
    if arguments.end_date < arguments.start_date:
        parser.error("--end-date must not precede --start-date")
    if arguments.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")

    cursor = arguments.start_date
    while cursor <= arguments.end_date:
        wal_path = arguments.wal_root / "trade" / f"date={cursor.isoformat()}" / "events.jsonl.gz"
        normalized_path = arguments.output_root / f"date={cursor.isoformat()}" / "events.parquet"
        manifest = normalize_trade_wal_jsonl(
            wal_path=wal_path,
            normalized_path=normalized_path,
            symbol=arguments.symbol,
            utc_date=cursor,
            chunk_rows=arguments.chunk_rows,
        )
        print(manifest.model_dump_json(), flush=True)  # noqa: T201
        cursor += timedelta(days=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

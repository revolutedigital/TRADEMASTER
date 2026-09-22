"""Build verified daily Binance public trade partitions without credentials."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.market.microstructure_dataset import (
    ArchiveProduct,
    BinancePublicArchiveClient,
    combined_manifest_hash,
    normalize_trade_archive,
)


DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"


def _dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


async def build(arguments: argparse.Namespace) -> dict[str, object]:
    client = BinancePublicArchiveClient(market=arguments.market)
    manifests = []
    for kind in arguments.kinds:
        for utc_date in _dates(arguments.start_date, arguments.end_date):
            filename = f"{arguments.symbol}-{kind}-{utc_date.isoformat()}.zip"
            raw_path = arguments.root / _raw_directory(arguments.market, kind) / filename
            normalized_path = (
                arguments.root
                / "normalized"
                / _normalized_directory(arguments.market, kind)
                / f"date={utc_date.isoformat()}"
                / "events.parquet"
            )
            archive_path, source_sha256, source_url = await client.download_daily_archive(
                symbol=arguments.symbol,
                kind=kind,
                utc_date=utc_date,
                destination=raw_path,
            )
            manifest = normalize_trade_archive(
                archive_path=archive_path,
                normalized_path=normalized_path,
                source_url=source_url,
                source_sha256=source_sha256,
                symbol=arguments.symbol,
                kind=kind,
                utc_date=utc_date,
                product=arguments.market,
            )
            manifests.append(manifest)
            sys.stdout.write(
                f"{utc_date} {kind}: {manifest.row_count:,} rows, "
                f"gaps={manifest.sequence_gap_count}, {manifest.quality_status}\n"
            )

    start_at = datetime.combine(arguments.start_date, datetime.min.time(), tzinfo=UTC)
    end_at = datetime.combine(
        arguments.end_date + timedelta(days=1),
        datetime.min.time(),
        tzinfo=UTC,
    )
    funding_rows = 0
    mark_price_rows = 0
    if arguments.market == "usdm_perpetual":
        funding = await client.funding_history(
            symbol=arguments.symbol,
            start_at=start_at,
            end_at=end_at,
        )
        mark_prices = await client.mark_price_klines(
            symbol=arguments.symbol,
            start_at=start_at,
            end_at=end_at,
        )
        reference_root = arguments.root / "normalized" / "reference"
        reference_root.mkdir(parents=True, exist_ok=True)
        funding.to_parquet(reference_root / "funding.parquet", index=False)
        mark_prices.to_parquet(reference_root / "mark-price-1m.parquet", index=False)
        funding_rows = len(funding)
        mark_price_rows = len(mark_prices)

    summary = {
        "schema_version": 1,
        "symbol": arguments.symbol,
        "product": arguments.market,
        "start_date": arguments.start_date.isoformat(),
        "end_date": arguments.end_date.isoformat(),
        "partition_count": len(manifests),
        "partition_manifest_sha256": combined_manifest_hash(manifests),
        "funding_rows": funding_rows,
        "mark_price_rows": mark_price_rows,
        "partitions": [manifest.model_dump(mode="json") for manifest in manifests],
    }
    manifest_name = (
        "dataset-manifest.json"
        if arguments.market == "usdm_perpetual"
        else f"{arguments.market}-dataset-manifest.json"
    )
    (arguments.root / manifest_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _raw_directory(market: ArchiveProduct, kind: str) -> Path:
    return Path("raw") / kind if market == "usdm_perpetual" else Path("raw") / market / kind


def _normalized_directory(market: ArchiveProduct, kind: str) -> str:
    return kind if market == "usdm_perpetual" else f"{market}-{kind}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--market",
        choices=("usdm_perpetual", "spot"),
        default="usdm_perpetual",
        help="Public archive market to download. spot is signal-only for research joins.",
    )
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--kinds", nargs="+", choices=("aggTrades", "trades"), default=["aggTrades"]
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args()
    if arguments.end_date < arguments.start_date:
        parser.error("--end-date must not precede --start-date")
    asyncio.run(build(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

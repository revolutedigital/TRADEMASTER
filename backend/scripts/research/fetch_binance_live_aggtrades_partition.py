"""Fetch a diagnostic live Binance USD-M aggTrades partition.

This is a research-only bridge for an unclosed UTC day, when Binance Vision's
checksumed daily archive is not available yet. It uses the credential-free
futures market-data API, writes raw JSONL plus normalized Parquet, and marks the
manifest as DIAGNOSTIC_LIVE_PARTIAL rather than archive-verified.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.microstructure import DatasetPartitionManifest, MarketEventType  # noqa: E402

FUTURES_MARKET_URL = "https://fapi.binance.com"
DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_PAGE_DELAY_SECONDS = 0.05
MAX_RETRY_DELAY_SECONDS = 30.0
PAGE_LIMIT = 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--utc-date", type=date.fromisoformat, required=True)
    parser.add_argument("--start-at", type=parse_utc_datetime, required=True)
    parser.add_argument("--end-at", type=parse_utc_datetime, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--market-base-url", default=FUTURES_MARKET_URL)
    parser.add_argument(
        "--replace-partial",
        action="store_true",
        help="Allow replacing an existing DIAGNOSTIC_LIVE_PARTIAL partition.",
    )
    arguments = parser.parse_args(argv)
    manifest = asyncio.run(
        fetch_and_normalize_live_aggtrades_partition(
            symbol=arguments.symbol,
            utc_date=arguments.utc_date,
            start_at=arguments.start_at,
            end_at=arguments.end_at,
            root=arguments.root,
            market_base_url=arguments.market_base_url,
            replace_partial=arguments.replace_partial,
        )
    )
    print(manifest.model_dump_json(indent=2))  # noqa: T201
    return 0


def parse_utc_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include timezone")
    return parsed.astimezone(UTC)


async def fetch_and_normalize_live_aggtrades_partition(
    *,
    symbol: str,
    utc_date: date,
    start_at: datetime,
    end_at: datetime,
    root: Path,
    market_base_url: str = FUTURES_MARKET_URL,
    replace_partial: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DatasetPartitionManifest:
    validate_range(utc_date=utc_date, start_at=start_at, end_at=end_at)
    symbol = symbol.upper()
    normalized_path = root / "normalized" / "aggTrades" / f"date={utc_date.isoformat()}" / "events.parquet"
    assert_replace_is_safe(normalized_path=normalized_path, replace_partial=replace_partial)
    rows = await fetch_aggregate_trades(
        symbol=symbol,
        start_at=start_at,
        end_at=end_at,
        market_base_url=market_base_url,
        transport=transport,
    )
    if not rows:
        raise ValueError("Binance returned no aggregate trades for the requested range")
    raw_path = (
        root
        / "raw"
        / "live-aggTrades"
        / symbol
        / f"date={utc_date.isoformat()}"
        / f"{symbol}-aggTrades-{start_at:%H%M%S}-{end_at:%H%M%S}.jsonl.gz"
    )
    write_raw_rows(raw_path, rows)
    source_url = (
        f"{market_base_url.rstrip('/')}/fapi/v1/aggTrades?"
        f"symbol={symbol}&startTime={milliseconds(start_at)}&endTime={milliseconds(end_at)}&limit={PAGE_LIMIT}"
    )
    return normalize_live_aggregate_trades(
        rows=rows,
        normalized_path=normalized_path,
        raw_path=raw_path,
        source_url=source_url,
        symbol=symbol,
        utc_date=utc_date,
    )


async def extend_and_normalize_live_aggtrades_partition(
    *,
    symbol: str,
    utc_date: date,
    start_at: datetime,
    end_at: datetime,
    root: Path,
    market_base_url: str = FUTURES_MARKET_URL,
    replace_partial: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DatasetPartitionManifest:
    """Extend an existing diagnostic live partial partition by fetching only the missing tail."""

    validate_range(utc_date=utc_date, start_at=start_at, end_at=end_at)
    symbol = symbol.upper()
    normalized_path = root / "normalized" / "aggTrades" / f"date={utc_date.isoformat()}" / "events.parquet"
    manifest_path = normalized_path.with_suffix(".manifest.json")
    if not normalized_path.exists() or not manifest_path.exists():
        return await fetch_and_normalize_live_aggtrades_partition(
            symbol=symbol,
            utc_date=utc_date,
            start_at=start_at,
            end_at=end_at,
            root=root,
            market_base_url=market_base_url,
            replace_partial=replace_partial,
            transport=transport,
        )
    assert_replace_is_safe(normalized_path=normalized_path, replace_partial=replace_partial)
    existing_manifest = DatasetPartitionManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if existing_manifest.symbol != symbol:
        raise ValueError(f"existing manifest symbol mismatch: {existing_manifest.symbol} != {symbol}")
    if existing_manifest.utc_date != utc_date.isoformat():
        raise ValueError(f"existing manifest utc_date mismatch: {existing_manifest.utc_date}")
    if existing_manifest.last_event_time >= end_at:
        return existing_manifest

    fetch_start = max(start_at, existing_manifest.last_event_time.astimezone(UTC))
    fetch_start = datetime.fromtimestamp(fetch_start.timestamp() + 0.001, UTC)
    if fetch_start >= end_at:
        return existing_manifest
    rows = await fetch_aggregate_trades(
        symbol=symbol,
        start_at=fetch_start,
        end_at=end_at,
        market_base_url=market_base_url,
        transport=transport,
    )
    if not rows:
        return existing_manifest
    raw_path = (
        root
        / "raw"
        / "live-aggTrades"
        / symbol
        / f"date={utc_date.isoformat()}"
        / f"{symbol}-aggTrades-{fetch_start:%H%M%S}-{end_at:%H%M%S}.jsonl.gz"
    )
    write_raw_rows(raw_path, rows)
    existing = pd.read_parquet(normalized_path)
    new = aggregate_trade_rows_to_frame(rows=rows, symbol=symbol, utc_date=utc_date)
    merged = (
        pd.concat([existing, new], ignore_index=True)
        .sort_values(["sequence_id", "event_time"], kind="stable")
        .drop_duplicates(subset=["sequence_id"], keep="last")
    )
    source_url = (
        f"{market_base_url.rstrip('/')}/fapi/v1/aggTrades?"
        f"symbol={symbol}&startTime={milliseconds(fetch_start)}&endTime={milliseconds(end_at)}&limit={PAGE_LIMIT}"
    )
    return write_normalized_manifest(
        normalized=merged,
        normalized_path=normalized_path,
        raw_path=raw_path,
        source_url=source_url,
        symbol=symbol,
        utc_date=utc_date,
    )


async def fetch_aggregate_trades(
    *,
    symbol: str,
    start_at: datetime,
    end_at: datetime,
    market_base_url: str = FUTURES_MARKET_URL,
    page_delay_seconds: float = DEFAULT_PAGE_DELAY_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, Any]]:
    start_milliseconds = milliseconds(start_at)
    end_milliseconds = milliseconds(end_at)
    rows: list[dict[str, Any]] = []
    next_from_id: int | None = None
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
        follow_redirects=True,
        transport=transport,
    ) as client:
        while True:
            params: dict[str, Any] = {"symbol": symbol.upper(), "limit": PAGE_LIMIT}
            if next_from_id is None:
                params["startTime"] = start_milliseconds
                params["endTime"] = end_milliseconds
            else:
                params["fromId"] = next_from_id
            response = await get_with_retry(
                client,
                f"{market_base_url.rstrip('/')}/fapi/v1/aggTrades",
                params=params,
            )
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError("Binance aggregate-trade endpoint returned a non-list payload")
            if not payload:
                break
            page_rows = [row for row in payload if start_milliseconds <= int(row["T"]) <= end_milliseconds]
            rows.extend(page_rows)
            last_id = max(int(row["a"]) for row in payload)
            last_time = max(int(row["T"]) for row in payload)
            if len(payload) < PAGE_LIMIT or last_time > end_milliseconds:
                break
            next_from_id = last_id + 1
            await asyncio.sleep(page_delay_seconds)
    return dedupe_rows(rows)


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any],
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as error:
            last_error = error
            if attempt < 5 and error.response.status_code in {418, 429}:
                await asyncio.sleep(retry_delay_seconds(error.response, attempt))
                continue
            if attempt < 5 and 500 <= error.response.status_code < 600:
                await asyncio.sleep(retry_delay_seconds(error.response, attempt))
                continue
        except httpx.HTTPError as error:
            last_error = error
            if attempt < 5:
                await asyncio.sleep(retry_delay_seconds(None, attempt))
    raise ValueError(f"Unable to fetch Binance aggregate trades: {last_error}") from last_error


def retry_delay_seconds(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(MAX_RETRY_DELAY_SECONDS, max(1.0, float(retry_after)))
            except ValueError:
                pass
    return min(MAX_RETRY_DELAY_SECONDS, float(2 ** attempt))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_id[int(row["a"])] = row
    return [by_id[sequence_id] for sequence_id in sorted(by_id)]


def normalize_live_aggregate_trades(
    *,
    rows: list[dict[str, Any]],
    normalized_path: Path,
    raw_path: Path,
    source_url: str,
    symbol: str,
    utc_date: date,
) -> DatasetPartitionManifest:
    normalized = aggregate_trade_rows_to_frame(rows=rows, symbol=symbol, utc_date=utc_date)
    return write_normalized_manifest(
        normalized=normalized,
        normalized_path=normalized_path,
        raw_path=raw_path,
        source_url=source_url,
        symbol=symbol,
        utc_date=utc_date,
    )


def aggregate_trade_rows_to_frame(
    *,
    rows: list[dict[str, Any]],
    symbol: str,
    utc_date: date,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {"a", "p", "q", "f", "l", "T", "m"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"aggregate trade rows are missing fields: {sorted(missing)}")
    event_times = pd.to_datetime(pd.to_numeric(frame["T"], errors="raise"), unit="ms", utc=True)
    if set(event_times.dt.date) != {utc_date}:
        raise ValueError("aggregate trade rows cross the requested UTC date")

    normalized = pd.DataFrame(
        {
            "venue": "binance",
            "product": "usdm_perpetual",
            "symbol": symbol.upper(),
            "event_type": MarketEventType.AGG_TRADE.value,
            "event_time": event_times,
            "sequence_id": pd.to_numeric(frame["a"], errors="raise").astype("int64"),
            "price": pd.to_numeric(frame["p"], errors="raise").astype("float64"),
            "quantity": pd.to_numeric(frame["q"], errors="raise").astype("float64"),
            "is_buyer_maker": frame["m"].map(parse_bool),
            "first_sequence_id": pd.to_numeric(frame["f"], errors="raise").astype("int64"),
            "last_sequence_id": pd.to_numeric(frame["l"], errors="raise").astype("int64"),
        }
    ).sort_values(["sequence_id", "event_time"])
    normalized["quote_quantity"] = normalized["price"] * normalized["quantity"]
    value_columns = ["price", "quantity", "quote_quantity"]
    if not np.isfinite(normalized[value_columns]).all().all():
        raise ValueError("normalized aggregate trades contain non-finite numeric values")
    if (normalized[value_columns] <= 0).any().any():
        raise ValueError("normalized aggregate trades must have positive price, quantity and quote quantity")
    return normalized


def write_normalized_manifest(
    *,
    normalized: pd.DataFrame,
    normalized_path: Path,
    raw_path: Path,
    source_url: str,
    symbol: str,
    utc_date: date,
) -> DatasetPartitionManifest:
    sequence_ids = normalized["sequence_id"].to_numpy(dtype=np.int64)
    duplicate_sequence_count = int(pd.Series(sequence_ids).duplicated().sum())
    if duplicate_sequence_count:
        raise ValueError(f"aggregate trades contain {duplicate_sequence_count} duplicate sequence IDs")
    differences = np.diff(sequence_ids)
    if (differences < 0).any():
        raise ValueError("aggregate trade sequence IDs regress")
    sequence_gap_count = int(np.sum(differences > 1))

    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = normalized_path.with_suffix(f"{normalized_path.suffix}.part")
    temporary_path.unlink(missing_ok=True)
    normalized.to_parquet(temporary_path, compression="zstd", index=False)
    temporary_path.replace(normalized_path)
    quality_status = (
        "DIAGNOSTIC_LIVE_PARTIAL_VALID"
        if sequence_gap_count == 0
        else "DIAGNOSTIC_LIVE_PARTIAL_VALID_WITH_SEQUENCE_GAPS"
    )
    manifest = DatasetPartitionManifest(
        venue="binance",
        product="usdm_perpetual",
        symbol=symbol.upper(),
        event_type=MarketEventType.AGG_TRADE,
        utc_date=utc_date.isoformat(),
        source_url=source_url,
        source_sha256=sha256_file(raw_path),
        normalized_sha256=sha256_file(normalized_path),
        row_count=int(len(normalized)),
        first_event_time=normalized["event_time"].iloc[0].to_pydatetime(),
        last_event_time=normalized["event_time"].iloc[-1].to_pydatetime(),
        first_sequence_id=int(sequence_ids[0]),
        last_sequence_id=int(sequence_ids[-1]),
        sequence_gap_count=sequence_gap_count,
        duplicate_sequence_count=duplicate_sequence_count,
        quality_status=quality_status,
    )
    normalized_path.with_suffix(".manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return manifest


def validate_range(*, utc_date: date, start_at: datetime, end_at: datetime) -> None:
    if start_at.tzinfo is None or end_at.tzinfo is None:
        raise ValueError("start_at and end_at must be timezone-aware")
    start_at = start_at.astimezone(UTC)
    end_at = end_at.astimezone(UTC)
    if end_at <= start_at:
        raise ValueError("end_at must follow start_at")
    if start_at.date() != utc_date or end_at.date() != utc_date:
        raise ValueError("start_at and end_at must stay within utc_date")


def assert_replace_is_safe(*, normalized_path: Path, replace_partial: bool) -> None:
    if not normalized_path.exists():
        return
    manifest_path = normalized_path.with_suffix(".manifest.json")
    if not replace_partial or not manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing partition: {normalized_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality_status = str(manifest.get("quality_status", ""))
    if not quality_status.startswith("DIAGNOSTIC_LIVE_PARTIAL"):
        raise FileExistsError(f"refusing to overwrite non-live-partial partition: {normalized_path}")


def write_raw_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.part")
    with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary_path.replace(path)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def milliseconds(value: datetime) -> int:
    return int(value.astimezone(UTC).timestamp() * 1000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

"""Verified Binance USD-M public archives normalized into immutable Parquet."""

from __future__ import annotations

import asyncio
import hashlib
import json
import zipfile
from datetime import UTC, date, datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import pandas as pd

from app.schemas.microstructure import DatasetPartitionManifest, MarketEventType


ArchiveKind = Literal["aggTrades", "trades"]
ARCHIVE_BASE_URL = "https://data.binance.vision/data/futures/um/daily"
FUTURES_MARKET_URL = "https://fapi.binance.com"
ARCHIVE_COLUMNS: dict[ArchiveKind, tuple[str, ...]] = {
    "aggTrades": (
        "sequence_id",
        "price",
        "quantity",
        "first_sequence_id",
        "last_sequence_id",
        "event_timestamp",
        "is_buyer_maker",
    ),
    "trades": (
        "sequence_id",
        "price",
        "quantity",
        "quote_quantity",
        "event_timestamp",
        "is_buyer_maker",
    ),
}


class DatasetIntegrityError(ValueError):
    """Raised when source or normalized event data cannot be trusted."""


class BinancePublicArchiveClient:
    """Credential-free downloader with checksum validation and atomic writes."""

    def __init__(
        self,
        *,
        archive_base_url: str = ARCHIVE_BASE_URL,
        market_base_url: str = FUTURES_MARKET_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._archive_base_url = archive_base_url.rstrip("/")
        self._market_base_url = market_base_url.rstrip("/")
        self._transport = transport

    def archive_url(self, *, symbol: str, kind: ArchiveKind, utc_date: date) -> str:
        filename = f"{symbol.upper()}-{kind}-{utc_date.isoformat()}.zip"
        return f"{self._archive_base_url}/{kind}/{symbol.upper()}/{filename}"

    async def download_daily_archive(
        self,
        *,
        symbol: str,
        kind: ArchiveKind,
        utc_date: date,
        destination: Path,
    ) -> tuple[Path, str, str]:
        source_url = self.archive_url(symbol=symbol, kind=kind, utc_date=utc_date)
        checksum_url = f"{source_url}.CHECKSUM"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120),
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            checksum_response = await self._get_with_retry(client, checksum_url)
            expected_sha256 = checksum_response.text.strip().split()[0].lower()
            if len(expected_sha256) != 64:
                raise DatasetIntegrityError(f"Invalid checksum document at {checksum_url}")
            checksum_path = destination.with_name(f"{destination.name}.CHECKSUM")

            if destination.exists() and _sha256_file(destination) == expected_sha256:
                if not checksum_path.exists():
                    checksum_path.parent.mkdir(parents=True, exist_ok=True)
                    checksum_path.write_text(checksum_response.text, encoding="utf-8")
                return destination, expected_sha256, source_url

            archive_response = await self._get_with_retry(client, source_url)
            observed_sha256 = hashlib.sha256(archive_response.content).hexdigest()
            if observed_sha256 != expected_sha256:
                raise DatasetIntegrityError(
                    f"Checksum mismatch for {source_url}: {observed_sha256} != {expected_sha256}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_suffix(f"{destination.suffix}.part")
            temporary_path.write_bytes(archive_response.content)
            temporary_path.replace(destination)
            checksum_path.write_text(checksum_response.text, encoding="utf-8")
            return destination, expected_sha256, source_url

    async def funding_history(
        self,
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
    ) -> pd.DataFrame:
        _validate_utc_range(start_at, end_at)
        rows: list[dict[str, Any]] = []
        cursor = int(start_at.timestamp() * 1000)
        end_milliseconds = int(end_at.timestamp() * 1000)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30),
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            while cursor < end_milliseconds:
                response = await self._get_with_retry(
                    client,
                    f"{self._market_base_url}/fapi/v1/fundingRate",
                    params={
                        "symbol": symbol.upper(),
                        "startTime": cursor,
                        "endTime": end_milliseconds,
                        "limit": 1000,
                    },
                )
                payload = response.json()
                if not isinstance(payload, list):
                    raise DatasetIntegrityError("Funding endpoint returned a non-list payload")
                if not payload:
                    break
                rows.extend(item for item in payload if isinstance(item, dict))
                next_cursor = max(int(item["fundingTime"]) for item in payload) + 1
                if next_cursor <= cursor:
                    raise DatasetIntegrityError("Funding pagination did not advance")
                cursor = next_cursor
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["event_time", "funding_rate", "mark_price"])
        frame["event_time"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
        frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="raise")
        frame["mark_price"] = pd.to_numeric(frame.get("markPrice"), errors="coerce")
        return (
            frame[["event_time", "funding_rate", "mark_price"]]
            .drop_duplicates("event_time")
            .sort_values("event_time")
            .reset_index(drop=True)
        )

    async def mark_price_klines(
        self,
        *,
        symbol: str,
        start_at: datetime,
        end_at: datetime,
        interval: str = "1m",
    ) -> pd.DataFrame:
        _validate_utc_range(start_at, end_at)
        rows: list[list[Any]] = []
        cursor = int(start_at.timestamp() * 1000)
        end_milliseconds = int(end_at.timestamp() * 1000)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30),
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            while cursor < end_milliseconds:
                response = await self._get_with_retry(
                    client,
                    f"{self._market_base_url}/fapi/v1/markPriceKlines",
                    params={
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "startTime": cursor,
                        "endTime": end_milliseconds,
                        "limit": 1500,
                    },
                )
                payload = response.json()
                if not isinstance(payload, list):
                    raise DatasetIntegrityError("Mark-price endpoint returned a non-list payload")
                if not payload:
                    break
                rows.extend(item for item in payload if isinstance(item, list))
                next_cursor = max(int(item[6]) for item in payload) + 1
                if next_cursor <= cursor:
                    raise DatasetIntegrityError("Mark-price pagination did not advance")
                cursor = next_cursor
                await asyncio.sleep(0)
        columns = (
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "ignore_volume",
            "close_time",
            "ignore_quote_volume",
            "ignore_trade_count",
            "ignore_taker_base",
            "ignore_taker_quote",
            "ignore",
        )
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return pd.DataFrame(columns=["event_time", "open", "high", "low", "close"])
        frame["event_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return (
            frame[["event_time", "open", "high", "low", "close"]]
            .drop_duplicates("event_time")
            .sort_values("event_time")
            .reset_index(drop=True)
        )

    @staticmethod
    async def _get_with_retry(
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response
            except httpx.HTTPError as error:
                last_error = error
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise DatasetIntegrityError(f"Unable to download {url}: {last_error}") from last_error


def normalize_trade_archive(
    *,
    archive_path: Path,
    normalized_path: Path,
    source_url: str,
    source_sha256: str,
    symbol: str,
    kind: ArchiveKind,
    utc_date: date,
    chunk_rows: int = 250_000,
) -> DatasetPartitionManifest:
    """Normalize one verified archive without loading the whole day into memory."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("The research extra with pyarrow is required") from error

    columns = ARCHIVE_COLUMNS[kind]
    event_type = MarketEventType.AGG_TRADE if kind == "aggTrades" else MarketEventType.TRADE
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = normalized_path.with_suffix(f"{normalized_path.suffix}.part")
    temporary_path.unlink(missing_ok=True)

    row_count = 0
    first_event_time: datetime | None = None
    last_event_time: datetime | None = None
    first_sequence_id: int | None = None
    last_sequence_id: int | None = None
    sequence_gap_count = 0
    duplicate_sequence_count = 0
    parquet_writer: pq.ParquetWriter | None = None

    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise DatasetIntegrityError(
                f"Expected one CSV in {archive_path.name}, found {len(members)}"
            )
        with (
            archive.open(members[0]) as csv_file,
            TextIOWrapper(csv_file, encoding="utf-8") as text_file,
        ):
            chunks: Any = pd.read_csv(
                text_file,
                header=None,
                names=columns,
                chunksize=chunk_rows,
                dtype=str,
            )
            for raw_chunk in chunks:
                chunk = _normalize_trade_chunk(raw_chunk, kind=kind, symbol=symbol)
                if chunk.empty:
                    continue
                sequence_ids = chunk["sequence_id"].to_numpy(dtype=np.int64)
                duplicate_sequence_count += int(pd.Series(sequence_ids).duplicated().sum())
                differences = np.diff(sequence_ids)
                sequence_gap_count += int(np.sum(differences > 1))
                if (differences < 0).any():
                    raise DatasetIntegrityError("Sequence IDs are not monotonic")
                if last_sequence_id is not None:
                    boundary_difference = int(sequence_ids[0]) - last_sequence_id
                    if boundary_difference < 0:
                        raise DatasetIntegrityError("Sequence IDs regress across chunks")
                    if boundary_difference == 0:
                        duplicate_sequence_count += 1
                    elif boundary_difference > 1:
                        sequence_gap_count += 1

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        temporary_path,
                        table.schema,
                        compression="zstd",
                    )
                parquet_writer.write_table(table)
                row_count += len(chunk)
                first_event_time = first_event_time or chunk["event_time"].iloc[0].to_pydatetime()
                last_event_time = chunk["event_time"].iloc[-1].to_pydatetime()
                if first_sequence_id is None:
                    first_sequence_id = int(sequence_ids[0])
                last_sequence_id = int(sequence_ids[-1])

    if parquet_writer is None:
        raise DatasetIntegrityError(f"Archive {archive_path.name} contained no valid rows")
    parquet_writer.close()
    if duplicate_sequence_count:
        temporary_path.unlink(missing_ok=True)
        raise DatasetIntegrityError(
            f"Archive {archive_path.name} contains {duplicate_sequence_count} duplicate sequences"
        )
    temporary_path.replace(normalized_path)
    normalized_sha256 = _sha256_file(normalized_path)
    quality_status = "VALID" if sequence_gap_count == 0 else "VALID_WITH_SEQUENCE_GAPS"
    manifest = DatasetPartitionManifest(
        venue="binance",
        product="usdm_perpetual",
        symbol=symbol.upper(),
        event_type=event_type,
        utc_date=utc_date.isoformat(),
        source_url=source_url,
        source_sha256=source_sha256,
        normalized_sha256=normalized_sha256,
        row_count=row_count,
        first_event_time=first_event_time,
        last_event_time=last_event_time,
        first_sequence_id=first_sequence_id,
        last_sequence_id=last_sequence_id,
        sequence_gap_count=sequence_gap_count,
        duplicate_sequence_count=duplicate_sequence_count,
        quality_status=quality_status,
    )
    manifest_path = normalized_path.with_suffix(".manifest.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _normalize_trade_chunk(
    raw_chunk: pd.DataFrame,
    *,
    kind: ArchiveKind,
    symbol: str,
) -> pd.DataFrame:
    chunk = raw_chunk.copy()
    for column in ARCHIVE_COLUMNS[kind]:
        if column == "is_buyer_maker":
            continue
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
    chunk = chunk.dropna(subset=["sequence_id", "price", "quantity", "event_timestamp"])
    timestamp_values = chunk["event_timestamp"].astype("int64")
    timestamp_unit = "us" if timestamp_values.median() >= 100_000_000_000_000 else "ms"
    normalized = pd.DataFrame(
        {
            "venue": "binance",
            "product": "usdm_perpetual",
            "symbol": symbol.upper(),
            "event_type": "AGG_TRADE" if kind == "aggTrades" else "TRADE",
            "event_time": pd.to_datetime(timestamp_values, unit=timestamp_unit, utc=True),
            "sequence_id": chunk["sequence_id"].astype("int64"),
            "price": chunk["price"].astype("float64"),
            "quantity": chunk["quantity"].astype("float64"),
            "quote_quantity": (
                chunk["quote_quantity"].astype("float64")
                if "quote_quantity" in chunk
                else chunk["price"].astype("float64") * chunk["quantity"].astype("float64")
            ),
            "is_buyer_maker": chunk["is_buyer_maker"].map(_parse_bool),
            "first_sequence_id": (
                chunk["first_sequence_id"].astype("int64")
                if "first_sequence_id" in chunk
                else chunk["sequence_id"].astype("int64")
            ),
            "last_sequence_id": (
                chunk["last_sequence_id"].astype("int64")
                if "last_sequence_id" in chunk
                else chunk["sequence_id"].astype("int64")
            ),
        }
    )
    if not np.isfinite(normalized[["price", "quantity", "quote_quantity"]]).all().all():
        raise DatasetIntegrityError("Normalized trade values contain non-finite numbers")
    if (normalized[["price", "quantity", "quote_quantity"]] <= 0).any().any():
        raise DatasetIntegrityError("Normalized trade values must be positive")
    return normalized.sort_values(["sequence_id", "event_time"]).reset_index(drop=True)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise DatasetIntegrityError(f"Invalid boolean value: {value!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_utc_range(start_at: datetime, end_at: datetime) -> None:
    if start_at.tzinfo is None or end_at.tzinfo is None:
        raise ValueError("start_at and end_at must be timezone-aware")
    if end_at.astimezone(UTC) <= start_at.astimezone(UTC):
        raise ValueError("end_at must follow start_at")


def combined_manifest_hash(manifests: list[DatasetPartitionManifest]) -> str:
    """Stable experiment input hash independent of filesystem location."""
    canonical = json.dumps(
        [manifest.model_dump(mode="json") for manifest in manifests],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

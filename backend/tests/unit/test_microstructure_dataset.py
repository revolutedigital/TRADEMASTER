"""Verified archive normalization preserves event order and lineage."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.market.microstructure_dataset import (
    BinancePublicArchiveClient,
    DatasetIntegrityError,
    normalize_trade_archive,
)


def _zip_csv(rows: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-aggTrades-2026-01-01.csv", rows)
    return buffer.getvalue()


def test_normalize_aggregate_trades_and_write_manifest(tmp_path: Path) -> None:
    content = _zip_csv(
        "1,100.0,2.0,10,11,1767225600000,true\n"
        "2,101.0,1.5,12,12,1767225600100,false\n"
        "4,102.0,1.0,14,14,1767225600200,true\n"
    )
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(content)
    normalized_path = tmp_path / "events.parquet"

    manifest = normalize_trade_archive(
        archive_path=archive_path,
        normalized_path=normalized_path,
        source_url="https://data.binance.vision/example.zip",
        source_sha256=hashlib.sha256(content).hexdigest(),
        symbol="BTCUSDT",
        kind="aggTrades",
        utc_date=date(2026, 1, 1),
        chunk_rows=2,
    )

    normalized = pd.read_parquet(normalized_path)
    assert normalized["sequence_id"].tolist() == [1, 2, 4]
    assert normalized["quote_quantity"].tolist() == [200.0, 151.5, 102.0]
    assert manifest.row_count == 3
    assert manifest.sequence_gap_count == 1
    assert manifest.event_type == MarketEventType.AGG_TRADE
    assert normalized_path.with_suffix(".manifest.json").exists()


def test_duplicate_sequences_are_quarantined(tmp_path: Path) -> None:
    content = _zip_csv(
        "1,100.0,2.0,10,11,1767225600000,true\n1,101.0,1.5,12,12,1767225600100,false\n"
    )
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(content)

    with pytest.raises(DatasetIntegrityError, match="duplicate"):
        normalize_trade_archive(
            archive_path=archive_path,
            normalized_path=tmp_path / "events.parquet",
            source_url="https://data.binance.vision/example.zip",
            source_sha256=hashlib.sha256(content).hexdigest(),
            symbol="BTCUSDT",
            kind="aggTrades",
            utc_date=date(2026, 1, 1),
            chunk_rows=1,
        )


@pytest.mark.asyncio
async def test_downloader_verifies_checksum_and_writes_atomically(tmp_path: Path) -> None:
    content = _zip_csv("1,100.0,2.0,10,11,1767225600000,true\n")
    checksum = hashlib.sha256(content).hexdigest()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{checksum}  source.zip")
        return httpx.Response(200, content=content)

    client = BinancePublicArchiveClient(
        archive_base_url="https://example.test/data",
        transport=httpx.MockTransport(handler),
    )
    destination = tmp_path / "archive.zip"
    downloaded, observed_hash, _ = await client.download_daily_archive(
        symbol="BTCUSDT",
        kind="aggTrades",
        utc_date=date(2026, 1, 1),
        destination=destination,
    )

    assert downloaded == destination
    assert observed_hash == checksum
    assert destination.read_bytes() == content
    assert not destination.with_suffix(".zip.part").exists()


def test_canonical_event_rejects_crossed_quote() -> None:
    with pytest.raises(ValueError, match="ask_price"):
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.BOOK_TICKER,
            event_time=datetime.now(UTC),
            bid_price=101,
            ask_price=100,
        )

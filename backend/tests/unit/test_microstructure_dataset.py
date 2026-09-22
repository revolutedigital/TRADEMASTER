"""Verified archive normalization preserves event order and lineage."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
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
    normalize_trade_wal_jsonl,
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


def test_normalize_spot_aggregate_trades_with_microsecond_timestamp(tmp_path: Path) -> None:
    content = _zip_csv("1,100.0,2.0,10,11,1767225600000123,true,true\n")
    archive_path = tmp_path / "source.zip"
    archive_path.write_bytes(content)
    normalized_path = tmp_path / "events.parquet"

    manifest = normalize_trade_archive(
        archive_path=archive_path,
        normalized_path=normalized_path,
        source_url="https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/example.zip",
        source_sha256=hashlib.sha256(content).hexdigest(),
        symbol="BTCUSDT",
        kind="aggTrades",
        utc_date=date(2026, 1, 1),
        product="spot",
    )

    normalized = pd.read_parquet(normalized_path)
    assert normalized["product"].tolist() == ["spot"]
    assert normalized["event_time"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00.000123Z")
    assert normalized["quote_quantity"].tolist() == [200.0]
    assert manifest.product == "spot"
    assert manifest.event_type == MarketEventType.AGG_TRADE


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


def test_normalize_prospective_trade_wal_jsonl(tmp_path: Path) -> None:
    wal_path = tmp_path / "prospective-wal" / "trade" / "date=2026-01-01" / "events.jsonl.gz"
    wal_path.parent.mkdir(parents=True)
    with gzip.open(wal_path, "wt", encoding="utf-8") as handle:
        for sequence_id, price in ((10, 100.0), (11, 101.0)):
            handle.write(
                json.dumps(
                    {
                        "product": "usdm_perpetual",
                        "symbol": "BTCUSDT",
                        "event_type": "TRADE",
                        "event_time": f"2026-01-01T00:00:0{sequence_id - 9}+00:00",
                        "sequence_id": sequence_id,
                        "price": price,
                        "quantity": 2.0,
                        "quote_quantity": price * 2.0,
                        "is_buyer_maker": sequence_id % 2 == 0,
                    }
                )
                + "\n"
            )
    normalized_path = tmp_path / "normalized" / "trades" / "date=2026-01-01" / "events.parquet"

    manifest = normalize_trade_wal_jsonl(
        wal_path=wal_path,
        normalized_path=normalized_path,
        symbol="BTCUSDT",
        utc_date=date(2026, 1, 1),
        chunk_rows=1,
    )

    normalized = pd.read_parquet(normalized_path)
    assert normalized["sequence_id"].tolist() == [10, 11]
    assert normalized["quote_quantity"].tolist() == [200.0, 202.0]
    assert manifest.event_type == MarketEventType.TRADE
    assert manifest.source_sha256 == hashlib.sha256(wal_path.read_bytes()).hexdigest()
    assert manifest.row_count == 2
    assert manifest.sequence_gap_count == 0
    assert normalized_path.with_suffix(".manifest.json").exists()


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


def test_spot_downloader_uses_spot_public_archive_base() -> None:
    client = BinancePublicArchiveClient(market="spot")

    url = client.archive_url(
        symbol="BTCUSDT",
        kind="aggTrades",
        utc_date=date(2026, 1, 1),
    )

    assert url == (
        "https://data.binance.vision/data/spot/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2026-01-01.zip"
    )


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

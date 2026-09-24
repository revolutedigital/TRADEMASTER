from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from scripts.research import fetch_binance_live_aggtrades_partition as live_fetch
from scripts.research.fetch_binance_live_aggtrades_partition import (
    fetch_aggregate_trades,
    fetch_and_normalize_live_aggtrades_partition,
    normalize_live_aggregate_trades,
    write_raw_rows,
)


@pytest.mark.asyncio
async def test_fetch_aggregate_trades_paginates_with_from_id() -> None:
    calls: list[dict[str, str]] = []
    first_page = [
        {"a": index, "p": "100.0", "q": "1.0", "f": index, "l": index, "T": 1767225600000 + index, "m": False}
        for index in range(1, 1001)
    ]
    second_page = [
        {"a": 1001, "p": "101.0", "q": "2.0", "f": 1001, "l": 1001, "T": 1767225602000, "m": True}
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        if "fromId" in request.url.params:
            return httpx.Response(200, json=second_page)
        return httpx.Response(200, json=first_page)

    rows = await fetch_aggregate_trades(
        symbol="BTCUSDT",
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        market_base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )

    assert len(rows) == 1001
    assert rows[-1]["a"] == 1001
    assert calls[0]["startTime"] == "1767225600000"
    assert calls[0]["endTime"] == "1767225660000"
    assert calls[1]["fromId"] == "1001"


@pytest.mark.asyncio
async def test_fetch_aggregate_trades_retries_binance_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    attempts = 0
    row = {"a": 1, "p": "100.0", "q": "1.0", "f": 1, "l": 1, "T": 1767225600000, "m": False}

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="rate limited")
        return httpx.Response(200, json=[row])

    monkeypatch.setattr(live_fetch.asyncio, "sleep", fake_sleep)

    rows = await fetch_aggregate_trades(
        symbol="BTCUSDT",
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        market_base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )

    assert rows == [row]
    assert attempts == 2
    assert sleeps == [2.0]


def test_normalize_live_aggregate_trades_writes_partial_manifest(tmp_path: Path) -> None:
    rows = [
        {"a": 10, "p": "100.0", "q": "2.0", "f": 20, "l": 21, "T": 1767225600000, "m": True},
        {"a": 11, "p": "101.0", "q": "1.5", "f": 22, "l": 22, "T": 1767225601000, "m": False},
    ]
    raw_path = tmp_path / "raw" / "rows.jsonl.gz"
    write_raw_rows(raw_path, rows)
    normalized_path = tmp_path / "normalized" / "aggTrades" / "date=2026-01-01" / "events.parquet"

    manifest = normalize_live_aggregate_trades(
        rows=rows,
        normalized_path=normalized_path,
        raw_path=raw_path,
        source_url="https://example.test/fapi/v1/aggTrades",
        symbol="BTCUSDT",
        utc_date=date(2026, 1, 1),
    )

    normalized = pd.read_parquet(normalized_path)
    assert normalized["sequence_id"].tolist() == [10, 11]
    assert normalized["quote_quantity"].tolist() == [200.0, 151.5]
    assert manifest.quality_status == "DIAGNOSTIC_LIVE_PARTIAL_VALID"
    assert manifest.row_count == 2
    assert normalized_path.with_suffix(".manifest.json").exists()


@pytest.mark.asyncio
async def test_fetch_and_normalize_refuses_to_replace_archive_verified_partition(tmp_path: Path) -> None:
    rows = [{"a": 10, "p": "100.0", "q": "2.0", "f": 20, "l": 21, "T": 1767225600000, "m": True}]
    raw_path = tmp_path / "raw" / "rows.jsonl.gz"
    write_raw_rows(raw_path, rows)
    normalized_path = tmp_path / "normalized" / "aggTrades" / "date=2026-01-01" / "events.parquet"
    normalize_live_aggregate_trades(
        rows=rows,
        normalized_path=normalized_path,
        raw_path=raw_path,
        source_url="https://example.test/fapi/v1/aggTrades",
        symbol="BTCUSDT",
        utc_date=date(2026, 1, 1),
    )
    manifest_path = normalized_path.with_suffix(".manifest.json")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace("DIAGNOSTIC_LIVE_PARTIAL_VALID", "VALID"),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="non-live-partial"):
        await fetch_and_normalize_live_aggtrades_partition(
            symbol="BTCUSDT",
            utc_date=date(2026, 1, 1),
            start_at=datetime(2026, 1, 1, tzinfo=UTC),
            end_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            root=tmp_path,
            market_base_url="https://example.test",
            replace_partial=True,
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=rows)),
        )

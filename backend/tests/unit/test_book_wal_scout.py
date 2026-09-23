"""Short prospective WAL scout materializes causal book features safely."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_book_wal_scout import run_book_wal_scout


def test_book_wal_scout_materializes_fresh_book_rows(tmp_path: Path) -> None:
    wal_root = tmp_path / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(80):
        event_time = start + timedelta(seconds=second)
        price = 100 + second * 0.01
        _append_gzip_jsonl(
            wal_root / "trade" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "product": "usdm_perpetual",
                "sequence_id": second + 1,
                "price": price,
                "quantity": 1.0,
                "is_buyer_maker": second % 2 == 0,
            },
        )
        _append_gzip_jsonl(
            wal_root / "depth" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "sequence_id": second + 1,
                "bid_price": price - 0.5,
                "bid_quantity": 2.0,
                "ask_price": price + 0.5,
                "ask_quantity": 3.0,
            },
        )
        _append_gzip_jsonl(
            wal_root / "mark_price" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "price": price,
                "payload": {"index_price": price - 0.1, "funding_rate": 0.0001},
            },
        )
        _append_gzip_jsonl(
            wal_root / "spot_trade" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "product": "spot",
                "sequence_id": second + 1,
                "price": price - 0.2,
                "quantity": 1.0,
                "is_buyer_maker": second % 2 == 1,
            },
        )

    report = run_book_wal_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(20,),
        decision_stride_seconds=5,
        minimum_decisions_for_modeling=1,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["status"] == "READY_FOR_SCOUT_MODELING"
    assert report["decision_count"] > 0
    assert report["summary"]["book_available_min"] == 1.0
    assert report["summary"]["has_mark_features"] is True
    assert report["summary"]["has_spot_perp_features"] is True
    assert (tmp_path / "out" / "research_rows.parquet").exists()


def test_book_wal_scout_fails_closed_when_capture_is_too_short(tmp_path: Path) -> None:
    wal_root = tmp_path / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(10):
        event_time = start + timedelta(seconds=second)
        _append_gzip_jsonl(
            wal_root / "trade" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "product": "usdm_perpetual",
                "sequence_id": second + 1,
                "price": 100.0,
                "quantity": 1.0,
                "is_buyer_maker": False,
            },
        )
        _append_gzip_jsonl(
            wal_root / "depth" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "sequence_id": second + 1,
                "bid_price": 99.5,
                "bid_quantity": 2.0,
                "ask_price": 100.5,
                "ask_quantity": 3.0,
            },
        )

    report = run_book_wal_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(30,),
    )

    assert report["status"] == "INSUFFICIENT_INTERVAL"
    assert report["modeling_ready"] is False
    assert report["decision_count"] == 0


def test_book_wal_scout_filters_stale_book_decisions(tmp_path: Path) -> None:
    wal_root = tmp_path / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(80):
        event_time = start + timedelta(seconds=second)
        price = 100 + second * 0.01
        _append_gzip_jsonl(
            wal_root / "trade" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "product": "usdm_perpetual",
                "sequence_id": second + 1,
                "price": price,
                "quantity": 1.0,
                "is_buyer_maker": False,
            },
        )
        if second <= 35 or second >= 45:
            _append_gzip_jsonl(
                wal_root / "depth" / "date=2026-01-01" / "events.jsonl.gz",
                {
                    "event_time": event_time.isoformat(),
                    "sequence_id": second + 1,
                    "bid_price": price - 0.5,
                    "bid_quantity": 2.0,
                    "ask_price": price + 0.5,
                    "ask_quantity": 3.0,
                },
            )

    report = run_book_wal_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(20,),
        decision_stride_seconds=5,
        max_book_staleness_ms=100,
        minimum_decisions_for_modeling=1,
    )

    assert report["status"] == "READY_FOR_SCOUT_MODELING"
    assert report["decision_count"] == 5
    assert report["book_fresh_filter"]["applied"] is True
    assert report["book_fresh_filter"]["dropped_decision_count"] == 1


def test_book_wal_scout_filters_decisions_without_trade_path(tmp_path: Path) -> None:
    wal_root = tmp_path / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(70):
        event_time = start + timedelta(seconds=second)
        price = 100 + second * 0.01
        if not 31 <= second <= 40:
            _append_gzip_jsonl(
                wal_root / "trade" / "date=2026-01-01" / "events.jsonl.gz",
                {
                    "event_time": event_time.isoformat(),
                    "product": "usdm_perpetual",
                    "sequence_id": second + 1,
                    "price": price,
                    "quantity": 1.0,
                    "is_buyer_maker": False,
                },
            )
        _append_gzip_jsonl(
            wal_root / "depth" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "sequence_id": second + 1,
                "bid_price": price - 0.5,
                "bid_quantity": 2.0,
                "ask_price": price + 0.5,
                "ask_quantity": 3.0,
            },
        )

    report = run_book_wal_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(5,),
        decision_stride_seconds=5,
        max_book_staleness_ms=1_000,
        minimum_decisions_for_modeling=1,
    )

    assert report["status"] == "READY_FOR_SCOUT_MODELING"
    assert report["book_fresh_filter"]["path_filter_applied"] is True
    assert report["book_fresh_filter"]["book_staleness_filter_applied"] is False
    assert report["book_fresh_filter"]["dropped_decision_count"] > 0


def _append_gzip_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

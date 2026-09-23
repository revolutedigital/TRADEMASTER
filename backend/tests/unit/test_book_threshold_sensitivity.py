"""Threshold sensitivity diagnostics for prospective book WAL captures."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_book_threshold_sensitivity import (
    run_book_threshold_sensitivity,
)


def test_book_threshold_sensitivity_reports_reachable_targets(tmp_path: Path) -> None:
    wal_root = tmp_path / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(100):
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

    report = run_book_threshold_sensitivity(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(20,),
        thresholds_bps=(1.0, 5.0),
        decision_stride_seconds=5,
        minimum_decisions_for_modeling=1,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["status"] == "READY_FOR_SENSITIVITY_REVIEW"
    assert report["decision_count"] > 0
    low_threshold_buy = _side(report, threshold_bps=1.0, side="BUY")
    high_threshold_buy = _side(report, threshold_bps=5.0, side="BUY")
    assert low_threshold_buy["target_before_stop_rate"] >= high_threshold_buy[
        "target_before_stop_rate"
    ]
    assert report["best_touch_rate"] is not None
    assert report["best_with_target_touch_gross_path"] is not None
    assert report["best_unconditional_gross_path"] is not None


def test_book_threshold_sensitivity_fails_closed_when_capture_is_too_short(
    tmp_path: Path,
) -> None:
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

    report = run_book_threshold_sensitivity(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(30,),
        thresholds_bps=(1.0,),
    )

    assert report["status"] == "INSUFFICIENT_INTERVAL"
    assert report["modeling_ready"] is False
    assert report["decision_count"] == 0
    assert report["thresholds"] == []


def _side(report: dict[str, object], *, threshold_bps: float, side: str) -> dict[str, object]:
    thresholds = report["thresholds"]
    assert isinstance(thresholds, list)
    threshold = next(
        item for item in thresholds if item["threshold_bps"] == threshold_bps
    )
    by_side = threshold["by_side"]
    assert isinstance(by_side, list)
    return next(item for item in by_side if item["side"] == side)


def _append_gzip_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

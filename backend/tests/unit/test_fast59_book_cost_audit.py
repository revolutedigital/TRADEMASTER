import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_fast59_book_cost_audit import (
    WalRootRef,
    audit_depth_partition,
    discover_default_wal_roots,
    fast59_status,
    numeric_summary,
    run_fast59_book_cost_audit,
)


def test_numeric_summary_uses_interpolated_quantiles() -> None:
    summary = numeric_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["p50"] == 2.5
    assert summary["max"] == 4.0


def test_book_cost_audit_marks_clean_depth_sample_eligible(tmp_path: Path) -> None:
    wal_root = _write_depth_wal(tmp_path / "prospective-wal-fast-test", spread_ticks=1)
    depth_path = wal_root / "depth" / "date=2026-01-01" / "events.jsonl.gz"

    partition = audit_depth_partition(
        label="test",
        wal_root=wal_root,
        depth_path=depth_path,
        utc_date=datetime(2026, 1, 1, tzinfo=UTC).date(),
        max_roundtrip_book_cost_bps=2.0,
        max_p95_receive_delay_ms=1_000.0,
        min_p05_top_notional_usdt=10_000.0,
        min_depth_events=5,
    )

    assert partition["eligible_sample"] is True
    assert partition["row_count"] == 10
    assert partition["roundtrip_book_cost_bps"]["p95"] < 2.0


def test_book_cost_audit_fails_wide_spread_sample(tmp_path: Path) -> None:
    wal_root = _write_depth_wal(tmp_path / "prospective-wal-fast-test", spread_ticks=500)
    report = run_fast59_book_cost_audit(
        wal_roots=(WalRootRef("test", wal_root),),
        dates=None,
        output_root=tmp_path / "out",
        max_roundtrip_book_cost_bps=2.0,
        max_p95_receive_delay_ms=1_000.0,
        min_p05_top_notional_usdt=10_000.0,
        min_depth_events=5,
    )

    assert report["status"] == "FAST59_BOOK_COST_SAMPLE_FAILED"
    assert report["approved_strategy"] is False
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"


def test_default_discovery_only_returns_fast_wal_roots_with_depth(tmp_path: Path) -> None:
    _write_depth_wal(tmp_path / "prospective-wal-fast-a", spread_ticks=1)
    _write_depth_wal(tmp_path / "prospective-wal-smoke", spread_ticks=1)
    (tmp_path / "prospective-wal-fast-empty").mkdir()

    discovered = discover_default_wal_roots(tmp_path)

    assert [root.label for root in discovered] == ["prospective-wal-fast-a"]


def test_fast59_status_reports_partial_samples() -> None:
    status, reasons = fast59_status(
        {
            "partition_count": 2,
            "eligible_partition_count": 1,
            "unique_utc_dates": ["2026-01-01"],
            "max_p95_roundtrip_book_cost_bps": 1.0,
            "max_p95_receive_delay_ms": 100.0,
            "min_p05_top_notional_usdt": 20_000.0,
        }
    )

    assert status == "FAST59_BOOK_COST_SAMPLE_PARTIAL_RESEARCH_ONLY"
    assert "eligible_partitions:1/2" in reasons


def _write_depth_wal(root: Path, *, spread_ticks: int) -> Path:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    path = root / "depth" / "date=2026-01-01" / "events.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(10):
            bid = 100.0 + index * 0.01
            ask = bid + spread_ticks * 0.01
            event_time = start + timedelta(milliseconds=100 * index)
            receive_time = event_time + timedelta(milliseconds=50)
            handle.write(
                json.dumps(
                    {
                        "event_type": "DEPTH",
                        "product": "usdm_perpetual",
                        "symbol": "BTCUSDT",
                        "event_time": event_time.isoformat().replace("+00:00", "Z"),
                        "receive_time": receive_time.isoformat().replace("+00:00", "Z"),
                        "bid_price": bid,
                        "bid_quantity": 200.0,
                        "ask_price": ask,
                        "ask_quantity": 200.0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return root

"""Settle fixed-gate shadow decisions without execution side effects."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.settle_book_fixed_gate_shadow_decisions import (
    settle_book_fixed_gate_shadow_decisions,
)


BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def test_shadow_decision_settlement_replays_bid_ask_without_approval(
    tmp_path: Path,
) -> None:
    wal_root = tmp_path / "wal"
    _write_book_wal(wal_root)
    decisions_path = tmp_path / "shadow_decisions.jsonl"
    decisions_path.write_text(json.dumps(_decision()) + "\n", encoding="utf-8")
    shadow_report_path = tmp_path / "shadow_report.json"
    shadow_report_path.write_text(
        json.dumps(_shadow_report(decisions_path), sort_keys=True),
        encoding="utf-8",
    )

    report = settle_book_fixed_gate_shadow_decisions(
        shadow_report_path=shadow_report_path,
        wal_root=wal_root,
        output_root=tmp_path / "out",
        latency_ms=0,
        target_bps=5.0,
        initial_stop_bps=20.0,
        max_book_staleness_ms=1_000,
        minimum_executable_count=1,
        cost_levels_bps=(0.0, 0.5, 1.0),
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["committed"] is False
    assert report["approved_strategy"] is False
    assert report["status"] == "SHADOW_SETTLEMENT_READY_FOR_PROSPECTIVE_STREAM"
    assert report["independent_summary"]["executable_count"] == 1
    assert Path(report["settlements_path"]).exists()


def test_shadow_decision_settlement_rejects_committed_report(tmp_path: Path) -> None:
    decisions_path = tmp_path / "shadow_decisions.jsonl"
    decisions_path.write_text(json.dumps(_decision()) + "\n", encoding="utf-8")
    shadow_report = _shadow_report(decisions_path)
    shadow_report["committed"] = True
    shadow_report["dry_run"] = False
    shadow_report_path = tmp_path / "shadow_report.json"
    shadow_report_path.write_text(json.dumps(shadow_report), encoding="utf-8")

    try:
        settle_book_fixed_gate_shadow_decisions(
            shadow_report_path=shadow_report_path,
            wal_root=tmp_path / "wal",
            output_root=tmp_path / "out",
        )
    except ValueError as error:
        assert "dry-run report" in str(error)
    else:
        raise AssertionError("expected dry-run boundary validation to fail")


def _shadow_report(decisions_path: Path) -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "committed": False,
        "dry_run": True,
        "policy_sha256": "a" * 64,
        "selected_count": 1,
        "decisions_path": str(decisions_path),
    }


def _decision() -> dict[str, object]:
    decision_time_ms = int(BASE_TIME.timestamp() * 1000)
    return {
        "decision_time": BASE_TIME.isoformat(),
        "decision_time_ms": decision_time_ms,
        "side": "SELL",
        "horizon_seconds": 1,
        "would_enter": True,
        "policy_sha256": "a" * 64,
        "feature_vector_sha256": "b" * 64,
    }


def _write_book_wal(root: Path) -> None:
    for offset_ms, bid, ask in (
        (0, 99.99, 100.01),
        (100, 99.49, 99.51),
        (200, 99.39, 99.41),
    ):
        _append_gzip_jsonl(
            root / "depth" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": (BASE_TIME + timedelta(milliseconds=offset_ms)).isoformat(),
                "sequence_id": offset_ms + 1,
                "bid_price": bid,
                "bid_quantity": 1.0,
                "ask_price": ask,
                "ask_quantity": 1.0,
            },
        )


def _append_gzip_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

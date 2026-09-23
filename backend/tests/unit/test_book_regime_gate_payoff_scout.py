"""Nested regime gate payoff scout diagnostics."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_book_multi_window_model_scout import WalRootRef
from scripts.research.run_book_regime_gate_payoff_scout import (
    run_book_regime_gate_payoff_scout,
)


def test_regime_gate_payoff_scout_reports_without_approval(tmp_path: Path) -> None:
    wal_a = _write_directional_wal(tmp_path / "a", seconds=220)
    wal_b = _write_directional_wal(tmp_path / "b", seconds=225)
    wal_c = _write_directional_wal(tmp_path / "c", seconds=230)

    report = run_book_regime_gate_payoff_scout(
        wal_root_refs=(
            WalRootRef("a", wal_a),
            WalRootRef("b", wal_b),
            WalRootRef("c", wal_c),
        ),
        output_root=tmp_path / "out",
        model_name="hist_gradient_payoff",
        horizon_seconds=20,
        target_bps=1.0,
        trailing_distance_bps=2.0,
        top_fraction=0.25,
        gate_keep_fractions=(0.7, 1.0),
        decision_stride_seconds=5,
        max_book_staleness_ms=1_000,
        latency_ms=0,
        initial_stop_bps=20.0,
        cost_levels_bps=(0.0, 1.0),
        profit_cost_bps=1.0,
        stress_cost_bps=1.0,
        minimum_windows=3,
        minimum_executable_per_window=1,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["modeling_ready"] is True
    assert len(report["holdouts"]) == 3
    assert Path(report["report_path"]).exists()


def _write_directional_wal(root: Path, *, seconds: int) -> Path:
    wal_root = root / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(seconds):
        event_time = start + timedelta(seconds=second)
        trend = 1 if (second // 50) % 2 == 0 else -1
        leg_second = second % 50
        price = 100 + trend * leg_second * 0.02
        bid_quantity = 6.0 if trend > 0 else 1.0
        ask_quantity = 1.0 if trend > 0 else 6.0
        _append_gzip_jsonl(
            wal_root / "trade" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "product": "usdm_perpetual",
                "sequence_id": second + 1,
                "price": price,
                "quantity": 1.0,
                "is_buyer_maker": trend < 0,
            },
        )
        _append_gzip_jsonl(
            wal_root / "depth" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "sequence_id": second + 1,
                "bid_price": price - 0.005,
                "bid_quantity": bid_quantity,
                "ask_price": price + 0.005,
                "ask_quantity": ask_quantity,
            },
        )
    return wal_root


def _append_gzip_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

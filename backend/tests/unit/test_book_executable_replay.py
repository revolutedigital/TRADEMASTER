"""Executable top-of-book replay diagnostics for frozen book gates."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.event_replay import OrderSide
from scripts.research.run_book_executable_replay import (
    run_book_executable_replay,
    simulate_executable_trade,
    summarize_executable_trades,
)


def test_simulate_executable_sell_uses_bid_entry_and_ask_exit_target() -> None:
    book = {
        "event_time_ms": np.array([0, 100, 200, 300], dtype=np.int64),
        "bid_price": np.array([99.99, 99.49, 99.39, 99.29], dtype=np.float64),
        "ask_price": np.array([100.01, 99.51, 99.41, 99.31], dtype=np.float64),
    }

    trade = simulate_executable_trade(
        book=book,
        decision_time_ms=0,
        side=OrderSide.SELL,
        horizon_seconds=1,
        target_bps=5.0,
        initial_stop_bps=20.0,
        latency_ms=0,
        max_book_staleness_ms=1_000,
    )

    assert trade["would_enter"] is True
    assert trade["entry_price"] == 99.99
    assert trade["exit_reason"] == "TARGET"
    assert trade["gross_executable_bps"] == pytest.approx(5.0)


def test_summarize_executable_trades_reports_explicit_cost_sensitivity() -> None:
    trades = pd.DataFrame(
        [
            {"gross_executable_bps": 2.0, "exit_reason": "TARGET"},
            {"gross_executable_bps": -1.0, "exit_reason": "TIMEOUT"},
        ]
    )

    summary = summarize_executable_trades(
        trades,
        selected_signal_count=3,
        cost_levels_bps=(0.0, 0.5, 1.0),
    )

    assert summary["executable_count"] == 2
    assert summary["skipped_count"] == 1
    assert summary["gross_executable_mean_bps"] == 0.5
    assert summary["cost_sensitivity"][1]["net_executable_mean_bps"] == 0.0


def test_book_executable_replay_reports_without_approval(tmp_path: Path) -> None:
    base_wal = _write_sell_friendly_wal(tmp_path / "base", seconds=220)
    gate_wal = _write_sell_friendly_wal(tmp_path / "gate", seconds=240)
    confirmation_wal = _write_sell_friendly_wal(tmp_path / "confirmation", seconds=260)

    report = run_book_executable_replay(
        base_development_wal_root=base_wal,
        gate_development_wal_roots=(gate_wal,),
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "out",
        horizon_seconds=20,
        threshold_bps=1.0,
        base_condition_fraction=0.20,
        base_conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
        gate_feature="spot_update_age_ms",
        gate_direction="low",
        gate_fraction=0.50,
        decision_stride_seconds=5,
        max_book_staleness_ms=1_000,
        latency_ms=0,
        initial_stop_bps=20.0,
        minimum_executable_count=1,
        cost_levels_bps=(0.0, 0.5),
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["selected_signal_count"] >= 1
    assert report["independent_summary"]["executable_count"] >= 1
    assert Path(report["trades_path"]).exists()


def _write_sell_friendly_wal(root: Path, *, seconds: int) -> Path:
    wal_root = root / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(seconds):
        event_time = start + timedelta(seconds=second)
        price = 100 - second * 0.002
        _append_gzip_jsonl(
            wal_root / "trade" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "product": "usdm_perpetual",
                "sequence_id": second + 1,
                "price": price,
                "quantity": 1.0,
                "is_buyer_maker": True,
            },
        )
        _append_gzip_jsonl(
            wal_root / "depth" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "sequence_id": second + 1,
                "bid_price": price - 0.005,
                "bid_quantity": 1.0,
                "ask_price": price + 0.005,
                "ask_quantity": 6.0,
            },
        )
        _append_gzip_jsonl(
            wal_root / "spot_trade" / "date=2026-01-01" / "events.jsonl.gz",
            {
                "event_time": event_time.isoformat(),
                "product": "spot",
                "sequence_id": second + 1,
                "price": price + 0.02,
                "quantity": 1.0,
                "is_buyer_maker": True,
            },
        )
    return wal_root


def _append_gzip_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

"""Multi-window model scout with executable trailing-stop replay."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.event_replay import OrderSide
from scripts.research.run_book_multi_window_model_scout import (
    WalRootRef,
    _apply_one_position_constraint_by_score,
    run_book_multi_window_model_scout,
    simulate_executable_trailing_trade,
)


def test_simulate_executable_trailing_trade_moves_stop_after_favorable_move() -> None:
    book = {
        "event_time_ms": np.array([0, 100, 200, 300, 400], dtype=np.int64),
        "bid_price": np.array([99.99, 100.06, 100.12, 100.06, 100.05], dtype=np.float64),
        "ask_price": np.array([100.00, 100.07, 100.13, 100.07, 100.06], dtype=np.float64),
    }

    trade = simulate_executable_trailing_trade(
        book=book,
        decision_time_ms=0,
        side=OrderSide.BUY,
        horizon_seconds=1,
        breakeven_trigger_bps=5.0,
        trailing_activation_bps=5.0,
        trailing_distance_bps=5.0,
        initial_stop_bps=20.0,
        latency_ms=0,
        max_book_staleness_ms=1_000,
    )

    assert trade["would_enter"] is True
    assert trade["exit_reason"] == "TRAILING_STOP"
    assert trade["gross_executable_bps"] == pytest.approx(7.0, abs=0.2)


def test_multi_window_model_scout_reports_candidates_without_approval(
    tmp_path: Path,
) -> None:
    wal_a = _write_directional_wal(tmp_path / "a", seconds=170)
    wal_b = _write_directional_wal(tmp_path / "b", seconds=175)

    report = run_book_multi_window_model_scout(
        wal_root_refs=(WalRootRef("a", wal_a), WalRootRef("b", wal_b)),
        output_root=tmp_path / "out",
        horizon_seconds=20,
        target_bps_values=(1.0,),
        top_fractions=(0.50,),
        trailing_distance_bps_values=(2.0,),
        decision_stride_seconds=5,
        max_book_staleness_ms=1_000,
        latency_ms=0,
        initial_stop_bps=20.0,
        cost_levels_bps=(0.0, 1.0),
        profit_cost_bps=1.0,
        stress_cost_bps=1.0,
        minimum_windows=2,
        minimum_executable_per_window=1,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["candidate_count"] > 0
    assert report["best_candidate"] is not None
    assert Path(report["report_path"]).exists()


def test_apply_one_position_constraint_by_score_keeps_non_overlapping_rows() -> None:
    trades = pd.DataFrame(
        {
            "decision_time_ms": [0, 10_000, 70_000],
            "exit_time_ms": [60_000, 65_000, 120_000],
            "model_score": [3.0, 5.0, 4.0],
            "gross_executable_bps": [1.0, 2.0, 3.0],
            "custom_column": ["first", "overlap", "later"],
        },
        index=[10, 20, 30],
    )

    selected = _apply_one_position_constraint_by_score(trades)

    assert selected["decision_time_ms"].tolist() == [0, 70_000]
    assert selected["custom_column"].tolist() == ["first", "later"]
    assert list(selected.columns) == list(trades.columns)


def _write_directional_wal(root: Path, *, seconds: int) -> Path:
    wal_root = root / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(seconds):
        event_time = start + timedelta(seconds=second)
        trend = 1 if (second // 35) % 2 == 0 else -1
        leg_second = second % 35
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

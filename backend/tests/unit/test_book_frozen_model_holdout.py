"""Frozen model holdout evaluation remains research-only."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_book_frozen_model_holdout import (
    _status,
    run_book_frozen_model_holdout,
)
from scripts.research.run_book_multi_window_model_scout import WalRootRef


def test_frozen_model_holdout_reports_without_strategy_approval(tmp_path: Path) -> None:
    train_a = _write_directional_wal(tmp_path / "train-a", seconds=190)
    train_b = _write_directional_wal(tmp_path / "train-b", seconds=195)
    holdout = _write_directional_wal(tmp_path / "holdout", seconds=200)

    report = run_book_frozen_model_holdout(
        training_wal_root_refs=(WalRootRef("train_a", train_a), WalRootRef("train_b", train_b)),
        holdout_wal_root_ref=WalRootRef("holdout", holdout),
        output_root=tmp_path / "out",
        model_name="hist_gradient",
        horizon_seconds=20,
        target_bps=1.0,
        top_fraction=0.20,
        trailing_distance_bps=2.0,
        decision_stride_seconds=5,
        max_book_staleness_ms=1_000,
        latency_ms=0,
        initial_stop_bps=20.0,
        cost_levels_bps=(0.0, 1.0),
        profit_cost_bps=1.0,
        stress_cost_bps=1.0,
        minimum_executable_count=1,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["modeling_ready"] is True
    assert report["feature_count"] > 0
    assert report["selected_signal_count"] > 0
    assert Path(report["selected_rows_path"]).exists()
    assert Path(report["trades_path"]).exists()
    assert Path(report["one_position_trades_path"]).exists()


def test_status_can_gate_only_one_position_mode() -> None:
    independent_summary = _summary(executable_count=12, net_at_1bps=-0.5, net_at_2bps=-1.5)
    one_position_summary = _summary(executable_count=6, net_at_1bps=2.0, net_at_2bps=1.0)

    status, reasons = _status(
        independent_summary=independent_summary,
        one_position_summary=one_position_summary,
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
        minimum_executable_count=5,
        required_positive_modes=("one_position",),
    )

    assert status == "FROZEN_MODEL_HOLDOUT_PASSED_RESEARCH_ONLY"
    assert "additional_prospective_windows_required" in reasons


def _write_directional_wal(root: Path, *, seconds: int) -> Path:
    wal_root = root / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(seconds):
        event_time = start + timedelta(seconds=second)
        trend = 1 if (second // 40) % 2 == 0 else -1
        leg_second = second % 40
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


def _summary(
    *,
    executable_count: int,
    net_at_1bps: float,
    net_at_2bps: float,
) -> dict[str, object]:
    gross_mean = net_at_1bps + 1.0
    return {
        "executable_count": executable_count,
        "cost_sensitivity": [
            {
                "explicit_cost_bps": 1.0,
                "net_executable_mean_bps": net_at_1bps,
                "net_positive": net_at_1bps > 0,
            },
            {
                "explicit_cost_bps": 2.0,
                "net_executable_mean_bps": net_at_2bps,
                "net_positive": net_at_2bps > 0,
            },
        ],
        "gross_executable_mean_bps": gross_mean,
    }

"""Frozen book-rule confirmation tests."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_book_rule_confirmation import (
    apply_frozen_rule,
    fit_rule_thresholds,
    run_book_rule_confirmation,
)


def test_fit_and_apply_frozen_rule_uses_development_threshold() -> None:
    import pandas as pd

    development = pd.DataFrame({"signal": [1, 2, 3, 4, 5]})
    confirmation = pd.DataFrame({"signal": [1, 4, 5, 6]})

    thresholds = fit_rule_thresholds(
        development,
        conditions=({"feature": "signal", "direction": "high"},),
        condition_fraction=0.20,
    )
    selected = apply_frozen_rule(confirmation, thresholds)

    assert thresholds[0]["threshold"] == 4.2
    assert selected["signal"].tolist() == [5, 6]


def test_book_rule_confirmation_reports_confirmation_without_approval(
    tmp_path: Path,
) -> None:
    development_wal = _write_directional_wal(tmp_path / "development", seconds=220)
    confirmation_wal = _write_directional_wal(tmp_path / "confirmation", seconds=240)

    report = run_book_rule_confirmation(
        development_wal_root=development_wal,
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "out",
        horizon_seconds=20,
        threshold_bps=1.0,
        condition_fraction=0.20,
        conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
        decision_stride_seconds=5,
        minimum_selected_count=2,
        cost_levels_bps=(0.0, 0.5, 1.0),
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["status"] == "CONFIRMATION_READY_FOR_REPLICATION"
    assert report["confirmation_metrics"]["selected_count"] >= 2
    cost_sensitivity = report["confirmation_metrics"]["cost_sensitivity"]
    assert [item["total_cost_bps"] for item in cost_sensitivity] == [0.0, 0.5, 1.0]
    assert cost_sensitivity[1]["selected_net_mean_bps"] == (
        report["confirmation_metrics"]["selected_gross_path_payoff_mean_bps"] - 0.5
    )


def test_book_rule_confirmation_fails_closed_when_confirmation_too_short(
    tmp_path: Path,
) -> None:
    development_wal = _write_directional_wal(tmp_path / "development", seconds=220)
    confirmation_wal = _write_directional_wal(tmp_path / "confirmation", seconds=12)

    report = run_book_rule_confirmation(
        development_wal_root=development_wal,
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "out",
        horizon_seconds=20,
        threshold_bps=1.0,
        condition_fraction=0.20,
        conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
    )

    assert report["status"] == "CONFIRMATION_INSUFFICIENT_INTERVAL"
    assert report["modeling_ready"] is False
    assert report["approved_strategy"] is False


def _write_directional_wal(root: Path, *, seconds: int) -> Path:
    wal_root = root / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(seconds):
        event_time = start + timedelta(seconds=second)
        trend = 1 if (second // 40) % 2 == 0 else -1
        price = 100 + trend * (second % 40) * 0.015
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
                "bid_price": price - 0.5,
                "bid_quantity": bid_quantity,
                "ask_price": price + 0.5,
                "ask_quantity": ask_quantity,
            },
        )
    return wal_root


def _append_gzip_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

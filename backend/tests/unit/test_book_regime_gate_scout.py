"""Regime gate scout tests."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from scripts.research.run_book_regime_gate_confirmation import (
    run_book_regime_gate_confirmation,
)
from scripts.research.run_book_regime_gate_scout import (
    apply_gate_threshold,
    fit_single_gate_threshold,
    run_book_regime_gate_scout,
)


def test_fit_and_apply_single_gate_threshold_uses_development_rows() -> None:
    development = pd.DataFrame({"regime": [1, 2, 3, 4, 5]})
    confirmation = pd.DataFrame({"regime": [1, 4, 5, 6]})

    gate = fit_single_gate_threshold(
        development,
        feature="regime",
        direction="high",
        fraction=0.20,
    )
    selected = apply_gate_threshold(confirmation, gate)

    assert gate["threshold"] == 4.2
    assert selected["regime"].tolist() == [5, 6]


def test_book_regime_gate_scout_reports_holdout_without_approval(
    tmp_path: Path,
) -> None:
    base_wal = _write_directional_wal(tmp_path / "base", seconds=220)
    gate_a_wal = _write_directional_wal(tmp_path / "gate_a", seconds=240)
    gate_b_wal = _write_directional_wal(tmp_path / "gate_b", seconds=240)
    confirmation_wal = _write_directional_wal(tmp_path / "confirmation", seconds=260)

    report = run_book_regime_gate_scout(
        base_development_wal_root=base_wal,
        gate_development_wal_roots=(gate_a_wal, gate_b_wal),
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "out",
        horizon_seconds=20,
        threshold_bps=1.0,
        base_condition_fraction=0.20,
        base_conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
        gate_fractions=(0.20,),
        decision_stride_seconds=5,
        minimum_gate_development_selected_count=1,
        minimum_confirmation_selected_count=1,
        max_report_candidates=5,
        cost_levels_bps=(0.0, 0.5),
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["status"] == "READY_FOR_REGIME_GATE_REVIEW"
    assert report["selected_gate"] is not None
    assert report["confirmation_base_rule_metrics"]["selected_count"] >= 1


def test_book_regime_gate_scout_selects_gate_before_confirmation_count_gate(
    tmp_path: Path,
) -> None:
    base_wal = _write_directional_wal(tmp_path / "base", seconds=220)
    gate_wal = _write_directional_wal(tmp_path / "gate", seconds=240)
    confirmation_wal = _write_directional_wal(tmp_path / "confirmation", seconds=260)

    report = run_book_regime_gate_scout(
        base_development_wal_root=base_wal,
        gate_development_wal_roots=(gate_wal,),
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "out",
        horizon_seconds=20,
        threshold_bps=1.0,
        base_condition_fraction=0.20,
        base_conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
        gate_fractions=(0.20,),
        decision_stride_seconds=5,
        minimum_gate_development_selected_count=1,
        minimum_confirmation_selected_count=999,
    )

    assert report["selected_gate"] is not None
    assert report["status"] == "CONFIRMATION_INSUFFICIENT_SELECTIONS"
    assert report["modeling_ready"] is False
    assert report["leakage_guard"][
        "confirmation_minimum_count_checked_after_selection"
    ] is True


def test_book_regime_gate_confirmation_freezes_requested_gate(
    tmp_path: Path,
) -> None:
    base_wal = _write_directional_wal(tmp_path / "base", seconds=220)
    gate_wal = _write_directional_wal(tmp_path / "gate", seconds=240)
    confirmation_wal = _write_directional_wal(tmp_path / "confirmation", seconds=260)
    scout = run_book_regime_gate_scout(
        base_development_wal_root=base_wal,
        gate_development_wal_roots=(gate_wal,),
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "scout",
        horizon_seconds=20,
        threshold_bps=1.0,
        base_condition_fraction=0.20,
        base_conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
        gate_fractions=(0.20,),
        decision_stride_seconds=5,
        minimum_gate_development_selected_count=1,
        minimum_confirmation_selected_count=1,
    )
    gate = scout["selected_gate"]["gate"]

    report = run_book_regime_gate_confirmation(
        base_development_wal_root=base_wal,
        gate_development_wal_roots=(gate_wal,),
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "confirmation",
        horizon_seconds=20,
        threshold_bps=1.0,
        base_condition_fraction=0.20,
        base_conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
        gate_feature=gate["feature"],
        gate_direction=gate["direction"],
        gate_fraction=gate["fraction"],
        decision_stride_seconds=5,
        minimum_confirmation_selected_count=1,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["frozen_gate"]["feature"] == gate["feature"]
    assert report["confirmation_gated_metrics"]["selected_count"] >= 1


def test_book_regime_gate_scout_fails_closed_without_confirmation_rows(
    tmp_path: Path,
) -> None:
    base_wal = _write_directional_wal(tmp_path / "base", seconds=220)
    gate_wal = _write_directional_wal(tmp_path / "gate", seconds=240)
    confirmation_wal = _write_directional_wal(tmp_path / "confirmation", seconds=12)

    report = run_book_regime_gate_scout(
        base_development_wal_root=base_wal,
        gate_development_wal_roots=(gate_wal,),
        confirmation_wal_root=confirmation_wal,
        output_root=tmp_path / "out",
        horizon_seconds=20,
        threshold_bps=1.0,
        base_condition_fraction=0.20,
        base_conditions=({"feature": "directed_depth_imbalance", "direction": "high"},),
        gate_fractions=(0.20,),
    )

    assert report["status"] == "CONFIRMATION_INSUFFICIENT_INTERVAL"
    assert report["modeling_ready"] is False


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

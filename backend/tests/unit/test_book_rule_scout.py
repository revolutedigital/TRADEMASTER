"""Causal rule scout tests."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_book_rule_scout import (
    apply_train_fit_rule,
    run_book_rule_scout,
    select_rule_feature_columns,
)
from scripts.research.run_book_wal_scout import summarize_stream
from app.services.research.research_dataset import (
    ResearchDatasetConfig,
    build_research_rows,
    load_book_interval,
    load_trade_interval,
)


def test_rule_feature_columns_include_signals_not_future_labels(tmp_path: Path) -> None:
    wal_root = _write_directional_wal(tmp_path, seconds=140)
    stream = summarize_stream(wal_root / "trade")
    start = datetime.fromtimestamp(stream.first_event_time_ms / 1000, tz=UTC)
    end = datetime.fromtimestamp(stream.last_event_time_ms / 1000, tz=UTC)
    trades = load_trade_interval(wal_root / "trade", start, end)
    book = load_book_interval(wal_root / "depth", start, end)
    rows = build_research_rows(
        trades,
        decision_times_ms=trades["event_time_ms"].to_numpy()[30:100:5],
        config=ResearchDatasetConfig(
            horizons_seconds=(20,),
            expected_cost_bps=1.0,
            stress_cost_bps=1.0,
            feature_windows_seconds=(1, 5, 30),
            require_book_features=True,
        ),
        book_events=book,
    )

    feature_columns = set(select_rule_feature_columns(rows))

    assert "directed_depth_imbalance" in feature_columns
    assert "terminal_bps" not in feature_columns
    assert "mfe_bps" not in feature_columns
    assert "paid_expected_before_stop" not in feature_columns


def test_apply_train_fit_rule_uses_train_quantile_on_test() -> None:
    import pandas as pd

    train = pd.DataFrame({"signal": [1, 2, 3, 4, 5]})
    test = pd.DataFrame({"signal": [1, 4, 5, 6]})
    selected = apply_train_fit_rule(
        train,
        test,
        {
            "conditions": [{"feature": "signal", "direction": "high"}],
            "condition_fraction": 0.20,
        },
    )

    assert selected["signal"].tolist() == [5, 6]


def test_book_rule_scout_reports_candidates_without_approval(tmp_path: Path) -> None:
    wal_root = _write_directional_wal(tmp_path, seconds=220)

    report = run_book_rule_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(20,),
        thresholds_bps=(1.0,),
        condition_fractions=(0.20,),
        decision_stride_seconds=5,
        temporal_blocks=4,
        minimum_train_blocks=2,
        minimum_selected_count=2,
        max_pair_features=4,
        max_report_candidates=5,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["status"] == "READY_FOR_RULE_SCOUT_REVIEW"
    assert report["top_candidates"]
    assert report["best_candidate"] is not None


def test_book_rule_scout_fails_closed_when_capture_is_too_short(tmp_path: Path) -> None:
    wal_root = _write_directional_wal(tmp_path, seconds=12)

    report = run_book_rule_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(20,),
        thresholds_bps=(1.0,),
    )

    assert report["status"] == "INSUFFICIENT_INTERVAL"
    assert report["modeling_ready"] is False
    assert report["top_candidates"] == []


def _write_directional_wal(tmp_path: Path, *, seconds: int) -> Path:
    wal_root = tmp_path / "wal"
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

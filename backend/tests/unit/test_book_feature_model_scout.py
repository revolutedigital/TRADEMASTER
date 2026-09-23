"""Leak-safe model scout for prospective book WAL captures."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_book_feature_model_scout import (
    LABEL_AND_FUTURE_COLUMNS,
    run_book_feature_model_scout,
    select_causal_feature_columns,
)
from scripts.research.run_book_wal_scout import summarize_stream
from app.services.research.research_dataset import (
    ResearchDatasetConfig,
    build_research_rows,
    load_book_interval,
    load_trade_interval,
)


def test_select_causal_feature_columns_excludes_future_label_columns(
    tmp_path: Path,
) -> None:
    wal_root = _write_directional_wal(tmp_path, seconds=120)
    stream = summarize_stream(wal_root / "trade")
    start = datetime.fromtimestamp(stream.first_event_time_ms / 1000, tz=UTC)
    end = datetime.fromtimestamp(stream.last_event_time_ms / 1000, tz=UTC)
    trades = load_trade_interval(wal_root / "trade", start, end)
    book = load_book_interval(wal_root / "depth", start, end)
    rows = build_research_rows(
        trades,
        decision_times_ms=trades["event_time_ms"].to_numpy()[30:80:5],
        config=ResearchDatasetConfig(
            horizons_seconds=(20,),
            expected_cost_bps=1.0,
            stress_cost_bps=1.0,
            feature_windows_seconds=(1, 5, 30),
            require_book_features=True,
        ),
        book_events=book,
    )

    feature_columns = set(select_causal_feature_columns(rows))

    assert "mfe_bps" not in feature_columns
    assert "mae_bps" not in feature_columns
    assert "terminal_bps" not in feature_columns
    assert not feature_columns.intersection(LABEL_AND_FUTURE_COLUMNS)
    assert "directed_depth_imbalance" in feature_columns


def test_book_feature_model_scout_reports_candidates_without_approval(
    tmp_path: Path,
) -> None:
    wal_root = _write_directional_wal(tmp_path, seconds=180)

    report = run_book_feature_model_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(20,),
        thresholds_bps=(1.0,),
        top_fractions=(0.10,),
        decision_stride_seconds=5,
        temporal_blocks=4,
        minimum_train_blocks=2,
        minimum_selected_count=1,
    )

    assert report["research_only"] is True
    assert report["order_submission_allowed"] is False
    assert report["execution_authorization"] == "none"
    assert report["approved_strategy"] is False
    assert report["status"] == "READY_FOR_MODEL_SCOUT_REVIEW"
    assert report["feature_count"] > 0
    assert report["candidates"]
    assert report["best_candidate"] is not None


def test_book_feature_model_scout_fails_closed_when_capture_is_too_short(
    tmp_path: Path,
) -> None:
    wal_root = _write_directional_wal(tmp_path, seconds=15)

    report = run_book_feature_model_scout(
        wal_root=wal_root,
        output_root=tmp_path / "out",
        horizons_seconds=(20,),
        thresholds_bps=(1.0,),
    )

    assert report["status"] == "INSUFFICIENT_INTERVAL"
    assert report["modeling_ready"] is False
    assert report["approved_strategy"] is False
    assert report["candidates"] == []


def _write_directional_wal(tmp_path: Path, *, seconds: int) -> Path:
    wal_root = tmp_path / "wal"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for second in range(seconds):
        event_time = start + timedelta(seconds=second)
        trend = 1 if (second // 30) % 2 == 0 else -1
        price = 100 + trend * (second % 30) * 0.02
        bid_quantity = 5.0 if trend > 0 else 1.0
        ask_quantity = 1.0 if trend > 0 else 5.0
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

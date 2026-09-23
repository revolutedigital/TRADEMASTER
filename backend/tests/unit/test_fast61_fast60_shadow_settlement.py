import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_fast61_fast60_shadow_settlement import (
    fast61_status,
    run_fast61_fast60_shadow_settlement,
)


def test_fast61_settlement_marks_incomplete_when_horizon_is_not_covered(
    tmp_path: Path,
) -> None:
    wal_root = tmp_path / "wal-a"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _write_trade_partition(
        wal_root,
        start=start,
        offsets_and_prices=(
            (0, 100.0),
            (1_000, 100.2),
            (2_000, 100.3),
            (3_000, 100.4),
        ),
    )
    decisions_path = tmp_path / "gated_shadow_decisions.jsonl"
    decisions = [
        _decision("wal-a", start, horizon_seconds=2),
        _decision("wal-a", start + timedelta(milliseconds=2_500), horizon_seconds=2),
    ]
    _write_jsonl(decisions_path, decisions)
    report_path = tmp_path / "fast60-report.json"
    report_path.write_text(
        json.dumps(
            {
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
                "approved_strategy": False,
                "committed": False,
                "dry_run": True,
                "candidate": {
                    "model_sha256": "model",
                    "threshold": 0.9,
                    "tail_fraction": 0.01,
                    "horizon_seconds": 2,
                    "density_gate": {"lookback_seconds": 300, "limit": 0},
                },
                "runs": [{"label": "wal-a", "wal_root": str(wal_root)}],
                "gated_decisions_path": str(decisions_path),
            }
        ),
        encoding="utf-8",
    )

    report = run_fast61_fast60_shadow_settlement(
        fast60_report_path=report_path,
        output_root=tmp_path / "out",
        latency_ms=0,
        expected_cost_bps=0.0,
        stress_cost_bps=0.0,
        minimum_complete_outcomes=1,
    )

    assert report["aggregate"]["signal_count"] == 2
    assert report["aggregate"]["complete_outcome_count"] == 1
    assert report["aggregate"]["incomplete_outcome_count"] == 1
    assert report["status"] == (
        "FAST61_SHADOW_SETTLEMENT_POSITIVE_RESEARCH_ONLY_MORE_SAMPLE_REQUIRED"
    )
    incomplete = _read_jsonl(Path(str(report["incomplete_signals_path"])))
    assert incomplete[0]["incomplete_reason"] == "horizon_not_fully_covered_by_wal"
    assert incomplete[0]["flow_imbalance_5s"] == -0.3
    complete = _read_jsonl(Path(str(report["one_position_settlements_path"])))
    assert complete[0]["causal_feature_values"]["realized_vol_30s_bps"] == 50.0
    assert complete[0]["realized_vol_30s_bps"] == 50.0


def test_fast61_status_fails_when_stress_is_not_positive() -> None:
    status, reasons = fast61_status(
        aggregate={
            "signal_count": 10,
            "complete_outcome_count": 10,
            "incomplete_outcome_count": 0,
            "one_position_outcome_count": 10,
            "decision_day_count": 1,
            "label_count": 1,
            "complete_label_count": 1,
        },
        independent_summary={
            "trade_count": 10,
            "distinct_days": 1,
            "expected_mean_bps": 1.0,
            "stress_mean_bps": -0.1,
            "expected_total_log_return_bps": 10.0,
            "stress_total_log_return_bps": -1.0,
            "expected_win_rate": 0.9,
            "maximum_drawdown_bps": 1.0,
            "label_count": 1,
        },
        one_position_summary={
            "trade_count": 10,
            "distinct_days": 1,
            "expected_mean_bps": 1.0,
            "stress_mean_bps": 1.0,
            "expected_total_log_return_bps": 10.0,
            "stress_total_log_return_bps": 10.0,
            "expected_win_rate": 0.9,
            "maximum_drawdown_bps": 1.0,
            "label_count": 1,
        },
        minimum_complete_outcomes=10,
    )

    assert status == "FAST61_SHADOW_SETTLEMENT_FAILED"
    assert "stress_result_non_positive_after_trailing_replay" in reasons


def _decision(label: str, decision_time: datetime, *, horizon_seconds: int) -> dict[str, object]:
    return {
        "label": label,
        "decision_time": decision_time.isoformat(),
        "decision_time_ms": int(decision_time.timestamp() * 1000),
        "side": "BUY",
        "horizon_seconds": horizon_seconds,
        "probability": 0.95,
        "threshold": 0.9,
        "would_enter": True,
        "model_sha256": "model",
        "feature_vector_sha256": f"feature-{int(decision_time.timestamp() * 1000)}",
        "recent_signal_count": 0,
        "book_available": True,
        "book_age_ms": 10,
        "spread_bps": 0.01,
        "top_bid_notional_usdt": 100_000.0,
        "top_ask_notional_usdt": 100_000.0,
        "causal_feature_values": {
            "realized_vol_30s_bps": 50.0,
            "flow_imbalance_5s": -0.3,
        },
        "realized_vol_30s_bps": 50.0,
        "flow_imbalance_5s": -0.3,
    }


def _write_trade_partition(
    wal_root: Path,
    *,
    start: datetime,
    offsets_and_prices: tuple[tuple[int, float], ...],
) -> None:
    partition = wal_root / "trade" / f"date={start.date().isoformat()}"
    partition.mkdir(parents=True)
    with gzip.open(partition / "events.jsonl.gz", "wt", encoding="utf-8") as handle:
        for sequence_id, (offset_ms, price) in enumerate(offsets_and_prices):
            handle.write(
                json.dumps(
                    {
                        "event_time": (start + timedelta(milliseconds=offset_ms)).isoformat(),
                        "product": "usdm_perpetual",
                        "sequence_id": sequence_id,
                        "price": price,
                        "quantity": 1.0,
                        "is_buyer_maker": False,
                    }
                )
                + "\n"
            )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

from __future__ import annotations

from datetime import UTC, datetime

from scripts.research.run_fast185_live_partial_refresh_snapshot import build_fast185_snapshot_event


def test_build_fast185_snapshot_event_preserves_safety_flags_and_hashes(tmp_path) -> None:
    prospective_path = tmp_path / "prospective.jsonl"
    prospective_path.write_text(
        '{"decision_time":"2026-09-24T14:55:00+00:00","edge":8.0,'
        '"regime_gate_value":123.0,"fast185_decision_hour_utc":14}\n',
        encoding="utf-8",
    )
    event = build_fast185_snapshot_event(
        fast177_report={
            "report_sha256": "fast177-report",
            "artifacts": {
                "summary": {"sha256": "fast177-summary"},
                "prospective_diagnostic_rows": {"sha256": "fast177-rows"},
            },
        },
        fast185_report={
            "status": "FAST185_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
            "status_reasons": ["research_only"],
            "report_sha256": "fast185-report",
            "summary": {
                "prospective_row_count": 1,
                "eligible_observation_count": 1,
                "observation_summary": {
                    "decision_count": 1,
                    "day_count": 1,
                    "stress_net_total_bps": 2.0,
                    "stress_net_mean_bps": 2.0,
                    "positive_day_share": 0.0,
                },
            },
            "artifacts": {
                "summary": {"sha256": "fast185-summary"},
                "prospective_diagnostics": {"sha256": "fast185-prospective", "path": str(prospective_path)},
                "observations": {"sha256": "fast185-observations"},
            },
        },
        manifests={
            "BTCUSDT": {"normalized_sha256": "btc", "row_count": 10},
            "ETHUSDT": {"normalized_sha256": "eth", "row_count": 20},
            "SOLUSDT": {"normalized_sha256": "sol", "row_count": 30},
        },
        start_at=datetime(2026, 9, 24, 11, 20, tzinfo=UTC),
        end_at=datetime(2026, 9, 24, 15, 10, 59, 999000, tzinfo=UTC),
    )

    assert event["research_only"] is True
    assert event["approved_strategy"] is False
    assert event["order_submission_allowed"] is False
    assert event["orders_sent"] is False
    assert event["normalized_sha256_by_symbol"] == {
        "BTCUSDT": "btc",
        "ETHUSDT": "eth",
        "SOLUSDT": "sol",
    }
    assert event["prospective_latest_decision_time"] == "2026-09-24T14:55:00+00:00"
    assert event["report_sha256"] == "fast185-report"

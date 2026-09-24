from __future__ import annotations

import pandas as pd

from scripts.research.run_fast184_robust_session_filter_historical_audit import (
    POST_TRAINING_DATES,
    fast184_status,
    select_filter_rows,
    summarize_filter_records,
)


def test_select_filter_rows_applies_edge_regime_side_and_hour() -> None:
    rows = pd.DataFrame(
        [
            {
                "edge": 8.0,
                "ETHUSDT_qv_30m": 120_000_000.0,
                "side_mult": 1.0,
                "side_label": "LONG",
                "decision_hour_utc": 5,
            },
            {
                "edge": 8.0,
                "ETHUSDT_qv_30m": 120_000_000.0,
                "side_mult": -1.0,
                "side_label": "SHORT",
                "decision_hour_utc": 5,
            },
            {
                "edge": 6.0,
                "ETHUSDT_qv_30m": 120_000_000.0,
                "side_mult": 1.0,
                "side_label": "LONG",
                "decision_hour_utc": 5,
            },
            {
                "edge": 8.0,
                "ETHUSDT_qv_30m": 90_000_000.0,
                "side_mult": 1.0,
                "side_label": "LONG",
                "decision_hour_utc": 5,
            },
            {
                "edge": 8.0,
                "ETHUSDT_qv_30m": 120_000_000.0,
                "side_mult": 1.0,
                "side_label": "LONG",
                "decision_hour_utc": 17,
            },
        ]
    )

    selected = select_filter_rows(
        rows,
        {
            "edge_threshold_bps": 7.5,
            "side_filter": "LONG",
            "start_hour_utc": 5,
            "end_hour_utc": 16,
        },
    )

    assert len(selected) == 1
    assert selected.iloc[0]["side_label"] == "LONG"


def test_summarize_filter_records_reports_positive_days_and_worst_day() -> None:
    records = [
        {
            "candidate_id": "a",
            "role": "selected_survivor",
            "edge_threshold_bps": 7.5,
            "management_policy": "micro_5_2",
            "side_filter": "ALL",
            "start_hour_utc": 5,
            "end_hour_utc": 16,
            "date": "2026-09-15",
            "stress_net_bps": 3.0,
            "terminal_net_bps": 1.0,
            "exit_reason": "BREAKEVEN_STOP",
        },
        {
            "candidate_id": "a",
            "role": "selected_survivor",
            "edge_threshold_bps": 7.5,
            "management_policy": "micro_5_2",
            "side_filter": "ALL",
            "start_hour_utc": 5,
            "end_hour_utc": 16,
            "date": "2026-09-16",
            "stress_net_bps": -1.0,
            "terminal_net_bps": -2.0,
            "exit_reason": "INITIAL_STOP",
        },
    ]

    summary = summarize_filter_records(records)[0]

    assert summary["trade_count"] == 2
    assert summary["positive_day_count"] == 1
    assert summary["worst_day_stress_total_bps"] == -1.0


def test_fast184_status_promotes_historical_survivor_to_preregistration_only() -> None:
    selected = {
        "candidate_id": "edge7p5_micro5_all_utc05_16",
        "positive_day_count": len(POST_TRAINING_DATES),
        "worst_day_stress_total_bps": 1.0,
    }
    survivor = {
        "positive_day_count": len(POST_TRAINING_DATES),
    }

    status, reasons = fast184_status([selected], [survivor])

    assert status == "FAST184_ROBUST_SESSION_FILTER_HISTORICAL_SURVIVOR_REQUIRES_PREREGISTRATION"
    assert "fresh_future_preregistration_required" in reasons

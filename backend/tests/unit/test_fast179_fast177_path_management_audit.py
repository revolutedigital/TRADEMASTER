from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.services.backtest.event_replay import OrderSide
from scripts.research.run_fast179_fast177_path_management_audit import (
    compute_path_metrics,
    datetime_series_to_epoch_ms,
    fast179_status,
    first_touch_time,
    side_from_row,
    summarize_rows,
)


def test_compute_path_metrics_tracks_long_mfe_mae_and_touch_order() -> None:
    times = np.array([0, 100, 200, 300], dtype=np.int64)
    prices = np.array([100.0, 100.05, 99.94, 100.10], dtype=np.float64)

    metrics = compute_path_metrics(
        event_times_ms=times,
        prices=prices,
        decision_time_ms=0,
        side=OrderSide.BUY,
        horizon_seconds=1,
        latency_ms=0,
        stress_cost_bps=2.0,
    )

    assert metrics.entry_price == 100.0
    assert metrics.maximum_favorable_bps > 9.9
    assert metrics.maximum_adverse_bps < -5.9
    assert metrics.first_favorable_touch_ms == 100
    assert metrics.first_adverse_touch_ms == 200
    assert metrics.favorable_before_adverse is True
    assert math.isclose(metrics.terminal_stress_net_bps, metrics.terminal_gross_bps - 2.0)


def test_compute_path_metrics_tracks_short_direction_symmetrically() -> None:
    times = np.array([0, 100, 200], dtype=np.int64)
    prices = np.array([100.0, 99.90, 100.10], dtype=np.float64)

    metrics = compute_path_metrics(
        event_times_ms=times,
        prices=prices,
        decision_time_ms=0,
        side=OrderSide.SELL,
        horizon_seconds=1,
        latency_ms=0,
        stress_cost_bps=2.0,
    )

    assert metrics.maximum_favorable_bps > 9.9
    assert metrics.maximum_adverse_bps < -9.9
    assert metrics.first_favorable_touch_ms == 100
    assert metrics.first_adverse_touch_ms == 200


def test_first_touch_time_returns_none_when_threshold_never_touched() -> None:
    times = np.array([0, 100], dtype=np.int64)
    moves = np.array([0.0, 1.0], dtype=np.float64)

    assert first_touch_time(times, moves, threshold_bps=2.0, direction="gte") is None


def test_datetime_series_to_epoch_ms_preserves_millisecond_parquet_unit() -> None:
    series = pd.Series(
        pd.array(
            ["2026-09-24T11:20:00.095Z"],
            dtype=pd.DatetimeTZDtype(unit="ms", tz="UTC"),
        )
    )

    assert datetime_series_to_epoch_ms(series).tolist() == [1790248800095]


def test_side_from_row_rejects_unknown_side() -> None:
    assert side_from_row("LONG") == OrderSide.BUY
    assert side_from_row("SHORT") == OrderSide.SELL


def test_summarize_rows_reports_terminal_and_path_quality() -> None:
    rows = [
        {
            "eligible": True,
            "edge_gate_passed": True,
            "regime_gate_passed": True,
            "terminal_stress_net_bps": 3.0,
            "maximum_favorable_bps": 10.0,
            "maximum_adverse_bps": -1.0,
            "favorable_before_adverse": True,
        },
        {
            "eligible": False,
            "edge_gate_passed": False,
            "regime_gate_passed": True,
            "terminal_stress_net_bps": -5.0,
            "maximum_favorable_bps": 1.0,
            "maximum_adverse_bps": -7.0,
            "favorable_before_adverse": False,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["row_count"] == 2
    assert summary["eligible_count"] == 1
    assert summary["terminal_stress_net_total_bps"] == -2.0
    assert summary["terminal_stress_win_rate"] == 0.5
    assert summary["favorable_before_adverse_share"] == 0.5
    assert summary["mfe_threshold_shares"]["2bps"] == 0.5


def test_fast179_status_rejects_non_positive_best_policy() -> None:
    status, reasons = fast179_status(
        {
            "best_eligible_policy": {
                "row_count": 3,
                "stress_net_total_bps": -1.0,
            }
        }
    )

    assert status == "FAST179_PATH_MANAGEMENT_REJECTS_CURRENT_ELIGIBLE_ENTRIES"
    assert "best_eligible_policy_stress_total_non_positive" in reasons

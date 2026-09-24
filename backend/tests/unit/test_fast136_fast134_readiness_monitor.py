from __future__ import annotations

import pandas as pd

from scripts.research.run_fast136_fast134_readiness_monitor import (
    donchian_readiness_snapshot_from_daily_bars,
    fast136_status,
    select_post_preregistration_signals,
)


def test_fast136_status_waits_for_post_preregistration_m1_data() -> None:
    status, reasons = fast136_status(
        raw_m1_end=pd.Timestamp("2026-08-31T23:59:00Z"),
        post_preregistration_signal_count=0,
        completed_eligible_observation_count=0,
    )

    assert status == "FAST136_WAITING_FOR_POST_PREREGISTRATION_M1_DATA"
    assert "not_approved_strategy" in reasons
    assert "raw_m1_end_before_preregistration_timestamp" in reasons


def test_fast136_status_distinguishes_fresh_data_signal_and_completed_observation() -> None:
    fresh_status, fresh_reasons = fast136_status(
        raw_m1_end=pd.Timestamp("2026-10-01T00:00:00Z"),
        post_preregistration_signal_count=0,
        completed_eligible_observation_count=0,
    )
    signal_status, signal_reasons = fast136_status(
        raw_m1_end=pd.Timestamp("2026-10-01T00:00:00Z"),
        post_preregistration_signal_count=1,
        completed_eligible_observation_count=0,
    )
    ready_status, ready_reasons = fast136_status(
        raw_m1_end=pd.Timestamp("2026-10-30T00:00:00Z"),
        post_preregistration_signal_count=1,
        completed_eligible_observation_count=1,
    )

    assert fresh_status == "FAST136_DATA_FRESH_NO_POST_PREREGISTRATION_SIGNAL_YET"
    assert "no_frozen_signal_after_preregistration" in fresh_reasons
    assert signal_status == "FAST136_POST_PREREGISTRATION_SIGNAL_WAITING_FOR_FULL_OUTCOME"
    assert "completed_outcome_not_materialized_yet" in signal_reasons
    assert ready_status == "FAST136_ELIGIBLE_OBSERVATION_READY_FOR_FAST134_REVIEW"
    assert "rerun_fast134_required" in ready_reasons


def test_select_post_preregistration_signals_is_strictly_after_frozen_timestamp() -> None:
    signals = pd.DataFrame(
        [
            {
                "signal_timestamp": pd.Timestamp("2026-09-24T07:03:40Z"),
                "entry_timestamp": pd.Timestamp("2026-09-24T07:04:40Z"),
            },
            {
                "signal_timestamp": pd.Timestamp("2026-09-24T07:03:41Z"),
                "entry_timestamp": pd.Timestamp("2026-09-24T07:04:41Z"),
            },
        ]
    )

    selected = select_post_preregistration_signals(signals)

    assert len(selected) == 1
    assert selected.iloc[0]["entry_timestamp"] == pd.Timestamp("2026-09-24T07:04:41Z")


def test_donchian_readiness_snapshot_reports_active_upper_breakout_short_fade() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-09-21T23:59:00Z",
            "2026-09-22T23:59:00Z",
            "2026-09-23T23:59:00Z",
            "2026-09-24T23:59:00Z",
        ]
    )
    daily_bars = pd.DataFrame(
        {
            "open": [0.5950, 0.5960, 0.5970, 0.6010],
            "high": [0.6000, 0.6010, 0.6020, 0.6040],
            "low": [0.5900, 0.5910, 0.5920, 0.6000],
            "close": [0.5960, 0.5970, 0.5980, 0.6035],
        },
        index=index,
    )
    atr = pd.Series([25.0], index=[index[-1]])

    snapshot = donchian_readiness_snapshot_from_daily_bars(
        daily_bars,
        pair="NZDUSD",
        channel_days=3,
        atr_pips=atr,
    )

    assert snapshot["available"] is True
    assert snapshot["is_breakout_signal_now"] is True
    assert snapshot["current_breakout_direction"] == "UP"
    assert snapshot["current_fade_side"] == "SHORT"
    assert snapshot["nearest_trigger"] == "active_upper_breakout"
    assert snapshot["risk_pips_if_signal"] == 50.0
    assert snapshot["post_preregistration_signal_bar"] is True

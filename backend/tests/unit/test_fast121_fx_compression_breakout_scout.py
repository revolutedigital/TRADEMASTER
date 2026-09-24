"""Fast121 compression-breakout scout tests."""

import pandas as pd
import pytest

from scripts.research.run_fast121_fx_compression_breakout_scout import (
    Management,
    Policy,
    aggregate_mid_bars,
    fast121_status,
    policy_grid,
    signals_for_policy,
    true_range_pips,
)


def _bars(index: pd.DatetimeIndex, close_values: list[float]) -> pd.DataFrame:
    rows = []
    for value in close_values:
        rows.append(
            {
                "bid_open": value - 0.0001,
                "bid_high": value + 0.0004,
                "bid_low": value - 0.0004,
                "bid_close": value - 0.0001,
                "ask_open": value + 0.0001,
                "ask_high": value + 0.0006,
                "ask_low": value - 0.0002,
                "ask_close": value + 0.0001,
            }
        )
    return pd.DataFrame(rows, index=index)


def test_policy_grid_covers_h1_h4_and_managed_exits() -> None:
    grid = policy_grid()

    assert len(grid) == 96
    assert grid[0].identifier == "tf1h_ch20_cq0.25_atr1_hold720_fixed_2r"
    assert grid[-1].identifier == "tf4h_ch55_cq0.4_atr1.5_hold4320_trail_1r_1r"


def test_aggregate_mid_bars_labels_by_close_timestamp() -> None:
    index = pd.date_range("2026-01-01 00:00", periods=120, freq="min", tz="UTC")
    frame = _bars(index, [1.10] * len(index))

    bars = aggregate_mid_bars(frame, "1h")

    assert list(bars.index) == [
        pd.Timestamp("2026-01-01 00:59", tz="UTC"),
        pd.Timestamp("2026-01-01 01:59", tz="UTC"),
    ]
    assert bars.iloc[0]["close"] == 1.10


def test_true_range_uses_prior_close_and_pip_units() -> None:
    bars = pd.DataFrame(
        {
            "open": [1.0000, 1.0100],
            "high": [1.0020, 1.0200],
            "low": [0.9990, 1.0090],
            "close": [1.0010, 1.0110],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
    )

    ranges = true_range_pips(bars, 0.0001)

    assert ranges.iloc[0] == pytest.approx(30)
    assert ranges.iloc[1] == pytest.approx(190)


def test_signals_for_policy_emits_breakout_after_causal_compression() -> None:
    index = pd.date_range("2026-01-01 00:00", periods=700 * 60, freq="min", tz="UTC")
    closes = [1.1000] * len(index)
    for offset in range(650 * 60, len(index)):
        closes[offset] = 1.1100
    frame = _bars(index, closes)
    policy = Policy(
        "test",
        timeframe="1h",
        channel_bars=20,
        compression_quantile=0.4,
        stop_atr_multiple=1.0,
        max_hold_minutes=720,
        management=Management("fixed_2r", target_r=2.0, activation_r=None, trail_r=None),
    )

    signals = signals_for_policy(frame, "EURUSD", policy)

    assert signals
    assert signals[0].side == 1
    assert signals[0].entry_timestamp == signals[0].signal_timestamp + pd.Timedelta(minutes=1)


def test_fast121_status_never_approves_strategy() -> None:
    status, reasons = fast121_status(passed_count=1, finalists_count=2)

    assert status == "FAST121_FX_COMPRESSION_BREAKOUT_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

from __future__ import annotations

import pandas as pd

from scripts.research.run_fast128_fx_daily_donchian_scout import (
    Policy,
    breakout_signals,
    daily_mid_bars,
    fast128_status,
    policy_grid,
    time_exit_trades,
)


def _minute_frame(days: int = 80) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=days * 1440, freq="1min")
    values = 100.0 + pd.Series(range(len(index)), dtype="float64").to_numpy() * 0.0001
    return pd.DataFrame(
        {
            "bid_open": values,
            "bid_high": values,
            "bid_low": values,
            "bid_close": values,
            "ask_open": values + 0.01,
            "ask_high": values + 0.01,
            "ask_low": values + 0.01,
            "ask_close": values + 0.01,
        },
        index=index,
    )


def test_policy_grid_is_small_and_deterministic() -> None:
    policies = policy_grid()

    assert len(policies) == 9
    assert policies[0] == Policy("d1_donchian_ch20_hold5", 20, 5)
    assert policies[-1] == Policy("d1_donchian_ch100_hold20", 100, 20)


def test_daily_mid_bars_labels_full_days_by_close_minute() -> None:
    bars = daily_mid_bars(_minute_frame(days=2))

    assert len(bars) == 2
    assert bars.index[0].isoformat() == "2026-01-01T23:59:00+00:00"


def test_breakout_signals_are_next_minute_after_daily_close() -> None:
    signals = breakout_signals(_minute_frame(days=80), pair="USDJPY", policy=Policy("x", 20, 5))

    assert not signals.empty
    assert signals.iloc[0]["entry_timestamp"] == signals.iloc[0]["signal_timestamp"] + pd.Timedelta(
        minutes=1
    )
    assert set(signals["side"]) == {"LONG"}


def test_time_exit_trades_uses_executable_bid_ask_and_costs() -> None:
    frame = _minute_frame(days=30)
    signals = pd.DataFrame(
        [
            {
                "signal_timestamp": pd.Timestamp("2026-01-10T23:59:00Z"),
                "entry_timestamp": pd.Timestamp("2026-01-11T00:00:00Z"),
                "side": "LONG",
                "side_num": 1,
                "breakout_distance_pips": 1.0,
            }
        ]
    )

    trades = time_exit_trades(
        frame,
        pair="USDJPY",
        policy=Policy("x", 20, 5),
        signals=signals,
        commission_pips=0.1,
        slippage_per_side_pips=0.05,
    )

    assert len(trades) == 1
    assert trades.iloc[0]["side"] == "LONG"
    assert float(trades.iloc[0]["net_pips"]) < float(trades.iloc[0]["raw_pips"])


def test_fast128_status_never_approves_strategy() -> None:
    status, reasons = fast128_status(passed_count=1, finalists_count=1)

    assert status == "FAST128_FX_DAILY_DONCHIAN_RESEARCH_CANDIDATE_REQUIRES_RISK_AUDIT"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

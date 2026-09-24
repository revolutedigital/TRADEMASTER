from __future__ import annotations

import pandas as pd

from scripts.research.run_fast129_fx_daily_donchian_fade_scout import (
    Policy,
    fade_signals,
    fast129_status,
    policy_grid,
)


def _minute_frame(days: int = 80, *, direction: str = "up") -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=days * 1440, freq="1min")
    slope = 0.0001 if direction == "up" else -0.0001
    values = 100.0 + pd.Series(range(len(index)), dtype="float64").to_numpy() * slope
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
    assert policies[0] == Policy("d1_donchian_fade_ch20_hold5", 20, 5)
    assert policies[-1] == Policy("d1_donchian_fade_ch100_hold20", 100, 20)


def test_fade_signals_short_after_up_breakout() -> None:
    signals = fade_signals(_minute_frame(direction="up"), pair="USDJPY", policy=Policy("x", 20, 5))

    assert not signals.empty
    assert set(signals["side"]) == {"SHORT"}
    assert set(signals["breakout_direction"]) == {"UP"}


def test_fade_signals_long_after_down_breakout() -> None:
    signals = fade_signals(_minute_frame(direction="down"), pair="USDJPY", policy=Policy("x", 20, 5))

    assert not signals.empty
    assert set(signals["side"]) == {"LONG"}
    assert set(signals["breakout_direction"]) == {"DOWN"}


def test_fast129_status_never_approves_strategy() -> None:
    status, reasons = fast129_status(passed_count=1, finalists_count=1)

    assert status == "FAST129_FX_DAILY_DONCHIAN_FADE_RESEARCH_CANDIDATE_REQUIRES_RISK_AUDIT"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

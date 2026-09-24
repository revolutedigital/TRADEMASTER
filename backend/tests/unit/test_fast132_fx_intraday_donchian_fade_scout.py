from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.run_fast120_fx_currency_strength_scout import Management
from scripts.research.run_fast132_fx_intraday_donchian_fade_scout import (
    Policy,
    fast132_status,
    policy_grid,
    raw_fade_signals,
    side_label,
)


def test_policy_grid_is_deterministic() -> None:
    policies = policy_grid()

    assert len(policies) == 48
    assert policies[0] == Policy(
        identifier="1h_donchian_fade_ch20_stop1_hold6b_fixed_1r",
        timeframe="1h",
        timeframe_minutes=60,
        channel_bars=20,
        stop_atr_multiple=1.0,
        hold_bars=6,
        max_hold_minutes=360,
        management=Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
    )
    assert policies[-1].identifier == "4h_donchian_fade_ch100_stop1.5_hold12b_fixed_2r"


def test_side_label() -> None:
    assert side_label(1) == "LONG"
    assert side_label(-1) == "SHORT"


def test_raw_fade_signals_fades_breakout_direction() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=160 * 60, freq="1min")
    base = np.linspace(1.0000, 1.0010, len(index))
    base[130 * 60 :] += 0.0100
    frame = pd.DataFrame(
        {
            "bid_open": base,
            "bid_high": base + 0.0002,
            "bid_low": base - 0.0002,
            "bid_close": base,
            "ask_open": base + 0.0001,
            "ask_high": base + 0.0003,
            "ask_low": base - 0.0001,
            "ask_close": base + 0.0001,
        },
        index=index,
    )

    signals = raw_fade_signals(frame, pair="EURUSD", timeframe="1h", channel_bars=1)

    assert signals
    assert signals[-1].side_num == -1
    assert signals[-1].breakout_distance_pips > 0


def test_fast132_status_never_approves_strategy() -> None:
    status, reasons = fast132_status(passed_count=1, finalists_count=1)

    assert status == "FAST132_FX_INTRADAY_DONCHIAN_FADE_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

    status, reasons = fast132_status(passed_count=0, finalists_count=1)

    assert status == "FAST132_FX_INTRADAY_DONCHIAN_FADE_FAILED_AUDIT"
    assert "selection_positive_candidates_failed_2024_2026_audit" in reasons

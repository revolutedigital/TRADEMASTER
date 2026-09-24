"""Fast120 currency-strength scout tests."""

import numpy as np
import pandas as pd
import pytest

from scripts.research.run_fast120_fx_currency_strength_scout import (
    Management,
    Policy,
    currency_scores,
    fast120_status,
    pair_score_matrix,
    policy_grid,
    signals_for_policy,
    simulate_trade,
    split_pair,
    widen_quotes,
)


def _bars(
    index: pd.DatetimeIndex,
    rows: list[tuple[float, float, float, float, float, float, float, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
        ],
        index=index,
    )


def test_policy_grid_covers_modes_sessions_and_managed_exits() -> None:
    grid = policy_grid()

    assert len(grid) == 324
    assert grid[0].identifier == "momentum_lb60_h00_thr5_hold60_fixed_1r"
    assert grid[-1].identifier == "reversal_lb1440_h13_thr20_hold240_trail_1r_0_75r"


def test_split_pair_uses_instrument_parser() -> None:
    assert split_pair("EURUSD") == ("EUR", "USD")
    assert split_pair("USDJPY") == ("USD", "JPY")


def test_currency_scores_turn_pair_returns_into_currency_basket_scores() -> None:
    index = pd.DatetimeIndex(["2026-01-01 00:00"], tz="UTC")
    returns = pd.DataFrame(0.0, index=index, columns=[
        "AUDUSD",
        "EURGBP",
        "EURJPY",
        "EURUSD",
        "GBPJPY",
        "GBPUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "USDJPY",
    ])
    returns.loc[index[0], "EURUSD"] = 10.0

    scores = currency_scores(returns)
    pair_scores = pair_score_matrix(scores)

    assert scores.loc[index[0], "EUR"] > 0
    assert scores.loc[index[0], "USD"] < 0
    assert pair_scores.loc[index[0], "EURUSD"] > 0


def test_signals_choose_largest_tradable_divergence_and_flip_reversal() -> None:
    index = pd.date_range("2026-01-01 00:00", periods=62, freq="min", tz="UTC")
    mids = pd.DataFrame(1.0, index=index, columns=[
        "AUDUSD",
        "EURGBP",
        "EURJPY",
        "EURUSD",
        "GBPJPY",
        "GBPUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "USDJPY",
    ])
    mids.loc[index[60], "EURUSD"] = 1.01
    policy = Policy(
        "test",
        mode="reversal",
        lookback_minutes=60,
        decision_hour_utc=1,
        threshold_bps=1.0,
        max_hold_minutes=60,
        management=Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
    )

    signals = signals_for_policy(mids, policy)

    assert len(signals) == 1
    assert signals.iloc[0]["pair"] == "EURUSD"
    assert signals.iloc[0]["side"] == -1


def test_simulate_trade_uses_ask_entry_bid_exit_and_stop_first() -> None:
    index = pd.date_range("2026-01-01 00:01", periods=3, freq="min", tz="UTC")
    frame = _bars(
        index,
        [
            (1.1000, 1.1010, 1.0997, 1.1008, 1.1002, 1.1012, 1.0999, 1.1010),
            (1.1008, 1.1010, 1.1008, 1.1009, 1.1010, 1.1012, 1.1010, 1.1011),
            (1.1009, 1.1009, 1.1009, 1.1009, 1.1011, 1.1011, 1.1011, 1.1011),
        ],
    )

    result = simulate_trade(
        frame,
        pair="EURUSD",
        entry_timestamp=index[0],
        side=1,
        risk_pips=5.0,
        hold_minutes=2,
        management=Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
        commission=0.0,
        slippage_per_side_pips=0.0,
    )

    assert result is not None
    assert result.exit_reason == "stop"
    assert result.raw_pips == pytest.approx(-5.0)


def test_widen_quotes_keeps_midpoint_and_doubles_spread() -> None:
    index = pd.date_range("2026-01-01 00:00", periods=1, freq="min", tz="UTC")
    frame = _bars(index, [(1.0, 1.2, 0.9, 1.1, 1.1, 1.3, 1.0, 1.2)])

    widened = widen_quotes(frame, 2.0)

    assert np.allclose(0.5 * (frame["bid_open"] + frame["ask_open"]), 0.5 * (widened["bid_open"] + widened["ask_open"]))
    assert np.allclose(widened["ask_open"] - widened["bid_open"], 2.0 * (frame["ask_open"] - frame["bid_open"]))


def test_fast120_status_never_approves_strategy() -> None:
    status, reasons = fast120_status(passed_count=1, finalists_count=3)

    assert status == "FAST120_FX_CURRENCY_STRENGTH_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

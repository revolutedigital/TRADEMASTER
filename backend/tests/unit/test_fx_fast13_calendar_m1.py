"""Round-13 macro-calendar M1 simulation tests."""

import numpy as np
import pandas as pd

from scripts.research.fx_fast13_calendar_m1 import (
    Management,
    Policy,
    PreparedEvent,
    policy_grid,
    side_for_event,
    simulate_scenario,
    widen_quotes,
)


def _bars(rows: list[tuple[float, float, float, float, float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-02 13:30", periods=len(rows), freq="min", tz="UTC")
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


def _event(**extra: object) -> PreparedEvent:
    values = {
        "event_timestamp": pd.Timestamp("2026-01-02 13:30", tz="UTC"),
        "event_name": "NFP",
        "currency": "USD",
        "pair": "EURUSD",
        "impulse_pips": 10.0,
        "baseline_range_pips": 2.0,
        "risk_pips": 5.0,
    }
    values.update(extra)
    return PreparedEvent(**values)


def _policy(**extra: object) -> Policy:
    values = {
        "identifier": "test",
        "direction": "drift",
        "threshold": 1.5,
        "entry_delay_minutes": 1,
        "max_hold_minutes": 1,
        "management": Management("fixed_1_5r", target_r=1.5, activation_r=None, trail_r=None),
    }
    values.update(extra)
    return Policy(**values)


def test_policy_grid_is_the_preregistered_108_hypotheses() -> None:
    grid = policy_grid()
    assert len(grid) == 108
    assert grid[0].identifier == "drift_thr1.5_d5_h30_fixed_1_5r"
    assert grid[-1].identifier == "reversal_thr2_d15_h120_trail_0_75r_0_5r"


def test_reversal_flips_the_observed_impulse_side() -> None:
    assert side_for_event(_event(impulse_pips=8.0), _policy(direction="drift")) == 1
    assert side_for_event(_event(impulse_pips=8.0), _policy(direction="reversal")) == -1
    assert side_for_event(_event(impulse_pips=-8.0), _policy(direction="drift")) == -1
    assert side_for_event(_event(impulse_pips=-8.0), _policy(direction="reversal")) == 1


def test_long_uses_ask_entry_bid_exit_and_pessimistic_same_bar_stop() -> None:
    frame = _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
            # Entry at ask_open 1.1002. This same candle touches the 1.5R target and the 1R stop.
            (1.1000, 1.1010, 1.0997, 1.1008, 1.1002, 1.1012, 1.0999, 1.1010),
            (1.1008, 1.1008, 1.1008, 1.1008, 1.1010, 1.1010, 1.1010, 1.1010),
        ]
    )
    result = simulate_scenario(
        frame,
        _event(),
        _policy(),
        side=1,
        commission=0,
        slippage_per_side_pips=0,
    )
    assert result is not None
    assert result.exit_reason == "stop"
    assert np.isclose(result.raw_pips, -5.0)


def test_trailing_can_lock_a_runner_after_activation() -> None:
    frame = _bars(
        [
            (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
            (1.1000, 1.1005, 1.1000, 1.1004, 1.1002, 1.1007, 1.1002, 1.1006),
            (1.1004, 1.1010, 1.1006, 1.1008, 1.1006, 1.1012, 1.1008, 1.1010),
            (1.1008, 1.1009, 1.1005, 1.1006, 1.1010, 1.1011, 1.1007, 1.1008),
        ]
    )
    result = simulate_scenario(
        frame,
        _event(risk_pips=4.0),
        _policy(
            max_hold_minutes=2,
            management=Management("trail_0_75r_0_5r", target_r=None, activation_r=0.75, trail_r=0.5),
        ),
        side=1,
        commission=0,
        slippage_per_side_pips=0,
    )
    assert result is not None
    assert result.exit_reason == "stop"
    assert result.raw_pips > 0


def test_widen_quotes_keeps_midpoint_and_doubles_spread() -> None:
    frame = _bars([(1.0, 1.2, 0.9, 1.1, 1.1, 1.3, 1.0, 1.2)])
    widened = widen_quotes(frame, 2.0)
    original_mid = 0.5 * (frame["bid_open"] + frame["ask_open"])
    widened_mid = 0.5 * (widened["bid_open"] + widened["ask_open"])
    assert np.allclose(original_mid, widened_mid)
    assert np.allclose(
        widened["ask_open"] - widened["bid_open"],
        2.0 * (frame["ask_open"] - frame["bid_open"]),
    )

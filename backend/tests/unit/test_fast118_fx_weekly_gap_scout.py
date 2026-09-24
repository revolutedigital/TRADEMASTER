"""Fast118 weekly-gap scout tests."""

import numpy as np
import pandas as pd
import pytest

from scripts.research.run_fast118_fx_weekly_gap_scout import (
    Management,
    Policy,
    WeeklyGap,
    fast118_status,
    find_weekly_gaps,
    policy_grid,
    side_for_gap,
    simulate_scenario,
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


def _policy(**extra: object) -> Policy:
    values = {
        "identifier": "test",
        "direction": "reversion",
        "minimum_gap_pips": 2.0,
        "entry_delay_minutes": 1,
        "max_hold_minutes": 2,
        "management": Management(
            "fixed_half_gap",
            stop_gap_multiple=1.0,
            target_gap_fraction=0.5,
            activation_gap_fraction=None,
            trail_gap_fraction=None,
        ),
    }
    values.update(extra)
    return Policy(**values)


def _gap(**extra: object) -> WeeklyGap:
    values = {
        "pair": "EURUSD",
        "previous_timestamp": pd.Timestamp("2026-01-02 21:59", tz="UTC"),
        "reopen_timestamp": pd.Timestamp("2026-01-04 22:00", tz="UTC"),
        "previous_mid_close": 1.1000,
        "reopen_mid_open": 1.1010,
        "gap_pips": 10.0,
    }
    values.update(extra)
    return WeeklyGap(**values)


def test_policy_grid_has_pair_local_weekly_gap_hypotheses() -> None:
    grid = policy_grid()

    assert len(grid) == 108
    assert grid[0].identifier == "reversion_gap2_d5_h4_fixed_half_gap"
    assert grid[-1].identifier == "continuation_gap10_d30_h12_trail_after_half_gap"


def test_find_weekly_gaps_uses_the_first_bar_after_market_closure() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-02 21:58",
            "2026-01-02 21:59",
            "2026-01-04 22:00",
            "2026-01-04 22:01",
        ],
        tz="UTC",
    )
    frame = _bars(
        index,
        [
            (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
            (1.1000, 1.1000, 1.1000, 1.1000, 1.1002, 1.1002, 1.1002, 1.1002),
            (1.1010, 1.1010, 1.1010, 1.1010, 1.1012, 1.1012, 1.1012, 1.1012),
            (1.1010, 1.1010, 1.1010, 1.1010, 1.1012, 1.1012, 1.1012, 1.1012),
        ],
    )

    gaps = find_weekly_gaps(frame, "EURUSD")

    assert len(gaps) == 1
    assert gaps[0].previous_timestamp == pd.Timestamp("2026-01-02 21:59", tz="UTC")
    assert gaps[0].reopen_timestamp == pd.Timestamp("2026-01-04 22:00", tz="UTC")
    assert gaps[0].gap_pips == pytest.approx(10.0)


def test_reversion_flips_gap_direction_and_continuation_follows_it() -> None:
    assert side_for_gap(_gap(gap_pips=10.0), _policy(direction="reversion")) == -1
    assert side_for_gap(_gap(gap_pips=10.0), _policy(direction="continuation")) == 1
    assert side_for_gap(_gap(gap_pips=-10.0), _policy(direction="reversion")) == 1
    assert side_for_gap(_gap(gap_pips=-10.0), _policy(direction="continuation")) == -1


def test_short_reversion_pays_the_bid_entry_and_ask_exit() -> None:
    index = pd.date_range("2026-01-04 22:00", periods=4, freq="min", tz="UTC")
    frame = _bars(
        index,
        [
            (1.1010, 1.1010, 1.1010, 1.1010, 1.1012, 1.1012, 1.1012, 1.1012),
            # Entry short at bid_open 1.1008; target is 5 pips lower on ask_low.
            (1.1008, 1.1009, 1.1002, 1.1004, 1.1010, 1.1011, 1.1003, 1.1005),
            (1.1004, 1.1005, 1.1001, 1.1002, 1.1006, 1.1007, 1.1002, 1.1004),
            (1.1002, 1.1002, 1.1002, 1.1002, 1.1004, 1.1004, 1.1004, 1.1004),
        ],
    )

    result = simulate_scenario(
        frame,
        _gap(),
        _policy(),
        side=-1,
        commission=0.0,
        slippage_per_side_pips=0.0,
    )

    assert result is not None
    assert result.exit_reason == "target"
    assert result.raw_pips == pytest.approx(5.0)


def test_trailing_can_lock_a_weekly_gap_runner() -> None:
    index = pd.date_range("2026-01-04 22:00", periods=5, freq="min", tz="UTC")
    frame = _bars(
        index,
        [
            (1.1010, 1.1010, 1.1010, 1.1010, 1.1012, 1.1012, 1.1012, 1.1012),
            (1.1008, 1.1014, 1.1008, 1.1012, 1.1010, 1.1016, 1.1010, 1.1014),
            (1.1012, 1.1018, 1.1012, 1.1017, 1.1014, 1.1020, 1.1014, 1.1019),
            (1.1017, 1.1017, 1.1013, 1.1014, 1.1019, 1.1019, 1.1015, 1.1016),
            (1.1014, 1.1014, 1.1014, 1.1014, 1.1016, 1.1016, 1.1016, 1.1016),
        ],
    )

    result = simulate_scenario(
        frame,
        _gap(),
        _policy(
            direction="continuation",
            management=Management(
                "trail_after_half_gap",
                stop_gap_multiple=1.0,
                target_gap_fraction=None,
                activation_gap_fraction=0.5,
                trail_gap_fraction=0.35,
            ),
            max_hold_minutes=3,
        ),
        side=1,
        commission=0.0,
        slippage_per_side_pips=0.0,
    )

    assert result is not None
    assert result.exit_reason == "stop"
    assert result.raw_pips > 0


def test_widen_quotes_keeps_midpoint_and_doubles_spread() -> None:
    index = pd.date_range("2026-01-04 22:00", periods=1, freq="min", tz="UTC")
    frame = _bars(index, [(1.0, 1.2, 0.9, 1.1, 1.1, 1.3, 1.0, 1.2)])

    widened = widen_quotes(frame, 2.0)

    original_mid = 0.5 * (frame["bid_open"] + frame["ask_open"])
    widened_mid = 0.5 * (widened["bid_open"] + widened["ask_open"])
    assert np.allclose(original_mid, widened_mid)
    assert np.allclose(
        widened["ask_open"] - widened["bid_open"],
        2.0 * (frame["ask_open"] - frame["bid_open"]),
    )


def test_fast118_status_never_approves_strategy() -> None:
    status, reasons = fast118_status(passed_count=1, finalists_count=5)

    assert status == "FAST118_FX_WEEKLY_GAP_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

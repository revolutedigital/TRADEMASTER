"""First-touch and trailing-path correctness for round 5."""

import numpy as np
import pandas as pd

from app.fx.instruments import Instrument
from app.fx.sim.costs import CostScenario
from scripts.research.fx_fast4_events import EventCosts
from scripts.research.fx_fast5_events import (
    EXIT_INITIAL_STOP,
    EXIT_PRE_ACTIVATION_TIMEOUT,
    EXIT_TRAILING_STOP,
    build_entry_labels,
    simulate_trailing_outcome,
)

ZERO = CostScenario(
    name="zero",
    spread_multiplier=1.0,
    slippage_pips=0.0,
    slippage_range_fraction=0.0,
    commission_base_per_lot_per_side=0.0,
)
COSTS = EventCosts(0.0, 0.0, base=ZERO, stress=ZERO)
EURUSD = Instrument.from_symbol("EURUSD")


def ticks(prices: list[float], seconds: list[int] | None = None) -> pd.DataFrame:
    offsets = seconds or list(range(len(prices)))
    index = pd.DatetimeIndex(
        [pd.Timestamp("2021-01-04T00:00:00Z") + pd.Timedelta(seconds=value) for value in offsets]
    )
    return pd.DataFrame({"bid": prices, "ask": prices}, index=index)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"decision_index": [0], "mid_range_pips_256": [1.0]},
        index=pd.DatetimeIndex([frame.index[0]], name="decision_time"),
    )


def test_first_touch_uses_order_not_only_mfe_and_mae() -> None:
    winner = ticks([1.0, 1.0, 1.00005, 0.9998, 1.0002])
    loser = ticks([1.0, 1.0, 0.9999, 1.0002, 1.0003])
    won = build_entry_labels(winner, features(winner), EURUSD, COSTS, activation_seconds=3)
    lost = build_entry_labels(loser, features(loser), EURUSD, COSTS, activation_seconds=3)
    assert won.iloc[0]["long_hit_base"] == 1.0
    assert lost.iloc[0]["long_hit_base"] == 0.0


def test_first_touch_is_symmetric_for_short_entries() -> None:
    frame = ticks([1.0, 1.0, 0.99995, 0.99996, 0.99997])
    labels = build_entry_labels(frame, features(frame), EURUSD, COSTS, activation_seconds=3)
    assert labels.iloc[0]["short_hit_base"] == 1.0
    assert labels.iloc[0]["long_hit_base"] == 0.0


def test_runner_moves_to_net_breakeven_and_captures_unbounded_profit() -> None:
    frame = ticks([1.0, 1.0, 1.00005, 1.00010, 1.00020, 1.00016, 1.00014])
    result = simulate_trailing_outcome(
        frame,
        decision_index=0,
        side=1,
        instrument=EURUSD,
        risk_pips=1.0,
        activation_seconds=3,
        max_hold_seconds=10,
        trail_distance_r=0.5,
    )
    assert result.activated
    assert result.exit_reason == EXIT_TRAILING_STOP
    assert np.isclose(result.best_r, 2.0)
    assert np.isclose(result.result_r, 1.4)


def test_initial_stop_and_preactivation_timeout_are_distinct() -> None:
    stopped = ticks([1.0, 1.0, 0.9999, 1.0])
    timed_out = ticks([1.0, 1.0, 1.00001, 1.00002, 1.00003])
    stop_result = simulate_trailing_outcome(
        stopped,
        decision_index=0,
        side=1,
        instrument=EURUSD,
        risk_pips=1.0,
        activation_seconds=3,
        max_hold_seconds=10,
    )
    timeout_result = simulate_trailing_outcome(
        timed_out,
        decision_index=0,
        side=1,
        instrument=EURUSD,
        risk_pips=1.0,
        activation_seconds=3,
        max_hold_seconds=10,
    )
    assert stop_result.exit_reason == EXIT_INITIAL_STOP
    assert timeout_result.exit_reason == EXIT_PRE_ACTIVATION_TIMEOUT


def test_breakeven_is_net_of_commission() -> None:
    frame = ticks([1.0, 1.0, 1.00006, 1.00002, 1.00001])
    result = simulate_trailing_outcome(
        frame,
        decision_index=0,
        side=1,
        instrument=EURUSD,
        risk_pips=1.0,
        commission_pips=0.1,
        activation_seconds=3,
        max_hold_seconds=10,
        trail_distance_r=0.5,
    )
    assert result.activated
    assert result.exit_reason == EXIT_TRAILING_STOP
    assert np.isclose(result.result_r, 0.0)

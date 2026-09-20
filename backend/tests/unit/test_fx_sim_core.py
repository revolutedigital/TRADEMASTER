"""The strategy-driven simulator core against the validated spike core, and its own invariants."""

import numpy as np
import pandas as pd
import pytest
from numba import njit

from app.fx import strategy as fx
from app.fx.sim import core
from scripts.research.engine_spike import numba_core as spike
from tests.unit.test_engine_spike_numba import COLUMNS, random_bars

PIP = 0.0001


def matrix_of(bars: pd.DataFrame) -> np.ndarray:
    return fx.bars_to_matrix(bars)


def run(matrix, *, fast=8, slow=21, atr=14, stop_atr=1.5, rr=2.0, warmup=60, slippage=0.2 * PIP):
    params = fx.ema_cross_params(
        fast_span=fast, slow_span=slow, atr_period=atr, stop_atr=stop_atr, reward_risk=rr, warmup=warmup
    )
    return core.simulate(
        fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, matrix, slippage
    )


def run_spike(bars, *, fast=8, slow=21, atr=14, stop_atr=1.5, rr=2.0, warmup=60, slippage=0.2 * PIP):
    arrays = [bars[name].to_numpy(dtype=np.float64) for name in COLUMNS]
    return spike.simulate(*arrays, fast, slow, atr, stop_atr, rr, slippage, 0.0, warmup)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "overrides",
    [{}, {"fast": 5, "slow": 34}, {"stop_atr": 0.8, "rr": 1.0}, {"slippage": 0.0}],
)
def test_the_strategy_driven_core_reproduces_the_validated_spike_core_trade_by_trade(seed, overrides) -> None:
    bars = random_bars(seed)

    new = run(matrix_of(bars), **overrides)
    old = run_spike(bars, **overrides)

    assert len(old[2]) > 10
    for new_column, old_column in zip(new, old, strict=True):
        assert np.array_equal(new_column, old_column) or np.allclose(new_column, old_column, atol=1e-12, rtol=0)


def test_every_trade_obeys_the_fill_semantics() -> None:
    bars = random_bars(21, n=8000)
    slip = 0.2 * PIP
    entry_index, exit_index, side, entry_price, exit_price, distance, reason = run(matrix_of(bars))

    gross = (exit_price - entry_price) * side
    stop = reason == core.EXIT_STOP
    gapped = reason == core.EXIT_STOP_GAP
    target = reason == core.EXIT_TARGET

    assert stop.any() and gapped.any() and target.any()
    assert np.all(exit_index >= entry_index)
    assert np.allclose(gross[stop], -distance[stop] - slip, atol=1e-12)
    assert np.allclose(gross[target], 2.0 * distance[target], atol=1e-12)
    assert np.all(gross[gapped] <= -distance[gapped] - slip + 1e-12)


def test_no_trade_is_ever_entered_on_the_bar_that_produced_its_signal() -> None:
    bars = random_bars(31, n=4000)
    matrix = matrix_of(bars)
    intents, _, _ = fx.run_batch(
        fx.ema_cross_step, fx.ema_cross_init,
        fx.ema_cross_params(fast_span=8, slow_span=21, atr_period=14, stop_atr=1.5, reward_risk=2.0, warmup=60),
        fx.EMA_CROSS_STATE_SIZE, matrix, np.zeros(len(matrix), dtype=np.int64),
    )

    entry_index, *_ = run(matrix)
    first_signal = int(np.flatnonzero(intents != fx.HOLD)[0])

    assert entry_index[0] >= first_signal + 1


# --- a scripted strategy makes each rule testable by hand -----------------------------------------


@njit(cache=True)
def scripted_init(params, state):
    state[0] = 0.0


@njit(cache=True)
def scripted_step(params, state, bar, position):
    """params = [bar_index_to_act_on, intent, stop_distance, target_distance, second_index, second_intent]."""
    index = state[0]
    state[0] = index + 1.0
    if index == params[0]:
        return int(params[1]), params[2], params[3]
    if index == params[4]:
        return int(params[5]), params[2], params[3]
    return 0, 0.0, 0.0


def flat_bars(n: int, price: float = 1.1000, spread_pips: float = 1.0) -> np.ndarray:
    half = spread_pips * PIP / 2
    matrix = np.zeros((n, fx.BAR_WIDTH))
    matrix[:, fx.BAR_TIME] = np.arange(n) * 60.0
    for offset, sign in ((fx.BID_OPEN, -1), (fx.ASK_OPEN, 1)):
        for column in range(4):
            matrix[:, offset + column] = price + sign * half
    return matrix


def scripted(matrix, *, act, intent, stop=0.01, target=0.02, second=-1, second_intent=0, slippage=0.0):
    params = np.array([act, intent, stop, target, second, second_intent], dtype=np.float64)
    return core.simulate(scripted_step, scripted_init, params, 1, matrix, slippage)


def test_a_long_entry_pays_the_ask_at_the_next_open_and_a_target_exit_earns_the_planned_distance() -> None:
    matrix = flat_bars(6)
    matrix[3, fx.BID_HIGH] = 1.1300  # the bid trades up through the 1.1150 target on bar 3

    entry_index, exit_index, side, entry_price, exit_price, distance, reason = scripted(
        matrix, act=1, intent=fx.ENTER_LONG
    )

    assert (entry_index[0], exit_index[0], side[0]) == (2, 3, fx.LONG)
    assert entry_price[0] == pytest.approx(1.1000 + 0.5 * PIP)  # the ask
    assert reason[0] == core.EXIT_TARGET
    assert exit_price[0] - entry_price[0] == pytest.approx(0.02)


def test_a_short_entry_receives_the_bid_and_slippage_is_charged_against_the_trader() -> None:
    matrix = flat_bars(5)

    entry_index, _, side, entry_price, exit_price, *_ = scripted(
        matrix, act=1, intent=fx.ENTER_SHORT, slippage=0.5 * PIP
    )

    assert side[0] == fx.SHORT and entry_index[0] == 2
    assert entry_price[0] == pytest.approx(1.1000 - 0.5 * PIP - 0.5 * PIP)
    assert exit_price[0] == pytest.approx(1.1000 + 0.5 * PIP + 0.5 * PIP)  # closed at the end, at the ask


def test_the_stop_wins_when_a_bar_touches_both_the_stop_and_the_target() -> None:
    matrix = flat_bars(6)
    matrix[3, fx.BID_HIGH] = 1.1300
    matrix[3, fx.BID_LOW] = 1.0900

    *_, reason = scripted(matrix, act=1, intent=fx.ENTER_LONG)

    assert reason[0] == core.EXIT_STOP


def test_a_gap_through_the_stop_fills_at_the_gapped_open() -> None:
    matrix = flat_bars(6)
    for column in range(fx.BID_OPEN, fx.BID_CLOSE + 1):
        matrix[3:, column] = 1.0500  # the bid reopens far below the 1.0900 stop

    _, _, _, entry_price, exit_price, distance, reason = scripted(matrix, act=1, intent=fx.ENTER_LONG)

    assert reason[0] == core.EXIT_STOP_GAP
    assert exit_price[0] == pytest.approx(1.0500)
    assert entry_price[0] - exit_price[0] > distance[0]


def test_an_opposite_signal_reverses_and_an_exit_signal_flattens() -> None:
    matrix = flat_bars(10)

    reversed_run = scripted(matrix, act=1, intent=fx.ENTER_LONG, second=4, second_intent=fx.ENTER_SHORT)
    flattened = scripted(matrix, act=1, intent=fx.ENTER_LONG, second=4, second_intent=fx.EXIT)

    assert reversed_run[2].tolist() == [fx.LONG, fx.SHORT]
    assert reversed_run[6][0] == core.EXIT_SIGNAL and reversed_run[0][1] == reversed_run[1][0] == 5
    assert flattened[2].tolist() == [fx.LONG] and flattened[6][0] == core.EXIT_SIGNAL


def test_an_entry_without_a_stop_distance_is_refused_and_a_repeated_signal_does_not_pyramid() -> None:
    matrix = flat_bars(8)

    no_stop = scripted(matrix, act=1, intent=fx.ENTER_LONG, stop=0.0)
    same_side = scripted(matrix, act=1, intent=fx.ENTER_LONG, second=3, second_intent=fx.ENTER_LONG)

    assert len(no_stop[2]) == 0
    assert len(same_side[2]) == 1 and same_side[6][0] == core.EXIT_END


def test_a_position_still_open_at_the_end_is_closed_at_the_last_bar() -> None:
    result = scripted(flat_bars(6), act=1, intent=fx.ENTER_LONG)

    assert result[1][-1] == 5 and result[6][-1] == core.EXIT_END
    assert result[4][-1] == pytest.approx(1.1000 - 0.5 * PIP)  # sold at the bid


def test_net_pips_subtracts_the_round_trip_commission() -> None:
    side = np.array([fx.LONG, fx.SHORT], dtype=np.int8)
    entry = np.array([1.1000, 1.1000])
    exit_ = np.array([1.1010, 1.0990])

    result = core.net_pips(side, entry, exit_, 0.45 * PIP, PIP)

    assert result.tolist() == pytest.approx([10.0 - 0.45, 10.0 - 0.45])


def test_a_series_with_no_signal_never_trades() -> None:
    result = scripted(flat_bars(20), act=99, intent=fx.ENTER_LONG)

    assert len(result[2]) == 0

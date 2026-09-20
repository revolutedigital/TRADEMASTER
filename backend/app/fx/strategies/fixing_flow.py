"""Family 2: dollar flow around the 16:00 London fixing (F2a before it, F2b after it).

The hypothesis is that currency rebalancing near the fixing pushes the dollar before the fix and
gives some of it back afterwards. The bot is a clock: it opens a position in the dollar's
direction at the open of a fixed bar, holds it to the open of another fixed bar, and carries a
protective stop of a multiple of the recent ATR that it does not expect to touch. There is no
profit target.

`pair_side` is the direction taken in the pair, not in the dollar: buying dollars is buying a pair
that has the dollar as its base (USDJPY) and selling a pair that has it as its quote (EURUSD).
If the bar of the intended entry is missing from the data the entry happens at the next bar
that exists; the lab drops those trades (`entry_is_on_schedule`).

The exact rules are in docs/forex/fast-preregistration.md (F2a and F2b).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from app.fx.sessions import LONDON, NEW_YORK, local_day, local_time_of_day, seconds_of_day
from app.fx.strategies.common import (
    FAR_TARGET,
    ring_count,
    ring_mean,
    ring_push,
    ring_slots,
    true_range,
)
from app.fx.strategy import (
    ASK_CLOSE,
    ASK_HIGH,
    ASK_LOW,
    BAR_TIME,
    BID_CLOSE,
    BID_HIGH,
    BID_LOW,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT,
    FLAT,
    HOLD,
    LONG,
    SHORT,
)

ATR_PERIOD = 14
NO_DAY = -1.0e9

_ZONE, _ENTRY_TIME, _EXIT_TIME, _PAIR_SIDE, _BAR_SECONDS, _STOP_ATR = 0, 1, 2, 3, 4, 5

_PREVIOUS_CLOSE, _HAS_PREVIOUS, _ENTRY_DAY = 0, 1, 2
_ATR_RING = 3
FIXING_FLOW_STATE_SIZE = _ATR_RING + ring_slots(ATR_PERIOD)

USD_BASE = frozenset({"USDJPY", "USDCAD", "USDCHF"})
USD_QUOTE = frozenset({"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"})


def pair_side_for_dollar(pair: str, buy_dollar: bool) -> int:
    """The side to take in `pair` to buy (or sell) the dollar."""
    if pair in USD_BASE:
        return LONG if buy_dollar else SHORT
    if pair in USD_QUOTE:
        return SHORT if buy_dollar else LONG
    raise ValueError(f"{pair} does not have the dollar on either side")


def fixing_flow_params(
    *,
    pair_side: int,
    entry_time: str,
    exit_time: str,
    zone: int = LONDON,
    bar_seconds: int = 300,
    stop_atr: float = 3.0,
) -> np.ndarray:
    """Parameters for one leg of the fixing flow; the times are the opens of the entry and exit bars."""
    if pair_side not in (LONG, SHORT):
        raise ValueError("pair_side must be LONG or SHORT")
    if zone not in (LONDON, NEW_YORK):
        raise ValueError("zone must be sessions.LONDON or sessions.NEW_YORK")
    entry, exit_ = seconds_of_day(entry_time), seconds_of_day(exit_time)
    if not entry < exit_:
        raise ValueError("the exit must come after the entry")
    if bar_seconds < 60 or bar_seconds % 60 or stop_atr <= 0:
        raise ValueError("bar_seconds and stop_atr must be positive")
    return np.array([zone, entry, exit_, pair_side, bar_seconds, stop_atr], dtype=np.float64)


def pre_fixing_params(pair: str) -> np.ndarray:
    """F2a: buy the dollar at the 15:00 London open, sell at the 15:55 open."""
    return fixing_flow_params(
        pair_side=pair_side_for_dollar(pair, buy_dollar=True), entry_time="15:00", exit_time="15:55"
    )


def post_fixing_params(pair: str) -> np.ndarray:
    """F2b: sell the dollar at the 16:05 London open, buy it back at the 17:00 open."""
    return fixing_flow_params(
        pair_side=pair_side_for_dollar(pair, buy_dollar=False), entry_time="16:05", exit_time="17:00"
    )


def entry_is_on_schedule(bars: np.ndarray, entry_index: np.ndarray, params: np.ndarray) -> np.ndarray:
    """True for the trades that opened on the bar the rule names, and not on a later one."""
    zone = int(params[_ZONE])
    scheduled = np.array(
        [local_time_of_day(bars[index, BAR_TIME], zone) for index in entry_index], dtype=np.float64
    )
    return scheduled == params[_ENTRY_TIME]


@njit(cache=True)
def fixing_flow_init(params, state):
    for i in range(state.shape[0]):
        state[i] = 0.0
    state[_ENTRY_DAY] = NO_DAY


@njit(cache=True)
def fixing_flow_step(params, state, bar, position):
    zone = int(params[_ZONE])
    bar_seconds = params[_BAR_SECONDS]

    opened = bar[BAR_TIME]
    mid_high = 0.5 * (bar[BID_HIGH] + bar[ASK_HIGH])
    mid_low = 0.5 * (bar[BID_LOW] + bar[ASK_LOW])
    mid_close = 0.5 * (bar[BID_CLOSE] + bar[ASK_CLOSE])

    ring_push(
        state, _ATR_RING, ATR_PERIOD,
        true_range(mid_high, mid_low, state[_PREVIOUS_CLOSE], state[_HAS_PREVIOUS] > 0.5),
    )
    state[_PREVIOUS_CLOSE] = mid_close
    state[_HAS_PREVIOUS] = 1.0

    today = float(local_day(opened, zone))
    closes_at = float(local_time_of_day(opened, zone)) + bar_seconds

    intent = HOLD
    stop_distance = 0.0
    target_distance = 0.0
    if position != FLAT:
        if closes_at >= params[_EXIT_TIME] or today != state[_ENTRY_DAY]:
            intent = EXIT
    elif closes_at == params[_ENTRY_TIME] and ring_count(state, _ATR_RING) == ATR_PERIOD:
        stop_distance = params[_STOP_ATR] * ring_mean(state, _ATR_RING)
        if stop_distance > 0.0:
            intent = ENTER_LONG if params[_PAIR_SIDE] > 0 else ENTER_SHORT
            target_distance = FAR_TARGET
            state[_ENTRY_DAY] = today
        else:
            stop_distance = 0.0
    return intent, stop_distance, target_distance

"""Compiled event-driven simulator that replays bars through any streaming strategy.

The loop reproduces how a broker treats a bot, without looking ahead:

* a strategy decides at a bar's close and the order is filled at the next bar's open;
* buys fill at the ask and sells at the bid, with slippage always against the trader;
* the protective stop and target are tested inside each bar, the stop wins when both are
  touched, and a stop that the market gaps through fills at the gapped open, not at the stop;
* a signal in the opposite direction closes the position and opens the other one.

It is intentionally free of pandas, of I/O and of any cost that depends on the time of day:
those enter through the bars themselves (each bar carries its own bid and ask) and through the
cost model in `app.fx.sim.costs`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from app.fx.strategy import (
    ASK_CLOSE,
    ASK_HIGH,
    ASK_LOW,
    ASK_OPEN,
    BID_CLOSE,
    BID_HIGH,
    BID_LOW,
    BID_OPEN,
    ENTER_LONG,
    ENTER_SHORT,
    EXIT,
    HOLD,
    LONG,
    SHORT,
)

EXIT_SIGNAL = 1
EXIT_STOP = 2
EXIT_STOP_GAP = 3
EXIT_TARGET = 4
EXIT_END = 5


@njit(cache=True)
def simulate(step, init, params, state_size, bars, slippage):  # noqa: PLR0912, PLR0915
    """Replay `bars` through a strategy and return the trades it made.

    `bars` is the (n, 9) matrix of `app.fx.strategy`; `slippage` is in price units, per fill.
    Returns entry index, exit index, side, entry price, exit price, stop distance and the reason
    each trade ended, as parallel arrays.
    """
    n = bars.shape[0]
    capacity = 2 * n + 2  # a bar can hold a signal exit and the stop of the trade that replaced it
    entry_index = np.empty(capacity, dtype=np.int64)
    exit_index = np.empty(capacity, dtype=np.int64)
    side = np.empty(capacity, dtype=np.int8)
    entry_price = np.empty(capacity, dtype=np.float64)
    exit_price = np.empty(capacity, dtype=np.float64)
    stop_distance = np.empty(capacity, dtype=np.float64)
    reason = np.empty(capacity, dtype=np.int8)
    trades = 0

    state = np.zeros(state_size, dtype=np.float64)
    init(params, state)

    position = 0
    open_index = 0
    open_price = 0.0
    open_distance = 0.0
    stop_price = 0.0
    target_price = 0.0
    pending = HOLD
    pending_stop = 0.0
    pending_target = 0.0

    for t in range(n):
        bar = bars[t]

        # 1. Act on the decision taken at the previous close, at this bar's open.
        if pending != HOLD and t > 0:
            wants_exit = pending == EXIT
            wants_side = 0
            if pending == ENTER_LONG:
                wants_side = LONG
            elif pending == ENTER_SHORT:
                wants_side = SHORT
            if position != 0 and (wants_exit or (wants_side != 0 and wants_side != position)):
                if position == LONG:
                    fill = bar[BID_OPEN] - slippage
                else:
                    fill = bar[ASK_OPEN] + slippage
                entry_index[trades] = open_index
                exit_index[trades] = t
                side[trades] = position
                entry_price[trades] = open_price
                exit_price[trades] = fill
                stop_distance[trades] = open_distance
                reason[trades] = EXIT_SIGNAL
                trades += 1
                position = 0
            if position == 0 and wants_side != 0 and pending_stop > 0.0:
                if wants_side == LONG:
                    fill = bar[ASK_OPEN] + slippage
                    stop_price = fill - pending_stop
                    target_price = fill + pending_target
                else:
                    fill = bar[BID_OPEN] - slippage
                    stop_price = fill + pending_stop
                    target_price = fill - pending_target
                position = wants_side
                open_index = t
                open_price = fill
                open_distance = pending_stop
        pending = HOLD

        # 2. Protective orders inside this bar; the stop wins when both are touched.
        if position == LONG:
            hit_stop = False
            gapped = False
            fill = 0.0
            if bar[BID_OPEN] <= stop_price:
                hit_stop = True
                gapped = True
                fill = bar[BID_OPEN] - slippage
            elif bar[BID_LOW] <= stop_price:
                hit_stop = True
                fill = stop_price - slippage
            if hit_stop or bar[BID_HIGH] >= target_price:
                if not hit_stop:
                    fill = target_price
                entry_index[trades] = open_index
                exit_index[trades] = t
                side[trades] = LONG
                entry_price[trades] = open_price
                exit_price[trades] = fill
                stop_distance[trades] = open_distance
                if hit_stop:
                    reason[trades] = EXIT_STOP_GAP if gapped else EXIT_STOP
                else:
                    reason[trades] = EXIT_TARGET
                trades += 1
                position = 0
        elif position == SHORT:
            hit_stop = False
            gapped = False
            fill = 0.0
            if bar[ASK_OPEN] >= stop_price:
                hit_stop = True
                gapped = True
                fill = bar[ASK_OPEN] + slippage
            elif bar[ASK_HIGH] >= stop_price:
                hit_stop = True
                fill = stop_price + slippage
            if hit_stop or bar[ASK_LOW] <= target_price:
                if not hit_stop:
                    fill = target_price
                entry_index[trades] = open_index
                exit_index[trades] = t
                side[trades] = SHORT
                entry_price[trades] = open_price
                exit_price[trades] = fill
                stop_distance[trades] = open_distance
                if hit_stop:
                    reason[trades] = EXIT_STOP_GAP if gapped else EXIT_STOP
                else:
                    reason[trades] = EXIT_TARGET
                trades += 1
                position = 0

        # 3. The strategy reads the closed bar and decides for the next open.
        intent, stop_d, target_d = step(params, state, bar, position)
        if intent != HOLD:
            pending = intent
            pending_stop = stop_d
            pending_target = target_d

    if position != 0:
        last = n - 1
        if position == LONG:
            fill = bars[last, BID_CLOSE] - slippage
        else:
            fill = bars[last, ASK_CLOSE] + slippage
        entry_index[trades] = open_index
        exit_index[trades] = last
        side[trades] = position
        entry_price[trades] = open_price
        exit_price[trades] = fill
        stop_distance[trades] = open_distance
        reason[trades] = EXIT_END
        trades += 1

    return (
        entry_index[:trades], exit_index[:trades], side[:trades], entry_price[:trades],
        exit_price[:trades], stop_distance[:trades], reason[:trades],
    )


@njit(cache=True)
def net_pips(side, entry_price, exit_price, commission_price, pip_size):
    """Per-trade result in pips after a round-trip commission expressed in price units."""
    count = side.shape[0]
    result = np.empty(count, dtype=np.float64)
    for i in range(count):
        result[i] = ((exit_price[i] - entry_price[i]) * side[i] - commission_price) / pip_size
    return result

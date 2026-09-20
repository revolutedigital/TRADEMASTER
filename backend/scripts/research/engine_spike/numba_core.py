"""Spike: a compiled event-driven FX simulator core, to measure speed and prove correctness.

The lab must run hundreds of shuffled copies of years of M1 data across several strategies,
which is impossible at the ~7,000 bars per second of the pandas simulator. This core keeps
the whole state machine in one numba loop: quotes in, trades out. It implements the same
semantics as the research simulator (signal read at a bar close and acted on at the next open,
buy at the ask and sell at the bid, slippage against the trader, stop before target on a bar
that touches both, a stop crossed by a gap filled at the gapped open) around a deliberately
simple streaming strategy (EMA crossover with an ATR stop and a fixed reward-to-risk target),
so that its output can be compared trade by trade with an independent implementation.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

LONG = 1
SHORT = -1

EXIT_SIGNAL = 1
EXIT_STOP = 2
EXIT_STOP_GAP = 3
EXIT_TARGET = 4
EXIT_END = 5


@njit(cache=True)
def simulate(  # noqa: PLR0913, PLR0912, PLR0915
    bid_open, bid_high, bid_low, bid_close,
    ask_open, ask_high, ask_low, ask_close,
    fast_span, slow_span, atr_period, stop_atr, reward_risk,
    slippage, commission_price, warmup,
):
    """Run one configuration; return per-trade arrays and the number of trades.

    Prices are in price units. `slippage` and `commission_price` are per side and per round
    trip respectively, also in price units (pips times the pip size).
    """
    n = bid_open.shape[0]
    capacity = 2 * n + 2  # a bar can hold a signal exit and the stop of the trade that replaced it
    entry_index = np.empty(capacity, dtype=np.int64)
    exit_index = np.empty(capacity, dtype=np.int64)
    side = np.empty(capacity, dtype=np.int8)
    entry_price = np.empty(capacity, dtype=np.float64)
    exit_price = np.empty(capacity, dtype=np.float64)
    stop_distance = np.empty(capacity, dtype=np.float64)
    reason = np.empty(capacity, dtype=np.int8)
    trades = 0

    alpha_fast = 2.0 / (fast_span + 1.0)
    alpha_slow = 2.0 / (slow_span + 1.0)
    alpha_atr = 1.0 / atr_period

    fast = 0.0
    slow = 0.0
    atr = 0.0
    previous_close = 0.0
    previous_fast = 0.0
    previous_slow = 0.0

    position = 0
    open_entry_index = 0
    open_entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    open_stop_distance = 0.0
    pending = 0
    prior_atr = 0.0

    for t in range(n):
        # 1. Act on the signal read at the previous close, at this bar's open.
        if pending != 0 and t > 0:
            if position != 0 and position != pending:
                if position == LONG:
                    fill = bid_open[t] - slippage
                else:
                    fill = ask_open[t] + slippage
                entry_index[trades] = open_entry_index
                exit_index[trades] = t
                side[trades] = position
                entry_price[trades] = open_entry_price
                exit_price[trades] = fill
                stop_distance[trades] = open_stop_distance
                reason[trades] = EXIT_SIGNAL
                trades += 1
                position = 0
            if position == 0 and prior_atr > 0.0:
                distance = stop_atr * prior_atr
                if pending == LONG:
                    fill = ask_open[t] + slippage
                    stop_price = fill - distance
                    target_price = fill + reward_risk * distance
                else:
                    fill = bid_open[t] - slippage
                    stop_price = fill + distance
                    target_price = fill - reward_risk * distance
                position = pending
                open_entry_index = t
                open_entry_price = fill
                open_stop_distance = distance
        pending = 0

        # 2. Protective orders inside this bar; the stop wins when both are touched.
        if position == LONG:
            hit_stop = False
            gapped = False
            if bid_open[t] <= stop_price:
                hit_stop = True
                gapped = True
                fill = bid_open[t] - slippage
            elif bid_low[t] <= stop_price:
                hit_stop = True
                fill = stop_price - slippage
            if hit_stop or bid_high[t] >= target_price:
                if not hit_stop:
                    fill = target_price
                entry_index[trades] = open_entry_index
                exit_index[trades] = t
                side[trades] = LONG
                entry_price[trades] = open_entry_price
                exit_price[trades] = fill
                stop_distance[trades] = open_stop_distance
                if hit_stop:
                    reason[trades] = EXIT_STOP_GAP if gapped else EXIT_STOP
                else:
                    reason[trades] = EXIT_TARGET
                trades += 1
                position = 0
        elif position == SHORT:
            hit_stop = False
            gapped = False
            if ask_open[t] >= stop_price:
                hit_stop = True
                gapped = True
                fill = ask_open[t] + slippage
            elif ask_high[t] >= stop_price:
                hit_stop = True
                fill = stop_price + slippage
            if hit_stop or ask_low[t] <= target_price:
                if not hit_stop:
                    fill = target_price
                entry_index[trades] = open_entry_index
                exit_index[trades] = t
                side[trades] = SHORT
                entry_price[trades] = open_entry_price
                exit_price[trades] = fill
                stop_distance[trades] = open_stop_distance
                if hit_stop:
                    reason[trades] = EXIT_STOP_GAP if gapped else EXIT_STOP
                else:
                    reason[trades] = EXIT_TARGET
                trades += 1
                position = 0

        # 3. Update the streaming indicators with this bar's close and read the signal.
        mid_close = 0.5 * (bid_close[t] + ask_close[t])
        mid_high = 0.5 * (bid_high[t] + ask_high[t])
        mid_low = 0.5 * (bid_low[t] + ask_low[t])
        if t == 0:
            # Same seeding as pandas ewm(adjust=False), so a reference can match exactly.
            previous_fast = mid_close
            previous_slow = mid_close
            fast = mid_close
            slow = mid_close
            atr = mid_high - mid_low
        else:
            previous_fast = fast
            previous_slow = slow
            fast = fast + alpha_fast * (mid_close - fast)
            slow = slow + alpha_slow * (mid_close - slow)
            true_range = max(
                mid_high - mid_low, abs(mid_high - previous_close), abs(mid_low - previous_close)
            )
            atr = atr + alpha_atr * (true_range - atr)
        previous_close = mid_close
        prior_atr = atr

        if t >= warmup:
            if previous_fast <= previous_slow and fast > slow:
                pending = LONG
            elif previous_fast >= previous_slow and fast < slow:
                pending = SHORT

    if position != 0:
        last = n - 1
        if position == LONG:
            fill = bid_close[last] - slippage
        else:
            fill = ask_close[last] + slippage
        entry_index[trades] = open_entry_index
        exit_index[trades] = last
        side[trades] = position
        entry_price[trades] = open_entry_price
        exit_price[trades] = fill
        stop_distance[trades] = open_stop_distance
        reason[trades] = EXIT_END
        trades += 1

    return (
        entry_index[:trades], exit_index[:trades], side[:trades], entry_price[:trades],
        exit_price[:trades], stop_distance[:trades], reason[:trades],
    )


@njit(cache=True)
def summarize(side, entry_price, exit_price, commission_price, pip):
    """Net result in pips of a set of trades: (count, total net pips, wins)."""
    count = side.shape[0]
    total = 0.0
    wins = 0
    for i in range(count):
        gross = (exit_price[i] - entry_price[i]) * side[i]
        net = (gross - commission_price) / pip
        total += net
        if net > 0.0:
            wins += 1
    return count, total, wins


@njit(parallel=True, cache=True)
def sweep(  # noqa: PLR0913
    bid_open, bid_high, bid_low, bid_close,
    ask_open, ask_high, ask_low, ask_close,
    fast_spans, slow_spans, atr_period, stop_atr, reward_risk,
    slippage, commission_price, pip, warmup,
):
    """Run many configurations in parallel across cores; return count, net pips and wins."""
    configs = fast_spans.shape[0]
    counts = np.zeros(configs, dtype=np.int64)
    totals = np.zeros(configs, dtype=np.float64)
    wins = np.zeros(configs, dtype=np.int64)
    for c in prange(configs):
        _, _, side, entry_price, exit_price, _, _ = simulate(
            bid_open, bid_high, bid_low, bid_close,
            ask_open, ask_high, ask_low, ask_close,
            fast_spans[c], slow_spans[c], atr_period, stop_atr, reward_risk,
            slippage, commission_price, warmup,
        )
        counts[c], totals[c], wins[c] = summarize(side, entry_price, exit_price, commission_price, pip)
    return counts, totals, wins

"""The streaming strategy contract shared by the research lab and the live runner.

A strategy is a pair of compiled functions over flat arrays, so the very same code decides in a
backtest that replays years of bars in a compiled loop and in the live runner that feeds it one
bar at a time from Python. There is no second implementation to drift from the first.

    init(params, state)                          -> None      (resets the state)
    step(params, state, bar, position)           -> (intent, stop_distance, target_distance)

`step` is called once per closed bar with the bar's bid and ask OHLC, the current position
(1 long, -1 short, 0 flat), and returns what to do at the next open together with the distances,
in price units, of the protective stop and the profit target for a new entry. It may only use the
bars it has been given; a test in this repository proves that changing the future never changes
a decision already made.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

HOLD = 0
ENTER_LONG = 1
ENTER_SHORT = -1
EXIT = 2

FLAT = 0
LONG = 1
SHORT = -1

BAR_TIME = 0  # seconds since the epoch, UTC
BID_OPEN, BID_HIGH, BID_LOW, BID_CLOSE = 1, 2, 3, 4
ASK_OPEN, ASK_HIGH, ASK_LOW, ASK_CLOSE = 5, 6, 7, 8
BAR_WIDTH = 9

BAR_COLUMNS = (
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
)


def bars_to_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Pack a UTC-indexed bid/ask frame into the (n, 9) matrix the strategies consume."""
    if frame.index.tz is None:
        raise ValueError("the frame must be indexed by timezone-aware timestamps")
    matrix = np.empty((len(frame), BAR_WIDTH), dtype=np.float64)
    # Divide by a timedelta instead of using the raw integer, whose unit depends on the pandas version.
    matrix[:, BAR_TIME] = (frame.index - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    for offset, column in enumerate(BAR_COLUMNS, start=1):
        matrix[:, offset] = frame[column].to_numpy(dtype=np.float64)
    return matrix


# ---------------------------------------------------------------------------
# Reference strategy: EMA crossover with an ATR stop and a fixed reward-to-risk target
# ---------------------------------------------------------------------------

EMA_CROSS_STATE_SIZE = 5
_FAST, _SLOW, _ATR, _PREVIOUS_CLOSE, _BARS_SEEN = 0, 1, 2, 3, 4


def ema_cross_params(
    *, fast_span: int, slow_span: int, atr_period: int, stop_atr: float, reward_risk: float, warmup: int
) -> np.ndarray:
    if not 1 <= fast_span < slow_span:
        raise ValueError("fast_span must be at least 1 and below slow_span")
    if atr_period < 1 or stop_atr <= 0 or reward_risk <= 0 or warmup < 1:
        raise ValueError("atr_period, stop_atr, reward_risk and warmup must be positive")
    return np.array(
        [fast_span, slow_span, atr_period, stop_atr, reward_risk, warmup], dtype=np.float64
    )


@njit(cache=True)
def ema_cross_init(params, state):
    for i in range(state.shape[0]):
        state[i] = 0.0


@njit(cache=True)
def ema_cross_step(params, state, bar, position):
    fast_span = params[0]
    slow_span = params[1]
    atr_period = params[2]
    stop_atr = params[3]
    reward_risk = params[4]
    warmup = params[5]

    mid_close = 0.5 * (bar[BID_CLOSE] + bar[ASK_CLOSE])
    mid_high = 0.5 * (bar[BID_HIGH] + bar[ASK_HIGH])
    mid_low = 0.5 * (bar[BID_LOW] + bar[ASK_LOW])
    seen = state[_BARS_SEEN]

    if seen == 0:
        previous_fast = mid_close
        previous_slow = mid_close
        fast = mid_close
        slow = mid_close
        atr = mid_high - mid_low
    else:
        previous_fast = state[_FAST]
        previous_slow = state[_SLOW]
        fast = previous_fast + 2.0 / (fast_span + 1.0) * (mid_close - previous_fast)
        slow = previous_slow + 2.0 / (slow_span + 1.0) * (mid_close - previous_slow)
        previous_close = state[_PREVIOUS_CLOSE]
        true_range = max(
            mid_high - mid_low, abs(mid_high - previous_close), abs(mid_low - previous_close)
        )
        atr = state[_ATR] + (true_range - state[_ATR]) / atr_period

    state[_FAST] = fast
    state[_SLOW] = slow
    state[_ATR] = atr
    state[_PREVIOUS_CLOSE] = mid_close
    state[_BARS_SEEN] = seen + 1.0

    intent = HOLD
    if seen >= warmup:
        if previous_fast <= previous_slow and fast > slow and position != LONG:
            intent = ENTER_LONG
        elif previous_fast >= previous_slow and fast < slow and position != SHORT:
            intent = ENTER_SHORT
    stop_distance = 0.0
    target_distance = 0.0
    if intent != HOLD:
        stop_distance = stop_atr * atr
        target_distance = reward_risk * stop_distance
    return intent, stop_distance, target_distance


# ---------------------------------------------------------------------------
# Running any strategy over a matrix of bars
# ---------------------------------------------------------------------------


@njit  # not cached: a cached specialisation over function arguments can fail to reload
def run_batch(step, init, params, state_size, bars, positions):
    """Replay every bar through a strategy in one compiled loop; used by the lab and by tests."""
    count = bars.shape[0]
    state = np.zeros(state_size, dtype=np.float64)
    init(params, state)
    intents = np.zeros(count, dtype=np.int8)
    stops = np.zeros(count, dtype=np.float64)
    targets = np.zeros(count, dtype=np.float64)
    for t in range(count):
        intent, stop_distance, target_distance = step(params, state, bars[t], positions[t])
        intents[t] = intent
        stops[t] = stop_distance
        targets[t] = target_distance
    return intents, stops, targets


class StreamingRunner:
    """Feeds one closed bar at a time to a strategy, the way the live runner does."""

    def __init__(self, step, init, params: np.ndarray, state_size: int) -> None:
        self._step = step
        self._params = np.ascontiguousarray(params, dtype=np.float64)
        self._state = np.zeros(state_size, dtype=np.float64)
        init(self._params, self._state)

    def on_bar(self, bar: np.ndarray, position: int = FLAT) -> tuple[int, float, float]:
        if bar.shape != (BAR_WIDTH,):
            raise ValueError(f"a bar must have {BAR_WIDTH} values, got shape {bar.shape}")
        intent, stop_distance, target_distance = self._step(
            self._params, self._state, np.ascontiguousarray(bar, dtype=np.float64), position
        )
        return int(intent), float(stop_distance), float(target_distance)

"""Shared helpers for the strategy-family tests: synthetic bars and the two contract checks.

Every family is tested three ways: against an independent pandas implementation of its rule, by
feeding it bar by bar and in batch (which must agree), and by changing the future and confirming
that no decision already taken moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.fx import strategy as fx
from app.fx.sim import core

PIP = 0.0001


def synthetic_frame(
    *,
    start: str,
    weeks: int,
    bar_seconds: int,
    seed: int,
    sigma_pips: float = 3.0,
    half_spread_pips: float = 0.2,
    base: float = 1.10,
    mean_reversion: float = 0.0,
    pip: float = PIP,
) -> pd.DataFrame:
    """Monday-to-Friday bid/ask bars of a random walk (or a mean-reverting one), UTC-indexed.

    The market is open from Monday 00:00 UTC to Friday 22:00 UTC; the Sunday session is left out to
    keep the calendar simple, which still crosses every 17:00 New York boundary during the week.
    """
    stamps = pd.date_range(start, periods=weeks * 7 * 86_400 // bar_seconds, freq=f"{bar_seconds}s", tz="UTC")
    open_market = (stamps.weekday < 4) | ((stamps.weekday == 4) & (stamps.hour < 22))
    stamps = stamps[open_market]
    rng = np.random.default_rng(seed)
    count = len(stamps)
    noise = rng.normal(0.0, sigma_pips * pip, count)
    close = np.empty(count)
    level = base
    for i in range(count):
        level += noise[i] - mean_reversion * (level - base)
        close[i] = level
    open_ = np.concatenate(([base], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.5 * sigma_pips * pip, count))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.5 * sigma_pips * pip, count))
    frame = pd.DataFrame(index=stamps)
    for side, sign in (("bid", -1), ("ask", 1)):
        half = half_spread_pips * pip
        frame[f"{side}_open"] = open_ + sign * half
        frame[f"{side}_high"] = high + sign * half
        frame[f"{side}_low"] = low + sign * half
        frame[f"{side}_close"] = close + sign * half
    return frame


def mid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Mid open, high, low and close of a bid/ask frame."""
    return pd.DataFrame(
        {
            name: 0.5 * (frame[f"bid_{name}"] + frame[f"ask_{name}"])
            for name in ("open", "high", "low", "close")
        }
    )


def decisions(step, init, params, state_size, matrix, position: int = fx.FLAT):
    """Intents, stop distances and target distances of a batch replay with a constant position."""
    return fx.run_batch(
        step, init, params, state_size, matrix, np.full(matrix.shape[0], position, dtype=np.int64)
    )


def assert_streaming_equals_batch(step, init, params, state_size, matrix) -> None:
    intents, stops, targets = decisions(step, init, params, state_size, matrix)
    runner = fx.StreamingRunner(step, init, params, state_size)
    streamed = [runner.on_bar(bar) for bar in matrix]
    assert [s[0] for s in streamed] == intents.tolist()
    assert np.array_equal([s[1] for s in streamed], stops)
    assert np.array_equal([s[2] for s in streamed], targets)


def assert_the_future_never_changes_the_past(step, init, params, state_size, matrix) -> None:
    """A decision at bar t depends only on bars up to t: cutting or rewriting what follows changes nothing."""
    full = decisions(step, init, params, state_size, matrix)
    for cut in np.linspace(len(matrix) // 5, len(matrix) - 1, 6).astype(int):
        prefix = decisions(step, init, params, state_size, matrix[: cut + 1])
        for whole, part in zip(full, prefix, strict=True):
            assert np.array_equal(whole[: cut + 1], part), cut
        altered = matrix.copy()
        altered[cut + 1 :, fx.BID_OPEN : fx.ASK_CLOSE + 1] *= 1.05
        rewritten = decisions(step, init, params, state_size, altered)
        for whole, part in zip(full, rewritten, strict=True):
            assert np.array_equal(whole[: cut + 1], part[: cut + 1]), cut


def simulate(step, init, params, state_size, matrix):
    """Run the simulator with no slippage and return its trades as a frame of bar indices and prices."""
    entry, exit_, side, entry_price, exit_price, stop_distance, reason = core.run_simulation(
        step, init, params, state_size, matrix, 0.0
    )
    return pd.DataFrame(
        {
            "entry_index": entry, "exit_index": exit_, "side": side, "entry_price": entry_price,
            "exit_price": exit_price, "stop_distance": stop_distance, "reason": reason,
        }
    )

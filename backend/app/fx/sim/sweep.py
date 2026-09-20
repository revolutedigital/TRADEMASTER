"""Parallel parameter sweeps over the compiled simulator, summarized in pips and R."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from app.fx.sim.core import simulate


@njit(cache=True)
def summarize(side, entry_price, exit_price, stop_distance, commission_price, pip_size):
    """Count, total net pips, total R, and wins of a set of trades."""
    count = side.shape[0]
    total_pips = 0.0
    total_r = 0.0
    wins = 0
    for i in range(count):
        net = ((exit_price[i] - entry_price[i]) * side[i] - commission_price) / pip_size
        total_pips += net
        total_r += net * pip_size / stop_distance[i]
        if net > 0.0:
            wins += 1
    return count, total_pips, total_r, wins


@njit(parallel=True, cache=True)
def sweep(step, init, params_matrix, state_size, bars, slippage, commission_price, pip_size):
    """Run every row of `params_matrix` as one configuration, spread across all cores."""
    configs = params_matrix.shape[0]
    counts = np.zeros(configs, dtype=np.int64)
    pips = np.zeros(configs, dtype=np.float64)
    r_multiples = np.zeros(configs, dtype=np.float64)
    wins = np.zeros(configs, dtype=np.int64)
    for c in prange(configs):
        _, _, side, entry_price, exit_price, stop_distance, _ = simulate(
            step, init, params_matrix[c], state_size, bars, slippage
        )
        counts[c], pips[c], r_multiples[c], wins[c] = summarize(
            side, entry_price, exit_price, stop_distance, commission_price, pip_size
        )
    return counts, pips, r_multiples, wins

"""Statistics of the fast-strategy criterion, exactly as pre-registered.

Key A of the criterion is a stationary bootstrap of FX days with the maximum t over the whole
family of configurations (Westfall-Young / White's reality check): every configuration is centred
on its own mean, all of them are resampled with the same days, and the adjusted p-value of a
configuration is how often the best t of the family in a resample reaches its real t. Key B
(the placebo) compares the real t with the best t of the family on data with no edge by
construction; `placebo_p_value` is that comparison. `judge` applies the approval rules.

Everything works on daily tables: for each FX day and configuration, the sum of R of the trades
that entered that day (`sums`) and their count (`counts`). A trade's t is a ratio estimator with
a standard error that is robust to trades of the same day being correlated.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

SIGNIFICANCE = 0.05
MIN_TRADES_DISCOVERY = 300
MIN_TRADES_CONFIRMATION = 100
MIN_PAIR_SHARE = 0.6
MAX_MDE_R = 0.15
MDE_STANDARD_ERRORS = 3.4  # one-sided 0.5% significance and 80% power
BOOTSTRAP_DRAWS = 5_000
MEAN_BLOCK_DAYS = 10


def daily_tables(
    day_index: np.ndarray, r: np.ndarray, config_index: np.ndarray, n_days: int, n_configs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per (day, configuration) sum of R and number of trades from trade-level arrays."""
    flat = day_index.astype(np.int64) * n_configs + config_index.astype(np.int64)
    size = n_days * n_configs
    sums = np.bincount(flat, weights=r, minlength=size).reshape(n_days, n_configs)
    counts = np.bincount(flat, minlength=size).reshape(n_days, n_configs).astype(np.float64)
    return sums, counts


def cluster_t(sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean R, its cluster-robust standard error and t, per configuration."""
    total = counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums.sum(axis=0) / total
        residual = sums - mean * counts
        error = np.sqrt((residual**2).sum(axis=0)) / total
        t = mean / error
    return mean, error, t


@njit(cache=True)
def _stationary_weights(n_days, draws, restart, seed):
    np.random.seed(seed)
    weights = np.zeros((draws, n_days), dtype=np.float64)
    for draw in range(draws):
        day = np.random.randint(0, n_days)
        for _ in range(n_days):
            weights[draw, day] += 1.0
            if np.random.random() < restart:
                day = np.random.randint(0, n_days)
            else:
                day = (day + 1) % n_days
    return weights


def bootstrap_adjusted_p_values(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    mean_block: float = MEAN_BLOCK_DAYS,
    seed: int = 20260920,
) -> tuple[np.ndarray, np.ndarray]:
    """Real t of each configuration and its max-t adjusted p-value (key A)."""
    n_days = sums.shape[0]
    if n_days < 2 or mean_block < 1:
        raise ValueError("need at least two days and a mean block of at least one day")
    mean, _, t_real = cluster_t(sums, counts)
    weights = _stationary_weights(n_days, draws, 1.0 / mean_block, seed)
    total = weights @ counts
    with np.errstate(divide="ignore", invalid="ignore"):
        resampled_mean = (weights @ sums) / total
        squares = (
            weights @ sums**2
            - 2.0 * resampled_mean * (weights @ (sums * counts))
            + resampled_mean**2 * (weights @ counts**2)
        )
        error = np.sqrt(np.maximum(squares, 0.0)) / total
        t_star = (resampled_mean - mean) / error
    t_star = np.where(np.isfinite(t_star), t_star, -np.inf)
    best = t_star.max(axis=1)
    reachable = np.where(np.isfinite(t_real), t_real, -np.inf)  # no usable t can never pass
    reached = (best[:, None] >= reachable[None, :]).sum(axis=0)
    return t_real, (1.0 + reached) / (draws + 1.0)


def placebo_p_value(t_real: np.ndarray, null_best_t: np.ndarray) -> np.ndarray:
    """Adjusted p-value (key B): how often the best t on no-edge data reaches the real t."""
    reachable = np.where(np.isfinite(t_real), t_real, -np.inf)  # no usable t can never pass
    reached = (np.asarray(null_best_t)[:, None] >= reachable[None, :]).sum(axis=0)
    return (1.0 + reached) / (len(null_best_t) + 1.0)


def minimum_detectable_effect(error: np.ndarray) -> np.ndarray:
    return MDE_STANDARD_ERRORS * error


@dataclass(frozen=True)
class Verdict:
    approved: bool
    inconclusive: bool
    failed: tuple[str, ...]


def judge(
    *,
    p_bootstrap: float,
    p_placebo: float,
    mean_r: float,
    mean_r_stress: float,
    pair_share_positive: float,
    trades: int,
    mde: float,
    min_trades: int = MIN_TRADES_DISCOVERY,
    allow_inconclusive: bool = True,
) -> Verdict:
    """Apply the pre-registered approval rules; inconclusive is only for what was not approved."""
    failed = []
    if not p_bootstrap <= SIGNIFICANCE:
        failed.append("key A p-value above 0.05")
    if not p_placebo <= SIGNIFICANCE:
        failed.append("key B p-value above 0.05")
    if not mean_r > 0:
        failed.append("mean R not positive at base costs")
    if not pair_share_positive >= MIN_PAIR_SHARE:
        failed.append("fewer than 60% of pairs positive")
    if trades < min_trades:
        failed.append(f"fewer than {min_trades} trades")
    if not mean_r_stress > 0:
        failed.append("mean R not positive under stress")
    approved = not failed
    inconclusive = (
        allow_inconclusive and not approved and (trades < min_trades or not mde <= MAX_MDE_R)
    )
    return Verdict(approved, inconclusive, tuple(failed))

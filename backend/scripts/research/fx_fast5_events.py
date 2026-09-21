"""First-touch entry labels and deterministic breakeven/trailing outcomes for round 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from app.fx.instruments import Instrument
from scripts.research.fx_fast4_events import (
    MAX_EXIT_DELAY_MILLISECONDS,
    MAX_GAP_MILLISECONDS,
    EventCosts,
    _event_indices,
    _tick_arrays,
)

ACTIVATION_R = 0.5
INITIAL_STOP_R = -1.0
ACTIVATION_SECONDS = 600
MAX_HOLD_SECONDS = 6 * 60 * 60
TRAILING_DISTANCES_R = (0.5, 1.0, 1.5)
TOUCH_EPSILON_R = 1e-9

EXIT_INVALID = 0
EXIT_INITIAL_STOP = 1
EXIT_PRE_ACTIVATION_TIMEOUT = 2
EXIT_TRAILING_STOP = 3
EXIT_MAX_HOLD = 4


@dataclass(frozen=True)
class TrailingOutcome:
    result_r: float
    activated: bool
    exit_index: int
    exit_reason: int
    best_r: float
    holding_seconds: float


@njit(cache=True)
def _net_r(
    side,
    quote_index,
    bid,
    ask,
    pip,
    entry_price,
    risk_pips,
    commission_pips,
    slip_pips,
    spread_multiplier,
):
    mid = 0.5 * (bid[quote_index] + ask[quote_index])
    half_spread = 0.5 * (ask[quote_index] - bid[quote_index])
    if side == 1:
        exit_price = mid - spread_multiplier * half_spread - slip_pips * pip
        net_pips = (exit_price - entry_price) / pip - commission_pips
    else:
        exit_price = mid + spread_multiplier * half_spread + slip_pips * pip
        net_pips = (entry_price - exit_price) / pip - commission_pips
    return net_pips / risk_pips


@njit(cache=True)
def _entry_price(side, index, bid, ask, pip, slip_pips, spread_multiplier):
    mid = 0.5 * (bid[index] + ask[index])
    half_spread = 0.5 * (ask[index] - bid[index])
    if side == 1:
        return mid + spread_multiplier * half_spread + slip_pips * pip
    return mid - spread_multiplier * half_spread - slip_pips * pip


@njit(cache=True)
def _first_touch(  # noqa: PLR0913
    timestamps,
    bid,
    ask,
    block_id,
    decision_index,
    entry_index,
    deadline_index,
    side,
    pip,
    entry_price,
    risk_pips,
    commission_pips,
    slip_pips,
    spread_multiplier,
    activation_r,
):
    for quote_index in range(entry_index, deadline_index + 1):
        value = _net_r(
            side,
            quote_index,
            bid,
            ask,
            pip,
            entry_price,
            risk_pips,
            commission_pips,
            slip_pips,
            spread_multiplier,
        )
        if value <= INITIAL_STOP_R + TOUCH_EPSILON_R:
            return 0.0, timestamps[quote_index] - timestamps[decision_index]
        if value >= activation_r - TOUCH_EPSILON_R:
            return 1.0, timestamps[quote_index] - timestamps[decision_index]
    return 0.0, timestamps[deadline_index] - timestamps[decision_index]


@njit(cache=True)
def _entry_label_kernel(  # noqa: PLR0913, PLR0915
    timestamps,
    bid,
    ask,
    block_id,
    decision_indices,
    pip,
    range_pips,
    base_commission_pips,
    stress_commission_pips,
    base_slippage_pips,
    base_slippage_fraction,
    stress_slippage_pips,
    stress_slippage_fraction,
    stress_spread_multiplier,
    activation_r,
    activation_milliseconds,
):
    output = np.full((len(decision_indices), 11), np.nan)
    for event_position, decision_index in enumerate(decision_indices):
        entry_index = decision_index + 1
        if entry_index >= len(timestamps) or block_id[entry_index] != block_id[decision_index]:
            continue
        deadline = timestamps[decision_index] + activation_milliseconds
        deadline_index = np.searchsorted(timestamps, deadline)
        if deadline_index >= len(timestamps):
            continue
        if timestamps[deadline_index] > deadline + MAX_EXIT_DELAY_MILLISECONDS:
            continue
        if block_id[deadline_index] != block_id[decision_index]:
            continue

        recent_range = range_pips[event_position]
        base_slip = base_slippage_pips + base_slippage_fraction * recent_range
        stress_slip = stress_slippage_pips + stress_slippage_fraction * recent_range
        observed_spread_pips = (ask[entry_index] - bid[entry_index]) / pip
        complete_base_cost = observed_spread_pips + 2.0 * base_slip + base_commission_pips
        risk_pips = max(recent_range, 4.0 * complete_base_cost)
        output[event_position, 0] = entry_index
        output[event_position, 1] = risk_pips

        column = 2
        for side in (1, -1):
            base_entry = _entry_price(side, entry_index, bid, ask, pip, base_slip, 1.0)
            stress_entry = _entry_price(
                side,
                entry_index,
                bid,
                ask,
                pip,
                stress_slip,
                stress_spread_multiplier,
            )
            base_hit, base_touch = _first_touch(
                timestamps,
                bid,
                ask,
                block_id,
                decision_index,
                entry_index,
                deadline_index,
                side,
                pip,
                base_entry,
                risk_pips,
                base_commission_pips,
                base_slip,
                1.0,
                activation_r,
            )
            stress_hit, stress_touch = _first_touch(
                timestamps,
                bid,
                ask,
                block_id,
                decision_index,
                entry_index,
                deadline_index,
                side,
                pip,
                stress_entry,
                risk_pips,
                stress_commission_pips,
                stress_slip,
                stress_spread_multiplier,
                activation_r,
            )
            output[event_position, column] = base_hit
            output[event_position, column + 1] = stress_hit
            output[event_position, column + 2] = base_touch / 1_000.0
            output[event_position, column + 3] = stress_touch / 1_000.0
            column += 4
        output[event_position, 10] = timestamps[deadline_index] - timestamps[decision_index]
    return output


def build_entry_labels(
    cleaned_ticks: pd.DataFrame,
    features: pd.DataFrame,
    instrument: Instrument,
    costs: EventCosts,
    *,
    activation_r: float = ACTIVATION_R,
    activation_seconds: int = ACTIVATION_SECONDS,
) -> pd.DataFrame:
    """Label whether each side reaches activation before its initial stop."""
    if activation_r <= 0 or activation_seconds <= 0:
        raise ValueError("activation and deadline must be positive")
    timestamps, bid, ask = _tick_arrays(cleaned_ticks)
    _, block_id = _event_indices(timestamps)
    decision_indices = features["decision_index"].to_numpy(dtype=np.int64)
    values = _entry_label_kernel(
        timestamps,
        bid,
        ask,
        block_id,
        decision_indices,
        instrument.pip_size,
        features["mid_range_pips_256"].to_numpy(dtype=np.float64),
        costs.base_commission_pips,
        costs.stress_commission_pips,
        costs.base.slippage_pips,
        costs.base.slippage_range_fraction,
        costs.stress.slippage_pips,
        costs.stress.slippage_range_fraction,
        costs.stress.spread_multiplier,
        activation_r,
        activation_seconds * 1_000,
    )
    names = (
        "entry_index",
        "risk_pips",
        "long_hit_base",
        "long_hit_stress",
        "long_touch_seconds_base",
        "long_touch_seconds_stress",
        "short_hit_base",
        "short_hit_stress",
        "short_touch_seconds_base",
        "short_touch_seconds_stress",
        "observed_window_milliseconds",
    )
    return pd.DataFrame(values, columns=names, index=features.index)


@njit(cache=True)
def _simulate_trailing(  # noqa: PLR0913
    timestamps,
    bid,
    ask,
    block_id,
    decision_index,
    side,
    pip,
    entry_price,
    risk_pips,
    commission_pips,
    slip_pips,
    spread_multiplier,
    activation_r,
    trail_distance_r,
    activation_milliseconds,
    max_hold_milliseconds,
):
    entry_index = decision_index + 1
    if entry_index >= len(timestamps) or block_id[entry_index] != block_id[decision_index]:
        return np.nan, 0, -1, EXIT_INVALID, np.nan, np.nan
    activation_deadline = timestamps[decision_index] + activation_milliseconds
    hold_deadline = timestamps[decision_index] + max_hold_milliseconds
    activated = 0
    best_r = -np.inf
    previous_time = timestamps[entry_index]
    for quote_index in range(entry_index, len(timestamps)):
        if block_id[quote_index] != block_id[decision_index]:
            return np.nan, activated, -1, EXIT_INVALID, best_r, np.nan
        if (
            quote_index > entry_index
            and timestamps[quote_index] - previous_time > MAX_GAP_MILLISECONDS
        ):
            return np.nan, activated, -1, EXIT_INVALID, best_r, np.nan
        previous_time = timestamps[quote_index]
        value = _net_r(
            side,
            quote_index,
            bid,
            ask,
            pip,
            entry_price,
            risk_pips,
            commission_pips,
            slip_pips,
            spread_multiplier,
        )
        best_r = max(best_r, value)
        if activated == 0:
            if value <= INITIAL_STOP_R + TOUCH_EPSILON_R:
                return (
                    value,
                    0,
                    quote_index,
                    EXIT_INITIAL_STOP,
                    best_r,
                    (timestamps[quote_index] - timestamps[decision_index]) / 1_000.0,
                )
            if value >= activation_r - TOUCH_EPSILON_R:
                activated = 1
            elif timestamps[quote_index] >= activation_deadline:
                return (
                    value,
                    0,
                    quote_index,
                    EXIT_PRE_ACTIVATION_TIMEOUT,
                    best_r,
                    (timestamps[quote_index] - timestamps[decision_index]) / 1_000.0,
                )
        else:
            trailing_floor = max(0.0, best_r - trail_distance_r)
            if value <= trailing_floor + TOUCH_EPSILON_R:
                return (
                    value,
                    1,
                    quote_index,
                    EXIT_TRAILING_STOP,
                    best_r,
                    (timestamps[quote_index] - timestamps[decision_index]) / 1_000.0,
                )
        if timestamps[quote_index] >= hold_deadline:
            return (
                value,
                activated,
                quote_index,
                EXIT_MAX_HOLD,
                best_r,
                (timestamps[quote_index] - timestamps[decision_index]) / 1_000.0,
            )
    return np.nan, activated, -1, EXIT_INVALID, best_r, np.nan


def simulate_trailing_outcome(
    cleaned_ticks: pd.DataFrame,
    *,
    decision_index: int,
    side: int,
    instrument: Instrument,
    risk_pips: float,
    commission_pips: float = 0.0,
    slip_pips: float = 0.0,
    spread_multiplier: float = 1.0,
    activation_r: float = ACTIVATION_R,
    trail_distance_r: float = 0.5,
    activation_seconds: int = ACTIVATION_SECONDS,
    max_hold_seconds: int = MAX_HOLD_SECONDS,
) -> TrailingOutcome:
    """Reference wrapper for one path; batch policy simulation reuses the same kernel."""
    if side not in (-1, 1):
        raise ValueError("side must be 1 or -1")
    if risk_pips <= 0 or trail_distance_r <= 0:
        raise ValueError("risk and trailing distance must be positive")
    timestamps, bid, ask = _tick_arrays(cleaned_ticks)
    _, block_id = _event_indices(timestamps)
    entry_index = decision_index + 1
    if entry_index >= len(timestamps):
        return TrailingOutcome(float("nan"), False, -1, EXIT_INVALID, float("nan"), float("nan"))
    entry = _entry_price(
        side,
        entry_index,
        bid,
        ask,
        instrument.pip_size,
        slip_pips,
        spread_multiplier,
    )
    result = _simulate_trailing(
        timestamps,
        bid,
        ask,
        block_id,
        decision_index,
        side,
        instrument.pip_size,
        entry,
        risk_pips,
        commission_pips,
        slip_pips,
        spread_multiplier,
        activation_r,
        trail_distance_r,
        activation_seconds * 1_000,
        max_hold_seconds * 1_000,
    )
    return TrailingOutcome(
        result_r=float(result[0]),
        activated=bool(result[1]),
        exit_index=int(result[2]),
        exit_reason=int(result[3]),
        best_r=float(result[4]),
        holding_seconds=float(result[5]),
    )

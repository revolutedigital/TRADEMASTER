"""Causal quote-event features and executable tick outcomes for round 4.

The factory consumes top-of-book bid/ask updates only. It does not fit a model, select a threshold,
open a protected sample, touch the database, or send an order.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from app.fx.instruments import Instrument
from app.fx.sim.costs import CostScenario, FUSION_ZERO, STRESS

EVENT_WARMUP = 512
EVENT_STRIDE = 128
MAX_GAP_MILLISECONDS = 120_000
MAX_EXIT_DELAY_MILLISECONDS = 10_000
FEATURE_WINDOWS = (16, 64, 256)
HORIZONS_SECONDS = (30, 120, 600)
EPSILON = 1e-12

DIRECTIONAL_FEATURES = frozenset(
    {
        *(f"mid_return_pips_{window}" for window in FEATURE_WINDOWS),
        *(f"bid_direction_imbalance_{window}" for window in FEATURE_WINDOWS),
        *(f"ask_direction_imbalance_{window}" for window in FEATURE_WINDOWS),
        *(f"mid_direction_imbalance_{window}" for window in FEATURE_WINDOWS),
        "mid_run_direction",
    }
)


@dataclass(frozen=True)
class EventCosts:
    """Cost inputs with round-trip monetary commission already expressed in pips."""

    base_commission_pips: float
    stress_commission_pips: float
    base: CostScenario = FUSION_ZERO
    stress: CostScenario = STRESS

    def __post_init__(self) -> None:
        if self.base_commission_pips < 0 or self.stress_commission_pips < 0:
            raise ValueError("commission cannot be negative")


def clean_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    """Validate quotes and collapse exact consecutive duplicates without reordering them."""
    if not isinstance(ticks.index, pd.DatetimeIndex):
        raise ValueError("ticks must use a DatetimeIndex")
    if ticks.index.tz is None:
        raise ValueError("tick timestamps must be timezone-aware")
    if not {"bid", "ask"} <= set(ticks.columns):
        raise ValueError("ticks must contain bid and ask")
    frame = ticks[["bid", "ask"]].astype("float64").copy()
    if frame.empty:
        raise ValueError("no ticks")
    if frame[["bid", "ask"]].isna().any().any():
        raise ValueError("bid and ask must be finite")
    if not np.isfinite(frame[["bid", "ask"]].to_numpy()).all():
        raise ValueError("bid and ask must be finite")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("tick timestamps must be monotonic")
    if (frame["ask"] < frame["bid"]).any():
        raise ValueError("ask cannot be below bid")
    duplicate = frame.eq(frame.shift(1)).all(axis=1)
    return frame.loc[~duplicate]


def _tick_arrays(ticks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epoch_milliseconds = ticks.index.tz_convert("UTC").to_numpy(
        dtype="datetime64[ms]"
    ).astype(np.int64)
    return (
        np.asarray(epoch_milliseconds, dtype=np.int64),
        ticks["bid"].to_numpy(dtype=np.float64),
        ticks["ask"].to_numpy(dtype=np.float64),
    )


def _event_indices(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gap = np.concatenate(([True], np.diff(timestamps) > MAX_GAP_MILLISECONDS))
    block_starts = np.flatnonzero(gap)
    block_stops = np.concatenate((block_starts[1:], [len(timestamps)]))
    indices: list[np.ndarray] = []
    block_id = np.empty(len(timestamps), dtype=np.int32)
    for identifier, (start, stop) in enumerate(zip(block_starts, block_stops, strict=True)):
        block_id[start:stop] = identifier
        first = start + EVENT_WARMUP - 1
        if first < stop:
            indices.append(np.arange(first, stop, EVENT_STRIDE, dtype=np.int64))
    events = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    return events, block_id


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _window_sum(prefix: np.ndarray, indices: np.ndarray, window: int) -> np.ndarray:
    first_change = indices - window + 1
    return prefix[indices + 1] - prefix[first_change]


def _run_state(mid_change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.sign(mid_change).astype(np.int8)
    run_direction = np.zeros(len(direction), dtype=np.int8)
    run_length = np.zeros(len(direction), dtype=np.int16)
    for index in range(1, len(direction)):
        current = direction[index]
        if current == 0:
            continue
        run_direction[index] = current
        if current == run_direction[index - 1]:
            run_length[index] = min(64, run_length[index - 1] + 1)
        else:
            run_length[index] = 1
    return run_direction, run_length


def _rolling_extreme(values: np.ndarray, window: int, operation: str) -> np.ndarray:
    series = pd.Series(values)
    rolling = series.rolling(window + 1, min_periods=window + 1)
    if operation == "min":
        return rolling.min().to_numpy()
    if operation == "max":
        return rolling.max().to_numpy()
    raise ValueError(f"unknown rolling operation {operation!r}")


def _time_features(timestamps: np.ndarray) -> dict[str, np.ndarray]:
    utc = pd.to_datetime(timestamps, unit="ms", utc=True)
    second_of_week = (
        utc.dayofweek.to_numpy() * 86_400
        + utc.hour.to_numpy() * 3_600
        + utc.minute.to_numpy() * 60
        + utc.second.to_numpy()
    )
    angle = 2 * np.pi * second_of_week / (7 * 86_400)
    london = utc.tz_convert("Europe/London")
    new_york = utc.tz_convert("America/New_York")
    return {
        "week_sin": np.sin(angle),
        "week_cos": np.cos(angle),
        "london_session": ((london.hour >= 8) & (london.hour < 17)).astype(np.int8),
        "new_york_session": ((new_york.hour >= 8) & (new_york.hour < 17)).astype(np.int8),
    }


def build_feature_frame(ticks: pd.DataFrame, instrument: Instrument) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned ticks and causal features at each fixed quote-count event."""
    if instrument.pip_size <= 0:
        raise ValueError("pip size must be positive")
    cleaned = clean_ticks(ticks)
    timestamps, bid, ask = _tick_arrays(cleaned)
    events, _ = _event_indices(timestamps)
    mid = 0.5 * (bid + ask)
    spread_pips = (ask - bid) / instrument.pip_size
    bid_change = np.concatenate(([0.0], np.diff(bid)))
    ask_change = np.concatenate(([0.0], np.diff(ask)))
    mid_change_pips = np.concatenate(([0.0], np.diff(mid) / instrument.pip_size))
    bid_changed = (bid_change != 0).astype(np.float64)
    ask_changed = (ask_change != 0).astype(np.float64)
    both_changed = (bid_changed * ask_changed).astype(np.float64)
    run_direction, run_length = _run_state(mid_change_pips)

    feature: dict[str, np.ndarray] = {
        "decision_index": events,
        "spread_pips": spread_pips[events],
        "mid_run_direction": run_direction[events],
        "mid_run_length": run_length[events],
    }
    spread_prefix = _prefix(spread_pips)
    spread_square_prefix = _prefix(spread_pips**2)
    bid_direction_prefix = _prefix(np.sign(bid_change))
    ask_direction_prefix = _prefix(np.sign(ask_change))
    mid_direction_prefix = _prefix(np.sign(mid_change_pips))
    bid_activity_prefix = _prefix(bid_changed)
    ask_activity_prefix = _prefix(ask_changed)
    joint_prefix = _prefix(both_changed)
    volatility_prefix = _prefix(mid_change_pips**2)

    for window in FEATURE_WINDOWS:
        duration = np.maximum((timestamps[events] - timestamps[events - window]) / 1_000, 0.001)
        feature[f"duration_seconds_{window}"] = duration
        feature[f"update_intensity_{window}"] = window / duration
        feature[f"mid_return_pips_{window}"] = (
            mid[events] - mid[events - window]
        ) / instrument.pip_size
        feature[f"realized_quote_volatility_{window}"] = np.sqrt(
            _window_sum(volatility_prefix, events, window)
        )
        feature[f"spread_change_pips_{window}"] = spread_pips[events] - spread_pips[events - window]
        bid_moves = _window_sum(bid_activity_prefix, events, window)
        ask_moves = _window_sum(ask_activity_prefix, events, window)
        feature[f"bid_direction_imbalance_{window}"] = _window_sum(
            bid_direction_prefix, events, window
        ) / np.maximum(bid_moves, 1.0)
        feature[f"ask_direction_imbalance_{window}"] = _window_sum(
            ask_direction_prefix, events, window
        ) / np.maximum(ask_moves, 1.0)
        mid_moves = _window_sum(_prefix((mid_change_pips != 0).astype(float)), events, window)
        feature[f"mid_direction_imbalance_{window}"] = _window_sum(
            mid_direction_prefix, events, window
        ) / np.maximum(mid_moves, 1.0)
        total_activity = bid_moves + ask_moves
        feature[f"bid_ask_activity_imbalance_{window}"] = (
            bid_moves - ask_moves
        ) / np.maximum(total_activity, 1.0)
        feature[f"joint_update_fraction_{window}"] = _window_sum(
            joint_prefix, events, window
        ) / window
        if window in (64, 256):
            count = float(window)
            spread_sum = _window_sum(spread_prefix, events, window)
            spread_square_sum = _window_sum(spread_square_prefix, events, window)
            spread_mean = spread_sum / count
            spread_variance = np.maximum(spread_square_sum / count - spread_mean**2, 0.0)
            spread_std = np.sqrt(spread_variance)
            feature[f"spread_mean_{window}"] = spread_mean
            feature[f"spread_std_{window}"] = spread_std
            feature[f"spread_zscore_{window}"] = (
                spread_pips[events] - spread_mean
            ) / np.maximum(spread_std, EPSILON)
            feature[f"spread_min_{window}"] = _rolling_extreme(
                spread_pips, window, "min"
            )[events]
            feature[f"spread_max_{window}"] = _rolling_extreme(
                spread_pips, window, "max"
            )[events]
            mid_min = _rolling_extreme(mid, window, "min")[events]
            mid_max = _rolling_extreme(mid, window, "max")[events]
            feature[f"mid_range_pips_{window}"] = (mid_max - mid_min) / instrument.pip_size

    feature["intensity_acceleration_16_256"] = feature["update_intensity_16"] / np.maximum(
        feature["update_intensity_256"], EPSILON
    )
    feature.update(_time_features(timestamps[events]))
    index = pd.to_datetime(timestamps[events], unit="ms", utc=True)
    frame = pd.DataFrame(feature, index=pd.DatetimeIndex(index, name="decision_time"))
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return cleaned, frame


def directional_features(features: pd.DataFrame, side: int) -> pd.DataFrame:
    """Orient signed quote features so positive always points toward the proposed trade."""
    if side not in (-1, 1):
        raise ValueError("side must be 1 or -1")
    oriented = features.copy()
    for name in DIRECTIONAL_FEATURES & set(oriented.columns):
        oriented[name] = oriented[name] * side
    oriented["side"] = side
    return oriented


def _single_feature_row(
    timestamps: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    instrument: Instrument,
) -> dict[str, float | int]:
    """Reference streaming calculation for the newest quote in a 257-quote buffer."""
    index = len(timestamps) - 1
    mid = 0.5 * (bid + ask)
    spread = (ask - bid) / instrument.pip_size
    bid_change = np.concatenate(([0.0], np.diff(bid)))
    ask_change = np.concatenate(([0.0], np.diff(ask)))
    mid_change = np.concatenate(([0.0], np.diff(mid) / instrument.pip_size))
    row: dict[str, float | int] = {
        "spread_pips": spread[index],
    }
    nonzero = np.flatnonzero(mid_change[max(1, index - 63) : index + 1] != 0)
    if nonzero.size:
        direction = int(np.sign(mid_change[max(1, index - 63) + nonzero[-1]]))
        length = 0
        for value in mid_change[index : max(0, index - 64) : -1]:
            if int(np.sign(value)) != direction:
                break
            length += 1
    else:
        direction, length = 0, 0
    row["mid_run_direction"] = direction
    row["mid_run_length"] = length
    for window in FEATURE_WINDOWS:
        start = index - window
        changes = slice(start + 1, index + 1)
        duration = max((timestamps[index] - timestamps[start]) / 1_000, 0.001)
        bid_moved = bid_change[changes] != 0
        ask_moved = ask_change[changes] != 0
        mid_moved = mid_change[changes] != 0
        row[f"duration_seconds_{window}"] = duration
        row[f"update_intensity_{window}"] = window / duration
        row[f"mid_return_pips_{window}"] = (mid[index] - mid[start]) / instrument.pip_size
        row[f"realized_quote_volatility_{window}"] = float(
            np.sqrt(np.sum(mid_change[changes] ** 2))
        )
        row[f"spread_change_pips_{window}"] = spread[index] - spread[start]
        row[f"bid_direction_imbalance_{window}"] = float(
            np.sign(bid_change[changes]).sum() / max(bid_moved.sum(), 1)
        )
        row[f"ask_direction_imbalance_{window}"] = float(
            np.sign(ask_change[changes]).sum() / max(ask_moved.sum(), 1)
        )
        row[f"mid_direction_imbalance_{window}"] = float(
            np.sign(mid_change[changes]).sum() / max(mid_moved.sum(), 1)
        )
        total_activity = bid_moved.sum() + ask_moved.sum()
        row[f"bid_ask_activity_imbalance_{window}"] = float(
            (bid_moved.sum() - ask_moved.sum()) / max(total_activity, 1)
        )
        row[f"joint_update_fraction_{window}"] = float(np.mean(bid_moved & ask_moved))
        if window in (64, 256):
            observed = spread[index - window + 1 : index + 1]
            spread_mean = float(observed.mean())
            spread_std = float(observed.std(ddof=0))
            row[f"spread_mean_{window}"] = spread_mean
            row[f"spread_std_{window}"] = spread_std
            row[f"spread_zscore_{window}"] = (spread[index] - spread_mean) / max(
                spread_std, EPSILON
            )
            row[f"spread_min_{window}"] = float(spread[index - window : index + 1].min())
            row[f"spread_max_{window}"] = float(spread[index - window : index + 1].max())
            observed_mid = mid[index - window : index + 1]
            row[f"mid_range_pips_{window}"] = float(
                (observed_mid.max() - observed_mid.min()) / instrument.pip_size
            )
    row["intensity_acceleration_16_256"] = float(
        row["update_intensity_16"] / max(float(row["update_intensity_256"]), EPSILON)
    )
    time = _time_features(np.asarray([timestamps[index]], dtype=np.int64))
    row.update({name: values[0] for name, values in time.items()})
    return row


class FeatureState:
    """Streaming reference state used to prove parity with the batch factory."""

    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument
        self._quotes: deque[tuple[int, float, float]] = deque(maxlen=257)
        self._block_count = 0

    def push(self, timestamp: pd.Timestamp, bid: float, ask: float) -> dict[str, float | int] | None:
        if timestamp.tzinfo is None:
            raise ValueError("tick timestamps must be timezone-aware")
        if not np.isfinite([bid, ask]).all() or ask < bid:
            raise ValueError("invalid bid/ask quote")
        milliseconds = int(
            timestamp.tz_convert("UTC").to_datetime64().astype("datetime64[ms]").astype(np.int64)
        )
        if self._quotes:
            previous_time, previous_bid, previous_ask = self._quotes[-1]
            if milliseconds < previous_time:
                raise ValueError("tick timestamps must be monotonic")
            if bid == previous_bid and ask == previous_ask:
                return None
            if milliseconds - previous_time > MAX_GAP_MILLISECONDS:
                self._quotes.clear()
                self._block_count = 0
        self._quotes.append((milliseconds, bid, ask))
        self._block_count += 1
        if self._block_count < EVENT_WARMUP:
            return None
        if (self._block_count - EVENT_WARMUP) % EVENT_STRIDE:
            return None
        quote = np.asarray(self._quotes, dtype=np.float64)
        return _single_feature_row(
            quote[:, 0].astype(np.int64), quote[:, 1], quote[:, 2], self.instrument
        )


@njit(cache=True)
def _outcome_kernel(  # noqa: PLR0913
    timestamps,
    bid,
    ask,
    block_id,
    decision_indices,
    horizons_milliseconds,
    pip,
    range_pips,
    base_commission_pips,
    stress_commission_pips,
    base_slippage_pips,
    base_slippage_fraction,
    stress_slippage_pips,
    stress_slippage_fraction,
    stress_spread_multiplier,
):
    output = np.full((len(decision_indices), len(horizons_milliseconds), 15), np.nan)
    for event_position, decision_index in enumerate(decision_indices):
        entry_index = decision_index + 1
        if entry_index >= len(timestamps) or block_id[entry_index] != block_id[decision_index]:
            continue
        recent_range = range_pips[event_position]
        base_slip = base_slippage_pips + base_slippage_fraction * recent_range
        stress_slip = stress_slippage_pips + stress_slippage_fraction * recent_range
        entry_mid = 0.5 * (bid[entry_index] + ask[entry_index])
        entry_half_spread = 0.5 * (ask[entry_index] - bid[entry_index])
        stress_entry_bid = entry_mid - stress_spread_multiplier * entry_half_spread
        stress_entry_ask = entry_mid + stress_spread_multiplier * entry_half_spread
        observed_spread_pips = (ask[entry_index] - bid[entry_index]) / pip
        complete_base_cost = observed_spread_pips + 2.0 * base_slip + base_commission_pips
        risk_pips = max(recent_range, 4.0 * complete_base_cost)
        for horizon_position, horizon in enumerate(horizons_milliseconds):
            deadline = timestamps[decision_index] + horizon
            exit_index = np.searchsorted(timestamps, deadline)
            if exit_index >= len(timestamps):
                continue
            if timestamps[exit_index] > deadline + MAX_EXIT_DELAY_MILLISECONDS:
                continue
            if block_id[exit_index] != block_id[decision_index]:
                continue
            exit_mid = 0.5 * (bid[exit_index] + ask[exit_index])
            exit_half_spread = 0.5 * (ask[exit_index] - bid[exit_index])
            stress_exit_bid = exit_mid - stress_spread_multiplier * exit_half_spread
            stress_exit_ask = exit_mid + stress_spread_multiplier * exit_half_spread

            long_entry = ask[entry_index] + base_slip * pip
            long_exit = bid[exit_index] - base_slip * pip
            short_entry = bid[entry_index] - base_slip * pip
            short_exit = ask[exit_index] + base_slip * pip
            stress_long_entry = stress_entry_ask + stress_slip * pip
            stress_long_exit = stress_exit_bid - stress_slip * pip
            stress_short_entry = stress_entry_bid - stress_slip * pip
            stress_short_exit = stress_exit_ask + stress_slip * pip

            maximum_bid = -np.inf
            minimum_bid = np.inf
            maximum_ask = -np.inf
            minimum_ask = np.inf
            for quote_index in range(entry_index, exit_index + 1):
                maximum_bid = max(maximum_bid, bid[quote_index] - base_slip * pip)
                minimum_bid = min(minimum_bid, bid[quote_index] - base_slip * pip)
                maximum_ask = max(maximum_ask, ask[quote_index] + base_slip * pip)
                minimum_ask = min(minimum_ask, ask[quote_index] + base_slip * pip)

            long_base_pips = (long_exit - long_entry) / pip - base_commission_pips
            long_stress_pips = (
                (stress_long_exit - stress_long_entry) / pip - stress_commission_pips
            )
            short_base_pips = (short_entry - short_exit) / pip - base_commission_pips
            short_stress_pips = (
                (stress_short_entry - stress_short_exit) / pip - stress_commission_pips
            )
            output[event_position, horizon_position, 0] = entry_index
            output[event_position, horizon_position, 1] = exit_index
            output[event_position, horizon_position, 2] = risk_pips
            output[event_position, horizon_position, 3] = long_base_pips
            output[event_position, horizon_position, 4] = long_stress_pips
            output[event_position, horizon_position, 5] = long_base_pips / risk_pips
            output[event_position, horizon_position, 6] = long_stress_pips / risk_pips
            output[event_position, horizon_position, 7] = (
                (maximum_bid - long_entry) / pip - base_commission_pips
            ) / risk_pips
            output[event_position, horizon_position, 8] = (
                (long_entry - minimum_bid) / pip + base_commission_pips
            ) / risk_pips
            output[event_position, horizon_position, 9] = short_base_pips
            output[event_position, horizon_position, 10] = short_stress_pips
            output[event_position, horizon_position, 11] = short_base_pips / risk_pips
            output[event_position, horizon_position, 12] = short_stress_pips / risk_pips
            output[event_position, horizon_position, 13] = (
                (short_entry - minimum_ask) / pip - base_commission_pips
            ) / risk_pips
            output[event_position, horizon_position, 14] = (
                (maximum_ask - short_entry) / pip + base_commission_pips
            ) / risk_pips
    return output


def build_outcome_wide(
    cleaned_ticks: pd.DataFrame,
    features: pd.DataFrame,
    instrument: Instrument,
    costs: EventCosts,
    *,
    horizons_seconds: Iterable[int] = HORIZONS_SECONDS,
) -> pd.DataFrame:
    """Build executable terminal outcomes while preserving feature-row alignment."""
    horizons = tuple(int(value) for value in horizons_seconds)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must be positive seconds")
    timestamps, bid, ask = _tick_arrays(cleaned_ticks)
    _, block_id = _event_indices(timestamps)
    decision_indices = features["decision_index"].to_numpy(dtype=np.int64)
    range_pips = features["mid_range_pips_256"].to_numpy(dtype=np.float64)
    values = _outcome_kernel(
        timestamps,
        bid,
        ask,
        block_id,
        decision_indices,
        np.asarray(horizons, dtype=np.int64) * 1_000,
        instrument.pip_size,
        range_pips,
        costs.base_commission_pips,
        costs.stress_commission_pips,
        costs.base.slippage_pips,
        costs.base.slippage_range_fraction,
        costs.stress.slippage_pips,
        costs.stress.slippage_range_fraction,
        costs.stress.spread_multiplier,
    )
    names = (
        "entry_index",
        "exit_index",
        "risk_pips",
        "long_net_pips_base",
        "long_net_pips_stress",
        "long_terminal_r_base",
        "long_terminal_r_stress",
        "long_mfe_r_base",
        "long_mae_r_base",
        "short_net_pips_base",
        "short_net_pips_stress",
        "short_terminal_r_base",
        "short_terminal_r_stress",
        "short_mfe_r_base",
        "short_mae_r_base",
    )
    columns: dict[str, np.ndarray] = {}
    for horizon_position, horizon in enumerate(horizons):
        for field_position, name in enumerate(names):
            columns[f"h{horizon}_{name}"] = values[:, horizon_position, field_position]
    return pd.DataFrame(columns, index=features.index)

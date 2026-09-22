"""Exact ordered-trade path labels for microstructure experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from app.services.backtest.event_replay import OrderSide


class FirstTouch(StrEnum):
    STOP = "STOP"
    BREAKEVEN = "BREAKEVEN"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class PathLabelConfig:
    horizons_seconds: tuple[int, ...] = (120, 300)
    expected_cost_bps: float = 12.0
    stress_cost_bps: float = 24.0
    initial_stop_bps: float = 20.0

    def __post_init__(self) -> None:
        if not self.horizons_seconds or any(horizon <= 0 for horizon in self.horizons_seconds):
            raise ValueError("horizons must be positive")
        values = (
            self.expected_cost_bps,
            self.stress_cost_bps,
            self.initial_stop_bps,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("cost and stop thresholds must be finite and positive")
        if self.stress_cost_bps < self.expected_cost_bps:
            raise ValueError("stress cost cannot be below expected cost")


@dataclass(frozen=True)
class EventPathLabel:
    decision_time_ms: int
    entry_sequence_id: int
    entry_price: float
    side: OrderSide
    horizon_seconds: int
    event_count: int
    mfe_bps: float
    mae_bps: float
    terminal_bps: float
    first_touch: FirstTouch
    first_touch_time_ms: int
    first_touch_sequence_id: int | None
    expected_touch_time_ms: int | None
    expected_touch_sequence_id: int | None
    stress_touch_time_ms: int | None
    stress_touch_sequence_id: int | None
    stop_touch_time_ms: int | None
    stop_touch_sequence_id: int | None
    paid_expected_before_stop: bool
    paid_stress_before_stop: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PriceRangeIndex:
    """Segment tree supporting exact range extrema and first crossing index."""

    def __init__(self, prices: np.ndarray) -> None:
        values = np.asarray(prices, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("prices must be a non-empty one-dimensional array")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("prices must be finite and positive")
        size = 1
        while size < len(values):
            size *= 2
        self._length = len(values)
        self._size = size
        self._max_tree = np.full(size * 2, -np.inf, dtype=np.float64)
        self._min_tree = np.full(size * 2, np.inf, dtype=np.float64)
        self._max_tree[size : size + len(values)] = values
        self._min_tree[size : size + len(values)] = values
        level_start = size
        level_length = size
        while level_length > 1:
            parent_start = level_start // 2
            self._max_tree[parent_start:level_start] = np.maximum(
                self._max_tree[level_start : level_start + level_length : 2],
                self._max_tree[level_start + 1 : level_start + level_length : 2],
            )
            self._min_tree[parent_start:level_start] = np.minimum(
                self._min_tree[level_start : level_start + level_length : 2],
                self._min_tree[level_start + 1 : level_start + level_length : 2],
            )
            level_start = parent_start
            level_length //= 2

    def extrema(self, left: int, right: int) -> tuple[float, float]:
        self._validate_range(left, right)
        left += self._size
        right += self._size
        maximum = -np.inf
        minimum = np.inf
        while left < right:
            if left & 1:
                maximum = max(maximum, self._max_tree[left])
                minimum = min(minimum, self._min_tree[left])
                left += 1
            if right & 1:
                right -= 1
                maximum = max(maximum, self._max_tree[right])
                minimum = min(minimum, self._min_tree[right])
            left //= 2
            right //= 2
        return float(minimum), float(maximum)

    def first_at_least(self, left: int, right: int, threshold: float) -> int | None:
        self._validate_range(left, right)
        found = self._first_crossing(1, 0, self._size, left, right, threshold, True)
        return None if found >= self._length else found

    def first_at_most(self, left: int, right: int, threshold: float) -> int | None:
        self._validate_range(left, right)
        found = self._first_crossing(1, 0, self._size, left, right, threshold, False)
        return None if found >= self._length else found

    def _first_crossing(
        self,
        node: int,
        node_left: int,
        node_right: int,
        query_left: int,
        query_right: int,
        threshold: float,
        at_least: bool,
    ) -> int:
        outside = node_right <= query_left or query_right <= node_left
        cannot_cross = (
            self._max_tree[node] < threshold if at_least else self._min_tree[node] > threshold
        )
        if outside or cannot_cross:
            return self._length
        if node_right - node_left == 1:
            return node_left
        midpoint = (node_left + node_right) // 2
        first = self._first_crossing(
            node * 2,
            node_left,
            midpoint,
            query_left,
            query_right,
            threshold,
            at_least,
        )
        if first < self._length:
            return first
        return self._first_crossing(
            node * 2 + 1,
            midpoint,
            node_right,
            query_left,
            query_right,
            threshold,
            at_least,
        )

    def _validate_range(self, left: int, right: int) -> None:
        if left < 0 or right > self._length or left >= right:
            raise ValueError(f"invalid half-open range [{left}, {right})")


class EventPathLabeler:
    """Label causal decision timestamps from ordered exchange trade events."""

    def __init__(
        self,
        event_times_ms: np.ndarray,
        sequence_ids: np.ndarray,
        prices: np.ndarray,
        config: PathLabelConfig | None = None,
    ) -> None:
        self._times = np.asarray(event_times_ms, dtype=np.int64)
        self._sequences = np.asarray(sequence_ids, dtype=np.int64)
        self._prices = np.asarray(prices, dtype=np.float64)
        if not (len(self._times) == len(self._sequences) == len(self._prices)):
            raise ValueError("event arrays must have equal lengths")
        if len(self._times) < 2:
            raise ValueError("at least two events are required")
        if (np.diff(self._times) < 0).any():
            raise ValueError("event times must be monotonic")
        same_time = np.diff(self._times) == 0
        if (np.diff(self._sequences)[same_time] <= 0).any():
            raise ValueError("same-timestamp events must preserve sequence order")
        self._config = config or PathLabelConfig()
        self._range = PriceRangeIndex(self._prices)

    def label(self, decision_time_ms: int, side: OrderSide, horizon_seconds: int) -> EventPathLabel:
        if horizon_seconds not in self._config.horizons_seconds:
            raise ValueError("horizon is not pre-registered")
        entry_index = int(np.searchsorted(self._times, decision_time_ms, side="right") - 1)
        if entry_index < 0:
            raise ValueError("decision precedes the first market event")
        path_left = entry_index + 1
        path_right = int(
            np.searchsorted(
                self._times,
                decision_time_ms + horizon_seconds * 1000,
                side="right",
            )
        )
        if path_left >= path_right:
            raise ValueError("decision has no events inside its complete horizon")
        if self._times[-1] < decision_time_ms + horizon_seconds * 1000:
            raise ValueError("decision horizon extends beyond the loaded events")

        entry_price = float(self._prices[entry_index])
        minimum, maximum = self._range.extrema(path_left, path_right)
        if side == OrderSide.BUY:
            mfe_bps = math.log(maximum / entry_price) * 10_000
            mae_bps = math.log(minimum / entry_price) * 10_000
            terminal_bps = math.log(self._prices[path_right - 1] / entry_price) * 10_000
            expected_index = self._range.first_at_least(
                path_left,
                path_right,
                entry_price * math.exp(self._config.expected_cost_bps / 10_000),
            )
            stress_index = self._range.first_at_least(
                path_left,
                path_right,
                entry_price * math.exp(self._config.stress_cost_bps / 10_000),
            )
            stop_index = self._range.first_at_most(
                path_left,
                path_right,
                entry_price * math.exp(-self._config.initial_stop_bps / 10_000),
            )
        else:
            mfe_bps = math.log(entry_price / minimum) * 10_000
            mae_bps = math.log(entry_price / maximum) * 10_000
            terminal_bps = math.log(entry_price / self._prices[path_right - 1]) * 10_000
            expected_index = self._range.first_at_most(
                path_left,
                path_right,
                entry_price * math.exp(-self._config.expected_cost_bps / 10_000),
            )
            stress_index = self._range.first_at_most(
                path_left,
                path_right,
                entry_price * math.exp(-self._config.stress_cost_bps / 10_000),
            )
            stop_index = self._range.first_at_least(
                path_left,
                path_right,
                entry_price * math.exp(self._config.initial_stop_bps / 10_000),
            )

        first_touch, first_index = self._first_touch(expected_index, stop_index)
        first_touch_time = (
            decision_time_ms + horizon_seconds * 1000
            if first_index is None
            else int(self._times[first_index])
        )
        return EventPathLabel(
            decision_time_ms=decision_time_ms,
            entry_sequence_id=int(self._sequences[entry_index]),
            entry_price=entry_price,
            side=side,
            horizon_seconds=horizon_seconds,
            event_count=path_right - path_left,
            mfe_bps=mfe_bps,
            mae_bps=mae_bps,
            terminal_bps=terminal_bps,
            first_touch=first_touch,
            first_touch_time_ms=first_touch_time,
            first_touch_sequence_id=self._sequence(first_index),
            expected_touch_time_ms=self._time(expected_index),
            expected_touch_sequence_id=self._sequence(expected_index),
            stress_touch_time_ms=self._time(stress_index),
            stress_touch_sequence_id=self._sequence(stress_index),
            stop_touch_time_ms=self._time(stop_index),
            stop_touch_sequence_id=self._sequence(stop_index),
            paid_expected_before_stop=self._before_stop(expected_index, stop_index),
            paid_stress_before_stop=self._before_stop(stress_index, stop_index),
        )

    @staticmethod
    def _first_touch(
        expected_index: int | None, stop_index: int | None
    ) -> tuple[FirstTouch, int | None]:
        if expected_index is None and stop_index is None:
            return FirstTouch.TIMEOUT, None
        if stop_index is not None and (expected_index is None or stop_index < expected_index):
            return FirstTouch.STOP, stop_index
        return FirstTouch.BREAKEVEN, expected_index

    @staticmethod
    def _before_stop(touch_index: int | None, stop_index: int | None) -> bool:
        return touch_index is not None and (stop_index is None or touch_index < stop_index)

    def _time(self, index: int | None) -> int | None:
        return None if index is None else int(self._times[index])

    def _sequence(self, index: int | None) -> int | None:
        return None if index is None else int(self._sequences[index])

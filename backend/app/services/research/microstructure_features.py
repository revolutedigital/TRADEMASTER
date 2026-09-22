"""Causal microstructure features shared by offline and online research paths."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.schemas.microstructure import MarketEventType, MicrostructureEvent


TRADE_WINDOWS_SECONDS = (1, 5, 30, 60)
BASE_BOOK_FEATURE_COLUMNS = (
    "book_available",
    "book_update_age_ms",
    "spread_bps",
    "depth_imbalance",
    "microprice_displacement_bps",
)
BOOK_WINDOW_FEATURE_PREFIXES = (
    "book_event_count",
    "book_bid_replenishment_qty",
    "book_ask_replenishment_qty",
    "book_bid_liquidity_removed_qty",
    "book_ask_liquidity_removed_qty",
    "book_pressure_imbalance",
    "book_spread_widening_bps",
    "book_spread_recovery_bps",
    "book_depth_imbalance_change",
    "book_microprice_displacement_change_bps",
)
BOOK_FEATURE_COLUMNS = BASE_BOOK_FEATURE_COLUMNS + tuple(
    f"{prefix}_{window}s"
    for window in TRADE_WINDOWS_SECONDS
    for prefix in BOOK_WINDOW_FEATURE_PREFIXES
)


@dataclass(frozen=True)
class FeatureSnapshot:
    decision_time_ms: int
    values: dict[str, float]


class CausalMicrostructureFeatureEngine:
    """Incrementally expose only facts received at or before a decision time."""

    def __init__(self, windows_seconds: tuple[int, ...] = TRADE_WINDOWS_SECONDS) -> None:
        if not windows_seconds or any(window <= 0 for window in windows_seconds):
            raise ValueError("feature windows must be positive")
        self._windows = tuple(sorted(set(windows_seconds)))
        self._max_window_ms = self._windows[-1] * 1000
        self._trades: deque[tuple[int, float, float, float, float]] = deque()
        self._liquidations: deque[tuple[int, float, float, float, float]] = deque()
        self._last_event_time_ms: int | None = None
        self._book: tuple[int, float, float, float, float] | None = None
        self._book_events: deque[tuple[int, float, float, float, float]] = deque()
        self._last_mark_event_time_ms: int | None = None
        self._mark_price: float | None = None
        self._index_price: float | None = None
        self._funding_rate: float | None = None

    def consume(self, event: MicrostructureEvent) -> None:
        timestamp_ms = int(event.event_time.timestamp() * 1000)
        if self._last_event_time_ms is not None and timestamp_ms < self._last_event_time_ms:
            raise ValueError("feature events must be chronological")
        self._last_event_time_ms = timestamp_ms
        if event.event_type in {MarketEventType.AGG_TRADE, MarketEventType.TRADE}:
            if event.price is None or event.quantity is None or event.quantity <= 0:
                return
            aggressor_sign = -1.0 if event.is_buyer_maker else 1.0
            quote_quantity = event.quote_quantity or event.price * event.quantity
            self._trades.append(
                (
                    timestamp_ms,
                    event.price,
                    event.quantity,
                    quote_quantity,
                    aggressor_sign,
                )
            )
        elif event.event_type in {
            MarketEventType.BOOK_TICKER,
            MarketEventType.DEPTH,
        }:
            if all(
                value is not None
                for value in (
                    event.bid_price,
                    event.bid_quantity,
                    event.ask_price,
                    event.ask_quantity,
                )
            ):
                self._book = (
                    timestamp_ms,
                    float(event.bid_price),
                    float(event.bid_quantity),
                    float(event.ask_price),
                    float(event.ask_quantity),
                )
                self._book_events.append(self._book)
        elif event.event_type == MarketEventType.MARK_PRICE and event.price is not None:
            self._last_mark_event_time_ms = timestamp_ms
            self._mark_price = event.price
            payload = event.payload or {}
            self._index_price = _optional_float(payload.get("index_price"))
            self._funding_rate = _optional_float(payload.get("funding_rate"))
        elif event.event_type == MarketEventType.LIQUIDATION:
            if event.quantity is not None and event.quantity > 0:
                sign = 1.0 if event.side == "BUY" else -1.0
                notional = event.quantity * event.price if event.price is not None else 0.0
                self._liquidations.append(
                    (
                        timestamp_ms,
                        sign * event.quantity,
                        event.quantity,
                        sign * notional,
                        notional,
                    )
                )
        self._evict(timestamp_ms)

    def snapshot(self, decision_time_ms: int) -> FeatureSnapshot:
        if self._last_event_time_ms is not None and decision_time_ms < self._last_event_time_ms:
            raise ValueError("cannot snapshot before the latest consumed event")
        self._evict(decision_time_ms)
        values: dict[str, float] = {}
        for window in self._windows:
            cutoff = decision_time_ms - window * 1000
            trades = [trade for trade in self._trades if trade[0] >= cutoff]
            values.update(_trade_window_features(trades, window))
            liquidations = [
                liquidation for liquidation in self._liquidations if liquidation[0] >= cutoff
            ]
            values[f"liquidation_count_{window}s"] = float(len(liquidations))
            values[f"liquidation_net_qty_{window}s"] = float(
                sum(liquidation[1] for liquidation in liquidations)
            )
            values[f"liquidation_abs_qty_{window}s"] = float(
                sum(liquidation[2] for liquidation in liquidations)
            )
            values[f"liquidation_net_notional_{window}s"] = float(
                sum(liquidation[3] for liquidation in liquidations)
            )
            values[f"liquidation_abs_notional_{window}s"] = float(
                sum(liquidation[4] for liquidation in liquidations)
            )
        values.update(self._book_features(decision_time_ms))
        values["mark_available"] = 1.0 if self._mark_price is not None else 0.0
        values["mark_update_age_ms"] = (
            float(decision_time_ms - self._last_mark_event_time_ms)
            if self._last_mark_event_time_ms is not None
            else 0.0
        )
        values["mark_index_basis_bps"] = _basis_bps(self._mark_price, self._index_price)
        values["funding_rate"] = self._funding_rate or 0.0
        values["hour_sin"], values["hour_cos"] = _cyclical_hour(decision_time_ms)
        return FeatureSnapshot(decision_time_ms=decision_time_ms, values=values)

    def _evict(self, timestamp_ms: int) -> None:
        cutoff = timestamp_ms - self._max_window_ms
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()
        while self._liquidations and self._liquidations[0][0] < cutoff:
            self._liquidations.popleft()
        while self._book_events and self._book_events[0][0] < cutoff:
            self._book_events.popleft()

    def _book_features(self, decision_time_ms: int) -> dict[str, float]:
        if self._book is None:
            return _empty_book_feature_row(self._windows)
        event_time_ms, bid, bid_quantity, ask, ask_quantity = self._book
        values = _book_feature_row(
            decision_time_ms=decision_time_ms,
            event_time_ms=event_time_ms,
            bid=bid,
            bid_quantity=bid_quantity,
            ask=ask,
            ask_quantity=ask_quantity,
        )
        for window in self._windows:
            cutoff = decision_time_ms - window * 1000
            window_events = [event for event in self._book_events if event[0] >= cutoff]
            values.update(_book_window_feature_row(window_events, window=window))
        return values


def materialize_trade_flow_features(
    trades: pd.DataFrame,
    decision_times_ms: np.ndarray,
    *,
    windows_seconds: tuple[int, ...] = TRADE_WINDOWS_SECONDS,
) -> pd.DataFrame:
    """Vectorized historical trade features with the same formulas as online."""
    required = {"event_time_ms", "price", "quantity", "is_buyer_maker"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trade frame is missing columns: {sorted(missing)}")
    times = trades["event_time_ms"].to_numpy(dtype=np.int64)
    prices = trades["price"].to_numpy(dtype=np.float64)
    quantities = trades["quantity"].to_numpy(dtype=np.float64)
    makers = trades["is_buyer_maker"].to_numpy(dtype=bool)
    if not len(times) or (np.diff(times) < 0).any():
        raise ValueError("trade events must be non-empty and chronological")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("trade prices must be finite and positive")
    decisions = np.asarray(decision_times_ms, dtype=np.int64)
    if decisions.ndim != 1 or (np.diff(decisions) < 0).any():
        raise ValueError("decision times must be a monotonic vector")

    quote = prices * quantities
    signed_quote = quote * np.where(makers, -1.0, 1.0)
    prefix_quote = _prefix_sum(quote)
    prefix_signed_quote = _prefix_sum(signed_quote)
    log_prices = np.log(prices)
    squared_returns = np.zeros(len(prices), dtype=np.float64)
    squared_returns[1:] = np.diff(log_prices) ** 2
    prefix_squared_returns = _prefix_sum(squared_returns)
    right = np.searchsorted(times, decisions, side="right")

    output: dict[str, np.ndarray] = {"decision_time_ms": decisions}
    for window in sorted(set(windows_seconds)):
        if window <= 0:
            raise ValueError("feature windows must be positive")
        left = np.searchsorted(times, decisions - window * 1000, side="left")
        counts = right - left
        total_quote = prefix_quote[right] - prefix_quote[left]
        net_quote = prefix_signed_quote[right] - prefix_signed_quote[left]
        output[f"trade_count_{window}s"] = counts.astype(np.float64)
        output[f"quote_volume_{window}s"] = total_quote
        output[f"flow_imbalance_{window}s"] = np.divide(
            net_quote,
            total_quote,
            out=np.zeros_like(net_quote),
            where=total_quote > 0,
        )
        first_price = np.where(counts > 0, prices[np.minimum(left, len(prices) - 1)], np.nan)
        last_indices = np.maximum(right - 1, 0)
        last_price = np.where(counts > 0, prices[last_indices], np.nan)
        output[f"return_{window}s_bps"] = np.where(
            counts > 0, np.log(last_price / first_price) * 10_000, 0.0
        )
        return_left = np.minimum(left + 1, right)
        sum_squared_returns = prefix_squared_returns[right] - prefix_squared_returns[return_left]
        output[f"realized_vol_{window}s_bps"] = np.sqrt(sum_squared_returns) * 10_000
        first_time = np.where(counts > 0, times[np.minimum(left, len(times) - 1)], 0)
        last_time = np.where(counts > 0, times[last_indices], 0)
        output[f"mean_interarrival_ms_{window}s"] = np.divide(
            last_time - first_time,
            counts - 1,
            out=np.zeros(len(decisions), dtype=np.float64),
            where=counts > 1,
        )
    hour_fraction = (decisions % 86_400_000) / 86_400_000
    output["hour_sin"] = np.sin(2 * np.pi * hour_fraction)
    output["hour_cos"] = np.cos(2 * np.pi * hour_fraction)
    return pd.DataFrame(output)


def materialize_book_features(
    book_events: pd.DataFrame,
    decision_times_ms: np.ndarray,
    *,
    windows_seconds: tuple[int, ...] = TRADE_WINDOWS_SECONDS,
) -> pd.DataFrame:
    """Vectorized top-of-book features using only the latest quote before each decision."""
    if not windows_seconds or any(window <= 0 for window in windows_seconds):
        raise ValueError("feature windows must be positive")
    windows = tuple(sorted(set(windows_seconds)))
    decisions = np.asarray(decision_times_ms, dtype=np.int64)
    if decisions.ndim != 1 or (np.diff(decisions) < 0).any():
        raise ValueError("decision times must be a monotonic vector")
    output = _empty_book_feature_frame(decisions, windows)
    required = {
        "event_time_ms",
        "bid_price",
        "bid_quantity",
        "ask_price",
        "ask_quantity",
    }
    missing = required - set(book_events.columns)
    if missing:
        raise ValueError(f"book frame is missing columns: {sorted(missing)}")
    if book_events.empty:
        return output

    sort_columns = ["event_time_ms"]
    if "sequence_id" in book_events.columns:
        sort_columns.append("sequence_id")
    ordered = book_events.sort_values(sort_columns, kind="stable")
    times = ordered["event_time_ms"].to_numpy(dtype=np.int64)
    if (np.diff(times) < 0).any():
        raise ValueError("book events must be chronological")
    bids = ordered["bid_price"].to_numpy(dtype=np.float64)
    bid_quantities = ordered["bid_quantity"].to_numpy(dtype=np.float64)
    asks = ordered["ask_price"].to_numpy(dtype=np.float64)
    ask_quantities = ordered["ask_quantity"].to_numpy(dtype=np.float64)
    _validate_book_arrays(bids, bid_quantities, asks, ask_quantities)
    spread_bps, depth_imbalance, microprice_displacement_bps = _book_state_arrays(
        bids,
        bid_quantities,
        asks,
        ask_quantities,
    )
    book_deltas = _book_liquidity_delta_arrays(bids, bid_quantities, asks, ask_quantities)

    latest_indices = np.searchsorted(times, decisions, side="right") - 1
    available = latest_indices >= 0
    safe_indices = np.maximum(latest_indices, 0)
    selected_times = times[safe_indices]
    output["book_available"] = available.astype(np.float64)
    output["book_update_age_ms"] = np.where(available, decisions - selected_times, 0.0)
    output["spread_bps"] = np.where(available, spread_bps[safe_indices], 0.0)
    output["depth_imbalance"] = np.where(available, depth_imbalance[safe_indices], 0.0)
    output["microprice_displacement_bps"] = np.where(
        available,
        microprice_displacement_bps[safe_indices],
        0.0,
    )
    for window in windows:
        output.update(
            _materialize_book_window_features(
                times=times,
                decisions=decisions,
                latest_indices=latest_indices,
                available=available,
                spread_bps=spread_bps,
                depth_imbalance=depth_imbalance,
                microprice_displacement_bps=microprice_displacement_bps,
                deltas=book_deltas,
                window=window,
            )
        )
    return pd.DataFrame(output)


def materialize_mark_features(
    mark_events: pd.DataFrame,
    decision_times_ms: np.ndarray,
) -> pd.DataFrame:
    """Latest mark/index/funding facts using only events before each decision."""
    decisions = np.asarray(decision_times_ms, dtype=np.int64)
    if decisions.ndim != 1 or (np.diff(decisions) < 0).any():
        raise ValueError("decision times must be a monotonic vector")
    output = {
        "decision_time_ms": decisions,
        "mark_available": np.zeros(len(decisions), dtype=np.float64),
        "mark_update_age_ms": np.zeros(len(decisions), dtype=np.float64),
        "mark_index_basis_bps": np.zeros(len(decisions), dtype=np.float64),
        "funding_rate": np.zeros(len(decisions), dtype=np.float64),
    }
    required = {"event_time_ms", "funding_rate", "index_price"}
    mark_column = "mark_price" if "mark_price" in mark_events.columns else "price"
    if mark_column not in mark_events.columns:
        required.add("mark_price")
    missing = required - set(mark_events.columns)
    if missing:
        raise ValueError(f"mark frame is missing columns: {sorted(missing)}")
    if mark_events.empty:
        return pd.DataFrame(output)

    ordered = mark_events.sort_values("event_time_ms", kind="stable")
    times = ordered["event_time_ms"].to_numpy(dtype=np.int64)
    if (np.diff(times) < 0).any():
        raise ValueError("mark events must be chronological")
    mark_prices = ordered[mark_column].to_numpy(dtype=np.float64)
    index_prices = ordered["index_price"].to_numpy(dtype=np.float64)
    funding_rates = ordered["funding_rate"].to_numpy(dtype=np.float64)
    if not np.isfinite(mark_prices).all() or (mark_prices <= 0).any():
        raise ValueError("mark prices must be finite and positive")
    if not np.isfinite(index_prices).all() or (index_prices <= 0).any():
        raise ValueError("index prices must be finite and positive")
    if not np.isfinite(funding_rates).all():
        raise ValueError("funding rates must be finite")

    latest_indices = np.searchsorted(times, decisions, side="right") - 1
    available = latest_indices >= 0
    safe_indices = np.maximum(latest_indices, 0)
    output["mark_available"] = available.astype(np.float64)
    output["mark_update_age_ms"] = np.where(available, decisions - times[safe_indices], 0.0)
    output["mark_index_basis_bps"] = np.where(
        available,
        (mark_prices[safe_indices] / index_prices[safe_indices] - 1) * 10_000,
        0.0,
    )
    output["funding_rate"] = np.where(available, funding_rates[safe_indices], 0.0)
    return pd.DataFrame(output)


def materialize_liquidation_features(
    liquidation_events: pd.DataFrame,
    decision_times_ms: np.ndarray,
    *,
    windows_seconds: tuple[int, ...] = TRADE_WINDOWS_SECONDS,
) -> pd.DataFrame:
    """Windowed liquidation-flow features using only prior force-order events."""
    if not windows_seconds or any(window <= 0 for window in windows_seconds):
        raise ValueError("feature windows must be positive")
    windows = tuple(sorted(set(windows_seconds)))
    decisions = np.asarray(decision_times_ms, dtype=np.int64)
    if decisions.ndim != 1 or (np.diff(decisions) < 0).any():
        raise ValueError("decision times must be a monotonic vector")
    output: dict[str, np.ndarray] = {"decision_time_ms": decisions}
    for window in windows:
        prefix = f"{window}s"
        output[f"liquidation_count_{prefix}"] = np.zeros(len(decisions), dtype=np.float64)
        output[f"liquidation_net_qty_{prefix}"] = np.zeros(len(decisions), dtype=np.float64)
        output[f"liquidation_abs_qty_{prefix}"] = np.zeros(len(decisions), dtype=np.float64)
        output[f"liquidation_net_notional_{prefix}"] = np.zeros(
            len(decisions), dtype=np.float64
        )
        output[f"liquidation_abs_notional_{prefix}"] = np.zeros(
            len(decisions), dtype=np.float64
        )
    required = {"event_time_ms", "quantity", "side"}
    missing = required - set(liquidation_events.columns)
    if missing:
        raise ValueError(f"liquidation frame is missing columns: {sorted(missing)}")
    if liquidation_events.empty:
        return pd.DataFrame(output)

    sort_columns = ["event_time_ms"]
    if "sequence_id" in liquidation_events.columns:
        sort_columns.append("sequence_id")
    ordered = liquidation_events.sort_values(sort_columns, kind="stable")
    times = ordered["event_time_ms"].to_numpy(dtype=np.int64)
    if (np.diff(times) < 0).any():
        raise ValueError("liquidation events must be chronological")
    quantities = ordered["quantity"].to_numpy(dtype=np.float64)
    if not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("liquidation quantities must be finite and non-negative")
    sides = ordered["side"].astype(str).str.upper().to_numpy()
    if not np.isin(sides, ["BUY", "SELL"]).all():
        raise ValueError("liquidation sides must be BUY or SELL")
    prices = (
        ordered["price"].fillna(0).to_numpy(dtype=np.float64)
        if "price" in ordered.columns
        else np.zeros(len(ordered), dtype=np.float64)
    )
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("liquidation prices must be finite and non-negative")
    signs = np.where(sides == "BUY", 1.0, -1.0)
    notional = quantities * prices
    signed_quantity = quantities * signs
    signed_notional = notional * signs

    prefix_count = _prefix_sum(np.ones(len(times), dtype=np.float64))
    prefix_signed_quantity = _prefix_sum(signed_quantity)
    prefix_abs_quantity = _prefix_sum(quantities)
    prefix_signed_notional = _prefix_sum(signed_notional)
    prefix_abs_notional = _prefix_sum(notional)
    right = np.searchsorted(times, decisions, side="right")
    for window in windows:
        prefix = f"{window}s"
        left = np.searchsorted(times, decisions - window * 1000, side="left")
        output[f"liquidation_count_{prefix}"] = prefix_count[right] - prefix_count[left]
        output[f"liquidation_net_qty_{prefix}"] = (
            prefix_signed_quantity[right] - prefix_signed_quantity[left]
        )
        output[f"liquidation_abs_qty_{prefix}"] = (
            prefix_abs_quantity[right] - prefix_abs_quantity[left]
        )
        output[f"liquidation_net_notional_{prefix}"] = (
            prefix_signed_notional[right] - prefix_signed_notional[left]
        )
        output[f"liquidation_abs_notional_{prefix}"] = (
            prefix_abs_notional[right] - prefix_abs_notional[left]
        )
    return pd.DataFrame(output)


def _trade_window_features(
    trades: list[tuple[int, float, float, float, float]], window: int
) -> dict[str, float]:
    prefix = f"{window}s"
    if not trades:
        return {
            f"trade_count_{prefix}": 0.0,
            f"quote_volume_{prefix}": 0.0,
            f"flow_imbalance_{prefix}": 0.0,
            f"return_{prefix}_bps": 0.0,
            f"realized_vol_{prefix}_bps": 0.0,
            f"mean_interarrival_ms_{prefix}": 0.0,
        }
    total_quote = sum(trade[3] for trade in trades)
    net_quote = sum(trade[3] * trade[4] for trade in trades)
    prices = np.fromiter((trade[1] for trade in trades), dtype=np.float64)
    timestamps = np.fromiter((trade[0] for trade in trades), dtype=np.int64)
    return {
        f"trade_count_{prefix}": float(len(trades)),
        f"quote_volume_{prefix}": float(total_quote),
        f"flow_imbalance_{prefix}": float(net_quote / total_quote) if total_quote else 0.0,
        f"return_{prefix}_bps": float(math.log(prices[-1] / prices[0]) * 10_000),
        f"realized_vol_{prefix}_bps": float(np.sqrt(np.sum(np.diff(np.log(prices)) ** 2)) * 10_000),
        f"mean_interarrival_ms_{prefix}": float(
            (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        )
        if len(timestamps) > 1
        else 0.0,
    }


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values)))


def _empty_book_feature_row(windows_seconds: tuple[int, ...]) -> dict[str, float]:
    return {column: 0.0 for column in _book_feature_columns(windows_seconds)}


def _empty_book_feature_frame(
    decisions: np.ndarray,
    windows_seconds: tuple[int, ...],
) -> dict[str, np.ndarray]:
    output = {"decision_time_ms": decisions}
    for column in _book_feature_columns(windows_seconds):
        output[column] = np.zeros(len(decisions), dtype=np.float64)
    return output


def _book_feature_columns(windows_seconds: tuple[int, ...]) -> tuple[str, ...]:
    return BASE_BOOK_FEATURE_COLUMNS + tuple(
        f"{prefix}_{window}s"
        for window in sorted(set(windows_seconds))
        for prefix in BOOK_WINDOW_FEATURE_PREFIXES
    )


def _book_feature_row(
    *,
    decision_time_ms: int,
    event_time_ms: int,
    bid: float,
    bid_quantity: float,
    ask: float,
    ask_quantity: float,
) -> dict[str, float]:
    spread, imbalance, microprice_displacement = _book_state_values(
        bid,
        bid_quantity,
        ask,
        ask_quantity,
    )
    return {
        "book_available": 1.0,
        "book_update_age_ms": float(decision_time_ms - event_time_ms),
        "spread_bps": spread,
        "depth_imbalance": imbalance,
        "microprice_displacement_bps": microprice_displacement,
    }


def _book_window_feature_row(
    events: list[tuple[int, float, float, float, float]],
    *,
    window: int,
) -> dict[str, float]:
    prefix = f"{window}s"
    if not events:
        return _empty_book_window_feature_row(prefix)

    spreads = []
    imbalances = []
    microprice_displacements = []
    for _, bid, bid_quantity, ask, ask_quantity in events:
        spread, imbalance, microprice_displacement = _book_state_values(
            bid,
            bid_quantity,
            ask,
            ask_quantity,
        )
        spreads.append(spread)
        imbalances.append(imbalance)
        microprice_displacements.append(microprice_displacement)

    bid_replenishment = 0.0
    ask_replenishment = 0.0
    bid_removed = 0.0
    ask_removed = 0.0
    for previous, current in zip(events, events[1:], strict=False):
        _, previous_bid, previous_bid_quantity, previous_ask, previous_ask_quantity = previous
        _, current_bid, current_bid_quantity, current_ask, current_ask_quantity = current
        bid_add, bid_remove, ask_add, ask_remove = _book_liquidity_delta_values(
            previous_bid=previous_bid,
            previous_bid_quantity=previous_bid_quantity,
            previous_ask=previous_ask,
            previous_ask_quantity=previous_ask_quantity,
            current_bid=current_bid,
            current_bid_quantity=current_bid_quantity,
            current_ask=current_ask,
            current_ask_quantity=current_ask_quantity,
        )
        bid_replenishment += bid_add
        bid_removed += bid_remove
        ask_replenishment += ask_add
        ask_removed += ask_remove

    pressure_denominator = bid_replenishment + ask_replenishment + bid_removed + ask_removed
    pressure_imbalance = (
        (bid_replenishment + ask_removed - ask_replenishment - bid_removed)
        / pressure_denominator
        if pressure_denominator
        else 0.0
    )
    latest_spread = spreads[-1]
    first_spread = spreads[0]
    return {
        f"book_event_count_{prefix}": float(len(events)),
        f"book_bid_replenishment_qty_{prefix}": bid_replenishment,
        f"book_ask_replenishment_qty_{prefix}": ask_replenishment,
        f"book_bid_liquidity_removed_qty_{prefix}": bid_removed,
        f"book_ask_liquidity_removed_qty_{prefix}": ask_removed,
        f"book_pressure_imbalance_{prefix}": pressure_imbalance,
        f"book_spread_widening_bps_{prefix}": max(latest_spread - first_spread, 0.0),
        f"book_spread_recovery_bps_{prefix}": max(first_spread - latest_spread, 0.0),
        f"book_depth_imbalance_change_{prefix}": imbalances[-1] - imbalances[0],
        f"book_microprice_displacement_change_bps_{prefix}": (
            microprice_displacements[-1] - microprice_displacements[0]
        ),
    }


def _empty_book_window_feature_row(prefix: str) -> dict[str, float]:
    return {f"{feature}_{prefix}": 0.0 for feature in BOOK_WINDOW_FEATURE_PREFIXES}


def _materialize_book_window_features(
    *,
    times: np.ndarray,
    decisions: np.ndarray,
    latest_indices: np.ndarray,
    available: np.ndarray,
    spread_bps: np.ndarray,
    depth_imbalance: np.ndarray,
    microprice_displacement_bps: np.ndarray,
    deltas: dict[str, np.ndarray],
    window: int,
) -> dict[str, np.ndarray]:
    left = np.searchsorted(times, decisions - window * 1000, side="left")
    right = np.where(available, latest_indices + 1, 0)
    counts = np.where(available & (left < right), right - left, 0).astype(np.float64)
    safe_first = np.minimum(left, len(times) - 1)
    safe_latest = np.maximum(latest_indices, 0)
    has_window_events = counts > 0
    prefix = f"{window}s"

    bid_replenishment = _window_sum_without_boundary_delta(
        deltas["bid_replenishment"],
        left,
        right,
        has_window_events,
    )
    ask_replenishment = _window_sum_without_boundary_delta(
        deltas["ask_replenishment"],
        left,
        right,
        has_window_events,
    )
    bid_removed = _window_sum_without_boundary_delta(
        deltas["bid_removed"],
        left,
        right,
        has_window_events,
    )
    ask_removed = _window_sum_without_boundary_delta(
        deltas["ask_removed"],
        left,
        right,
        has_window_events,
    )
    pressure_denominator = bid_replenishment + ask_replenishment + bid_removed + ask_removed
    pressure_imbalance = np.divide(
        bid_replenishment + ask_removed - ask_replenishment - bid_removed,
        pressure_denominator,
        out=np.zeros(len(decisions), dtype=np.float64),
        where=pressure_denominator > 0,
    )

    first_spread = spread_bps[safe_first]
    latest_spread = spread_bps[safe_latest]
    first_imbalance = depth_imbalance[safe_first]
    latest_imbalance = depth_imbalance[safe_latest]
    first_microprice = microprice_displacement_bps[safe_first]
    latest_microprice = microprice_displacement_bps[safe_latest]
    return {
        f"book_event_count_{prefix}": counts,
        f"book_bid_replenishment_qty_{prefix}": bid_replenishment,
        f"book_ask_replenishment_qty_{prefix}": ask_replenishment,
        f"book_bid_liquidity_removed_qty_{prefix}": bid_removed,
        f"book_ask_liquidity_removed_qty_{prefix}": ask_removed,
        f"book_pressure_imbalance_{prefix}": np.where(has_window_events, pressure_imbalance, 0.0),
        f"book_spread_widening_bps_{prefix}": np.where(
            has_window_events,
            np.maximum(latest_spread - first_spread, 0.0),
            0.0,
        ),
        f"book_spread_recovery_bps_{prefix}": np.where(
            has_window_events,
            np.maximum(first_spread - latest_spread, 0.0),
            0.0,
        ),
        f"book_depth_imbalance_change_{prefix}": np.where(
            has_window_events,
            latest_imbalance - first_imbalance,
            0.0,
        ),
        f"book_microprice_displacement_change_bps_{prefix}": np.where(
            has_window_events,
            latest_microprice - first_microprice,
            0.0,
        ),
    }


def _window_sum_without_boundary_delta(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    has_window_events: np.ndarray,
) -> np.ndarray:
    prefix = _prefix_sum(values)
    adjusted_left = np.minimum(left + 1, right)
    totals = prefix[right] - prefix[adjusted_left]
    return np.where(has_window_events, totals, 0.0)


def _book_state_arrays(
    bids: np.ndarray,
    bid_quantities: np.ndarray,
    asks: np.ndarray,
    ask_quantities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    midpoint = (bids + asks) / 2
    total_quantity = bid_quantities + ask_quantities
    microprice = np.divide(
        asks * bid_quantities + bids * ask_quantities,
        total_quantity,
        out=midpoint.copy(),
        where=total_quantity > 0,
    )
    return (
        (asks - bids) / midpoint * 10_000,
        np.divide(
            bid_quantities - ask_quantities,
            total_quantity,
            out=np.zeros(len(bids), dtype=np.float64),
            where=total_quantity > 0,
        ),
        (microprice / midpoint - 1) * 10_000,
    )


def _book_state_values(
    bid: float,
    bid_quantity: float,
    ask: float,
    ask_quantity: float,
) -> tuple[float, float, float]:
    spread, imbalance, microprice_displacement = _book_state_arrays(
        np.array([bid], dtype=np.float64),
        np.array([bid_quantity], dtype=np.float64),
        np.array([ask], dtype=np.float64),
        np.array([ask_quantity], dtype=np.float64),
    )
    return float(spread[0]), float(imbalance[0]), float(microprice_displacement[0])


def _book_liquidity_delta_arrays(
    bids: np.ndarray,
    bid_quantities: np.ndarray,
    asks: np.ndarray,
    ask_quantities: np.ndarray,
) -> dict[str, np.ndarray]:
    bid_replenishment = np.zeros(len(bids), dtype=np.float64)
    bid_removed = np.zeros(len(bids), dtype=np.float64)
    ask_replenishment = np.zeros(len(bids), dtype=np.float64)
    ask_removed = np.zeros(len(bids), dtype=np.float64)
    for index in range(1, len(bids)):
        bid_add, bid_remove, ask_add, ask_remove = _book_liquidity_delta_values(
            previous_bid=float(bids[index - 1]),
            previous_bid_quantity=float(bid_quantities[index - 1]),
            previous_ask=float(asks[index - 1]),
            previous_ask_quantity=float(ask_quantities[index - 1]),
            current_bid=float(bids[index]),
            current_bid_quantity=float(bid_quantities[index]),
            current_ask=float(asks[index]),
            current_ask_quantity=float(ask_quantities[index]),
        )
        bid_replenishment[index] = bid_add
        bid_removed[index] = bid_remove
        ask_replenishment[index] = ask_add
        ask_removed[index] = ask_remove
    return {
        "bid_replenishment": bid_replenishment,
        "bid_removed": bid_removed,
        "ask_replenishment": ask_replenishment,
        "ask_removed": ask_removed,
    }


def _book_liquidity_delta_values(
    *,
    previous_bid: float,
    previous_bid_quantity: float,
    previous_ask: float,
    previous_ask_quantity: float,
    current_bid: float,
    current_bid_quantity: float,
    current_ask: float,
    current_ask_quantity: float,
) -> tuple[float, float, float, float]:
    if current_bid > previous_bid:
        bid_replenishment = current_bid_quantity
        bid_removed = 0.0
    elif current_bid < previous_bid:
        bid_replenishment = 0.0
        bid_removed = previous_bid_quantity
    else:
        bid_delta = current_bid_quantity - previous_bid_quantity
        bid_replenishment = max(bid_delta, 0.0)
        bid_removed = max(-bid_delta, 0.0)

    if current_ask < previous_ask:
        ask_replenishment = current_ask_quantity
        ask_removed = 0.0
    elif current_ask > previous_ask:
        ask_replenishment = 0.0
        ask_removed = previous_ask_quantity
    else:
        ask_delta = current_ask_quantity - previous_ask_quantity
        ask_replenishment = max(ask_delta, 0.0)
        ask_removed = max(-ask_delta, 0.0)

    return bid_replenishment, bid_removed, ask_replenishment, ask_removed


def _validate_book_arrays(
    bids: np.ndarray,
    bid_quantities: np.ndarray,
    asks: np.ndarray,
    ask_quantities: np.ndarray,
) -> None:
    if not np.isfinite(bids).all() or not np.isfinite(asks).all():
        raise ValueError("book prices must be finite")
    if not np.isfinite(bid_quantities).all() or not np.isfinite(ask_quantities).all():
        raise ValueError("book quantities must be finite")
    if (bids <= 0).any() or (asks <= 0).any():
        raise ValueError("book prices must be positive")
    if (bid_quantities < 0).any() or (ask_quantities < 0).any():
        raise ValueError("book quantities cannot be negative")
    if (asks < bids).any():
        raise ValueError("book asks cannot be below bids")


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _basis_bps(mark_price: float | None, index_price: float | None) -> float:
    if mark_price is None or index_price is None or index_price <= 0:
        return 0.0
    return (mark_price / index_price - 1) * 10_000


def _cyclical_hour(timestamp_ms: int) -> tuple[float, float]:
    fraction = (timestamp_ms % 86_400_000) / 86_400_000
    return math.sin(2 * math.pi * fraction), math.cos(2 * math.pi * fraction)

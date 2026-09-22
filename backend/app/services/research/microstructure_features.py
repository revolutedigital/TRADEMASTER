"""Causal microstructure features shared by offline and online research paths."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.schemas.microstructure import MarketEventType, MicrostructureEvent


TRADE_WINDOWS_SECONDS = (1, 5, 30, 60)


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
        self._liquidations: deque[tuple[int, float]] = deque()
        self._last_event_time_ms: int | None = None
        self._book: tuple[float, float, float, float] | None = None
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
                    float(event.bid_price),
                    float(event.bid_quantity),
                    float(event.ask_price),
                    float(event.ask_quantity),
                )
        elif event.event_type == MarketEventType.MARK_PRICE and event.price is not None:
            self._mark_price = event.price
            payload = event.payload or {}
            self._index_price = _optional_float(payload.get("index_price"))
            self._funding_rate = _optional_float(payload.get("funding_rate"))
        elif event.event_type == MarketEventType.LIQUIDATION:
            if event.quantity is not None and event.quantity > 0:
                sign = 1.0 if event.side == "BUY" else -1.0
                self._liquidations.append((timestamp_ms, sign * event.quantity))
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
                signed_quantity
                for timestamp, signed_quantity in self._liquidations
                if timestamp >= cutoff
            ]
            values[f"liquidation_net_qty_{window}s"] = float(sum(liquidations))
        values.update(self._book_features())
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

    def _book_features(self) -> dict[str, float]:
        if self._book is None:
            return {
                "book_available": 0.0,
                "spread_bps": 0.0,
                "depth_imbalance": 0.0,
                "microprice_displacement_bps": 0.0,
            }
        bid, bid_quantity, ask, ask_quantity = self._book
        midpoint = (bid + ask) / 2
        total_quantity = bid_quantity + ask_quantity
        imbalance = (bid_quantity - ask_quantity) / total_quantity if total_quantity else 0.0
        microprice = (
            (ask * bid_quantity + bid * ask_quantity) / total_quantity
            if total_quantity
            else midpoint
        )
        return {
            "book_available": 1.0,
            "spread_bps": (ask - bid) / midpoint * 10_000,
            "depth_imbalance": imbalance,
            "microprice_displacement_bps": (microprice / midpoint - 1) * 10_000,
        }


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


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _basis_bps(mark_price: float | None, index_price: float | None) -> float:
    if mark_price is None or index_price is None or index_price <= 0:
        return 0.0
    return (mark_price / index_price - 1) * 10_000


def _cyclical_hour(timestamp_ms: int) -> tuple[float, float]:
    fraction = (timestamp_ms % 86_400_000) / 86_400_000
    return math.sin(2 * math.pi * fraction), math.cos(2 * math.pi * fraction)

"""Causal feature replay for broker-provided FX quotes and Level II deltas.

The cTrader Open API sends depth changes, not a documented historical order book. The first
subscription event is therefore provisional until an operator verifies it against the broker UI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

PRICE_SCALE = 100_000
SIZE_SCALE = 100
FEATURE_WINDOWS_SECONDS = (1, 5, 30)
MAX_TRACKED_SECONDS = max(FEATURE_WINDOWS_SECONDS)


def normalize_symbol_name(value: str) -> str:
    return value.replace("/", "").upper()


@dataclass(frozen=True)
class DepthQuote:
    side: str
    price_raw: int
    size_raw: int


class CTraderBookReplay:
    """Rebuild observed quote state and emit causal, explicitly qualified snapshots."""

    def __init__(
        self,
        *,
        symbol: str,
        pip_size: float,
        verified_generations: frozenset[int] = frozenset(),
        max_staleness_ms: int = 1_000,
    ) -> None:
        if pip_size <= 0:
            raise ValueError("pip_size must be positive")
        if max_staleness_ms < 0:
            raise ValueError("max_staleness_ms cannot be negative")
        self.symbol = normalize_symbol_name(symbol)
        self.pip_size = pip_size
        self.verified_generations = verified_generations
        self.max_staleness_ms = max_staleness_ms
        self._run_id: str | None = None
        self._generation: int | None = None
        self._quotes: dict[int, DepthQuote] = {}
        self._spot_bid_raw: int | None = None
        self._spot_ask_raw: int | None = None
        self._spot_bid_at_ms: int | None = None
        self._spot_ask_at_ms: int | None = None
        self._spot_bid_monotonic_ns: int | None = None
        self._spot_ask_monotonic_ns: int | None = None
        self._depth_received_at_ms: int | None = None
        self._depth_received_monotonic_ns: int | None = None
        self._quote_activity_available = False
        self._invalid_update_count = 0
        self._depth_changes: deque[tuple[int, int, int]] = deque()
        self._metric_history: deque[tuple[int, dict[str, float | None]]] = deque()

    def observe(self, event: dict[str, Any]) -> dict[str, Any]:
        if normalize_symbol_name(str(event.get("symbol", ""))) != self.symbol:
            raise ValueError("event symbol does not match the replay symbol")
        run_id = str(event["run_id"])
        generation = int(event["connection_generation"])
        if (self._run_id, self._generation) != (run_id, generation):
            self._reset(run_id, generation)

        received_at_ms = int(event["received_at_ms"])
        monotonic_ns = int(event.get("received_monotonic_ns", received_at_ms * 1_000_000))
        event_type = event["event_type"]
        new_count = 0
        deleted_count = 0

        if event_type == "spot":
            if event.get("bid_raw") is not None:
                self._spot_bid_raw = int(event["bid_raw"])
                self._spot_bid_at_ms = received_at_ms
                self._spot_bid_monotonic_ns = monotonic_ns
            if event.get("ask_raw") is not None:
                self._spot_ask_raw = int(event["ask_raw"])
                self._spot_ask_at_ms = received_at_ms
                self._spot_ask_monotonic_ns = monotonic_ns
        elif event_type == "depth_delta":
            new_quotes = event.get("new_quotes", [])
            deleted_quotes = event.get("deleted_quote_ids", [])
            new_count = len(new_quotes)
            deleted_count = len(deleted_quotes)
            for quote in new_quotes:
                quote_id = int(quote["id"])
                side = str(quote["side"])
                price_raw = int(quote["price_raw"])
                size_raw = int(quote["size_raw"])
                if side not in {"bid", "ask"} or price_raw <= 0 or size_raw <= 0:
                    self._invalid_update_count += 1
                    continue
                self._quotes[quote_id] = DepthQuote(side, price_raw, size_raw)
            for quote_id in deleted_quotes:
                self._quotes.pop(int(quote_id), None)
            self._depth_received_at_ms = received_at_ms
            self._depth_received_monotonic_ns = monotonic_ns
            self._quote_activity_available = True
            self._depth_changes.append((monotonic_ns, new_count, deleted_count))
        elif event_type == "market_snapshot":
            if event.get("spot_bid_raw") is not None:
                self._spot_bid_raw = int(event["spot_bid_raw"])
                self._spot_bid_at_ms = received_at_ms
                self._spot_bid_monotonic_ns = monotonic_ns
            if event.get("spot_ask_raw") is not None:
                self._spot_ask_raw = int(event["spot_ask_raw"])
                self._spot_ask_at_ms = received_at_ms
                self._spot_ask_monotonic_ns = monotonic_ns

            reconstructed: dict[int, DepthQuote] = {}
            synthetic_id = 0
            for side, key in (("bid", "bid_levels"), ("ask", "ask_levels")):
                for level in event.get(key, []):
                    price_raw = int(level["price_raw"])
                    size_raw = int(level["size_raw"])
                    if price_raw <= 0 or size_raw <= 0:
                        self._invalid_update_count += 1
                        continue
                    reconstructed[synthetic_id] = DepthQuote(side, price_raw, size_raw)
                    synthetic_id += 1
            self._quotes = reconstructed
            self._depth_received_at_ms = received_at_ms
            self._depth_received_monotonic_ns = monotonic_ns
            self._quote_activity_available = False
            self._depth_changes.append((monotonic_ns, 0, 0))
        else:
            raise ValueError(f"unsupported market event type: {event_type}")

        self._trim_windows(monotonic_ns)
        metrics = self._book_metrics()
        self._metric_history.append((monotonic_ns, metrics))
        self._trim_metric_history(monotonic_ns)
        return self._snapshot(event, received_at_ms, monotonic_ns, new_count, deleted_count, metrics)

    def _reset(self, run_id: str, generation: int) -> None:
        self._run_id = run_id
        self._generation = generation
        self._quotes.clear()
        self._spot_bid_raw = None
        self._spot_ask_raw = None
        self._spot_bid_at_ms = None
        self._spot_ask_at_ms = None
        self._spot_bid_monotonic_ns = None
        self._spot_ask_monotonic_ns = None
        self._depth_received_at_ms = None
        self._depth_received_monotonic_ns = None
        self._quote_activity_available = False
        self._invalid_update_count = 0
        self._depth_changes.clear()
        self._metric_history.clear()

    def _trim_windows(self, now_ns: int) -> None:
        cutoff = now_ns - MAX_TRACKED_SECONDS * 1_000_000_000
        while self._depth_changes and self._depth_changes[0][0] < cutoff:
            self._depth_changes.popleft()

    def _trim_metric_history(self, now_ns: int) -> None:
        cutoff = now_ns - MAX_TRACKED_SECONDS * 1_000_000_000
        while self._metric_history and self._metric_history[0][0] < cutoff:
            self._metric_history.popleft()

    def _book_metrics(self) -> dict[str, float | None]:
        grouped: dict[str, dict[int, int]] = {"bid": {}, "ask": {}}
        for quote in self._quotes.values():
            side = grouped[quote.side]
            side[quote.price_raw] = side.get(quote.price_raw, 0) + quote.size_raw

        bids = sorted(grouped["bid"].items(), reverse=True)
        asks = sorted(grouped["ask"].items())
        if not bids or not asks:
            return {
                "best_bid": None,
                "best_ask": None,
                "book_spread_pips": None,
                "depth_imbalance_1": None,
                "depth_imbalance_5": None,
                "depth_imbalance_10": None,
                "microprice_displacement_pips": None,
                "bid_depth_5_units": None,
                "ask_depth_5_units": None,
                "book_level_count": float(min(len(bids), len(asks))),
                "bid_level_count": float(len(bids)),
                "ask_level_count": float(len(asks)),
                "book_crossed": None,
            }

        best_bid_raw, best_bid_size_raw = bids[0]
        best_ask_raw, best_ask_size_raw = asks[0]
        best_bid, best_ask = best_bid_raw / PRICE_SCALE, best_ask_raw / PRICE_SCALE
        spread_raw = best_ask_raw - best_bid_raw
        top_bid_units = best_bid_size_raw / SIZE_SCALE
        top_ask_units = best_ask_size_raw / SIZE_SCALE
        top_total = top_bid_units + top_ask_units
        microprice = (
            best_ask * top_bid_units + best_bid * top_ask_units
        ) / top_total if top_total else None

        metrics: dict[str, float | None] = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "book_spread_pips": spread_raw / PRICE_SCALE / self.pip_size,
            "microprice_displacement_pips": (
                (microprice - (best_bid + best_ask) / 2) / self.pip_size
                if microprice is not None else None
            ),
            "bid_depth_5_units": sum(size for _, size in bids[:5]) / SIZE_SCALE,
            "ask_depth_5_units": sum(size for _, size in asks[:5]) / SIZE_SCALE,
            "book_level_count": float(min(len(bids), len(asks))),
            "bid_level_count": float(len(bids)),
            "ask_level_count": float(len(asks)),
            "book_crossed": float(spread_raw < 0),
        }
        for level_count in (1, 5, 10):
            bid_depth = sum(size for _, size in bids[:level_count]) / SIZE_SCALE
            ask_depth = sum(size for _, size in asks[:level_count]) / SIZE_SCALE
            total = bid_depth + ask_depth
            metrics[f"depth_imbalance_{level_count}"] = (
                (bid_depth - ask_depth) / total if total > 0 else None
            )
        return metrics

    def _snapshot(
        self,
        event: dict[str, Any],
        received_at_ms: int,
        monotonic_ns: int,
        new_count: int,
        deleted_count: int,
        metrics: dict[str, float | None],
    ) -> dict[str, Any]:
        depth_age_ms = (
            (monotonic_ns - self._depth_received_monotonic_ns) // 1_000_000
            if self._depth_received_monotonic_ns is not None else None
        )
        spot_bid = self._spot_bid_raw / PRICE_SCALE if self._spot_bid_raw is not None else None
        spot_ask = self._spot_ask_raw / PRICE_SCALE if self._spot_ask_raw is not None else None
        spot_spread = (
            (spot_ask - spot_bid) / self.pip_size
            if spot_bid is not None and spot_ask is not None else None
        )
        spot_age_ms = (
            max(
                (monotonic_ns - self._spot_bid_monotonic_ns) // 1_000_000,
                (monotonic_ns - self._spot_ask_monotonic_ns) // 1_000_000,
            )
            if self._spot_bid_monotonic_ns is not None and self._spot_ask_monotonic_ns is not None
            else None
        )
        current_metrics: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self._run_id,
            "environment": event.get("environment"),
            "connection_generation": self._generation,
            "event_sequence": int(event["event_sequence"]),
            "symbol": self.symbol,
            "received_at_ms": received_at_ms,
            "received_monotonic_ns": monotonic_ns,
            "source_event_type": event["event_type"],
            "depth_event_at_ms": self._depth_received_at_ms,
            "depth_age_ms": depth_age_ms,
            "spot_event_at_ms": max(
                (at for at in (self._spot_bid_at_ms, self._spot_ask_at_ms) if at is not None),
                default=None,
            ),
            "spot_quote_age_ms": spot_age_ms,
            "spot_quote_eligible": bool(
                spot_age_ms is not None
                and spot_age_ms <= self.max_staleness_ms
                and spot_bid is not None
                and spot_ask is not None
                and spot_ask >= spot_bid
            ),
            "spot_bid": spot_bid,
            "spot_ask": spot_ask,
            "spot_mid": (spot_bid + spot_ask) / 2 if spot_bid is not None and spot_ask is not None else None,
            "spot_spread_pips": spot_spread,
            "new_quote_count": new_count,
            "deleted_quote_count": deleted_count,
            "observed_quote_count": len(self._quotes),
            "invalid_depth_update_count": self._invalid_update_count,
            "book_snapshot_verified": self._generation in self.verified_generations,
            "book_state": "awaiting_depth" if self._depth_received_at_ms is None else (
                "verified" if self._generation in self.verified_generations else "provisional"
            ),
            **metrics,
        }
        current_metrics["book_eligible"] = bool(
            self._generation in self.verified_generations
            and self._depth_received_at_ms is not None
            and depth_age_ms is not None
            and depth_age_ms <= self.max_staleness_ms
            and self._invalid_update_count == 0
            and metrics["book_crossed"] == 0.0
            and metrics["depth_imbalance_5"] is not None
        )

        for window in FEATURE_WINDOWS_SECONDS:
            window_ns = window * 1_000_000_000
            baseline = next(
                (
                    snapshot for timestamp, snapshot in reversed(self._metric_history)
                    if timestamp <= monotonic_ns - window_ns
                ),
                None,
            )
            recent = [item for item in self._depth_changes if item[0] >= monotonic_ns - window_ns]
            added = sum(item[1] for item in recent)
            deleted = sum(item[2] for item in recent)
            current_metrics[f"depth_events_{window}s"] = len(recent)
            activity_available = self._quote_activity_available
            current_metrics[f"quotes_added_{window}s"] = added if activity_available else None
            current_metrics[f"quotes_deleted_{window}s"] = deleted if activity_available else None
            current_metrics[f"quote_add_cancel_imbalance_{window}s"] = (
                (added - deleted) / (added + deleted) if added + deleted else 0.0
            ) if activity_available else None
            for name in ("depth_imbalance_5", "microprice_displacement_pips"):
                old_value = baseline.get(name) if baseline is not None else None
                now_value = metrics[name]
                current_metrics[f"{name}_change_{window}s"] = (
                    now_value - old_value
                    if now_value is not None and old_value is not None else None
                )
        return current_metrics

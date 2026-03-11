"""Synthetic kline generator for when Binance WebSocket is unavailable.

Reads live prices from Redis (fed by frontend WebSocket) and builds
1-minute candles, publishing KLINE_UPDATE events to the event bus.
This allows the trading engine to function even when the backend
cannot connect to Binance directly (e.g., geo-restriction).
"""

import asyncio
import time
from datetime import datetime, timezone

from app.config import settings
from app.core.events import Event, EventType, event_bus
from app.core.logging import get_logger

logger = get_logger(__name__)


class SyntheticKlineGenerator:
    """Builds candles from Redis-cached spot prices and publishes kline events."""

    def __init__(self) -> None:
        self._running: bool = False
        # Track current candle state per symbol
        self._candles: dict[str, dict] = {}
        # Publish interval in seconds (1-minute candles)
        self._interval_seconds: int = 60

    async def start(self) -> None:
        """Start the synthetic kline generation loop."""
        self._running = True
        logger.info(
            "synthetic_kline_generator_starting",
            symbols=settings.symbols_list,
            interval_seconds=self._interval_seconds,
        )

        while self._running:
            try:
                await self._tick()
                await asyncio.sleep(5)  # Sample price every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("synthetic_kline_error", error=str(e))
                await asyncio.sleep(5)

        logger.info("synthetic_kline_generator_stopped")

    async def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        """Read prices from Redis and update/close candles."""
        if not event_bus._redis:
            return

        now_ms = int(time.time() * 1000)
        # Align to 1-minute boundaries
        candle_start_ms = (now_ms // (self._interval_seconds * 1000)) * (self._interval_seconds * 1000)
        candle_end_ms = candle_start_ms + (self._interval_seconds * 1000) - 1

        for symbol in settings.symbols_list:
            try:
                cached = await event_bus._redis.get(f"price:{symbol}")
                if not cached:
                    continue

                price = float(cached)
                if price <= 0:
                    continue

                candle = self._candles.get(symbol)

                # New candle period — close the old one and start fresh
                if candle is None or candle["open_time"] != candle_start_ms:
                    # Close the previous candle if it exists
                    if candle is not None:
                        await self._close_candle(symbol, candle)

                    # Start new candle
                    self._candles[symbol] = {
                        "symbol": symbol,
                        "interval": "1m",
                        "open_time": candle_start_ms,
                        "close_time": candle_end_ms,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0.0,
                        "quote_volume": 0.0,
                        "trade_count": 0,
                        "tick_count": 1,
                    }
                else:
                    # Update existing candle
                    candle["high"] = max(candle["high"], price)
                    candle["low"] = min(candle["low"], price)
                    candle["close"] = price
                    candle["tick_count"] += 1

            except Exception as e:
                logger.warning("synthetic_price_read_failed", symbol=symbol, error=str(e))

    async def _close_candle(self, symbol: str, candle: dict) -> None:
        """Publish a closed candle as a KLINE_UPDATE event."""
        event = Event(
            type=EventType.KLINE_UPDATE,
            data={
                "symbol": candle["symbol"],
                "interval": "1m",
                "is_closed": True,
                "open_time": candle["open_time"],
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volume"],
                "close_time": candle["close_time"],
                "quote_volume": candle["quote_volume"],
                "trade_count": candle["trade_count"],
            },
            source="synthetic_generator",
        )
        await event_bus.publish(event)
        logger.info(
            "synthetic_candle_closed",
            symbol=symbol,
            open=candle["open"],
            high=candle["high"],
            low=candle["low"],
            close=candle["close"],
            ticks=candle["tick_count"],
        )


synthetic_kline_generator = SyntheticKlineGenerator()

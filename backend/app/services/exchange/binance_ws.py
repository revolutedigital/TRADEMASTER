"""Binance WebSocket stream manager with auto-reconnect and Redis Streams fan-out."""

import asyncio
import json
from datetime import datetime, timezone

from binance import AsyncClient, BinanceSocketManager

from app.config import settings
from app.core.events import Event, EventType, event_bus
from app.core.logging import get_logger

logger = get_logger(__name__)

KLINE_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]


class BinanceWebSocketManager:
    """Manages multiple concurrent WebSocket streams for BTC and ETH."""

    def __init__(self) -> None:
        self._client: AsyncClient | None = None
        self._bsm: BinanceSocketManager | None = None
        self._tasks: list[asyncio.Task] = []
        self._running: bool = False
        self._reconnect_count: int = 0

    async def start(self, client: AsyncClient) -> None:
        """Start all WebSocket streams."""
        self._client = client
        self._bsm = BinanceSocketManager(client)
        self._running = True

        for symbol in settings.symbols_list:
            for interval in KLINE_INTERVALS:
                task = asyncio.create_task(
                    self._kline_stream(symbol.lower(), interval),
                    name=f"ws_kline_{symbol}_{interval}",
                )
                self._tasks.append(task)

            # Trade stream per symbol
            task = asyncio.create_task(
                self._trade_stream(symbol.lower()),
                name=f"ws_trade_{symbol}",
            )
            self._tasks.append(task)

        logger.info(
            "websocket_streams_started",
            symbols=settings.symbols_list,
            intervals=KLINE_INTERVALS,
            total_streams=len(self._tasks),
        )

    async def stop(self) -> None:
        """Stop all WebSocket streams."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("websocket_streams_stopped")

    async def _kline_stream(self, symbol: str, interval: str) -> None:
        """Subscribe to kline/candlestick stream for a symbol and interval."""
        while self._running:
            try:
                stream_name = f"{symbol}@kline_{interval}"
                async with self._bsm.kline_socket(symbol, interval=interval) as stream:
                    logger.info("ws_kline_connected", symbol=symbol, interval=interval)
                    while self._running:
                        msg = await asyncio.wait_for(stream.recv(), timeout=60)
                        if msg is None:
                            break
                        await self._handle_kline(msg, symbol.upper(), interval)
            except asyncio.TimeoutError:
                logger.warning("ws_kline_timeout", symbol=symbol, interval=interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._reconnect_count += 1
                logger.error(
                    "ws_kline_error",
                    symbol=symbol,
                    interval=interval,
                    error=str(e),
                    reconnect_count=self._reconnect_count,
                )
                await asyncio.sleep(min(2**self._reconnect_count, 60))

    async def _trade_stream(self, symbol: str) -> None:
        """Subscribe to trade stream for a symbol."""
        while self._running:
            try:
                async with self._bsm.trade_socket(symbol) as stream:
                    logger.info("ws_trade_connected", symbol=symbol)
                    while self._running:
                        msg = await asyncio.wait_for(stream.recv(), timeout=60)
                        if msg is None:
                            break
                        await self._handle_trade(msg, symbol.upper())
            except asyncio.TimeoutError:
                logger.warning("ws_trade_timeout", symbol=symbol)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._reconnect_count += 1
                logger.error(
                    "ws_trade_error",
                    symbol=symbol,
                    error=str(e),
                )
                await asyncio.sleep(min(2**self._reconnect_count, 60))

    async def _handle_kline(self, msg: dict, symbol: str, interval: str) -> None:
        """Parse kline message and publish to event bus."""
        if msg.get("e") != "kline":
            return

        k = msg["k"]
        is_closed = k["x"]  # Whether this candle is closed

        event = Event(
            type=EventType.KLINE_UPDATE,
            data={
                "symbol": symbol,
                "interval": interval,
                "is_closed": is_closed,
                "open_time": k["t"],
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "close_time": k["T"],
                "quote_volume": float(k["q"]),
                "trade_count": k["n"],
            },
        )
        await event_bus.publish(event)

    async def _handle_trade(self, msg: dict, symbol: str) -> None:
        """Parse trade message and publish to event bus."""
        if msg.get("e") != "trade":
            return

        event = Event(
            type=EventType.TRADE_UPDATE,
            data={
                "symbol": symbol,
                "price": float(msg["p"]),
                "quantity": float(msg["q"]),
                "trade_time": msg["T"],
                "is_buyer_maker": msg["m"],
            },
        )
        await event_bus.publish(event)


# Global instance
binance_ws_manager = BinanceWebSocketManager()

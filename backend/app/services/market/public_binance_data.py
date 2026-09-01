"""Credential-free Binance Spot market data for discovery and research."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

PUBLIC_MARKET_DATA_URL = "https://data-api.binance.vision/api/v3"


class PublicMarketDataUnavailable(RuntimeError):
    """Raised when public Binance market data cannot be trusted or reached."""


class PublicBinanceMarketData:
    """Small, bounded public-data client with no exchange-account access."""

    def __init__(self, base_url: str = PUBLIC_MARKET_DATA_URL) -> None:
        self._base_url = base_url.rstrip("/")

    async def exchange_info(self) -> dict[str, Any]:
        payload = await self._get_json("/exchangeInfo")
        if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
            raise PublicMarketDataUnavailable("Binance returned invalid exchange information")
        return payload

    async def ticker_24h(self) -> list[dict[str, Any]]:
        payload = await self._get_json("/ticker/24hr")
        if not isinstance(payload, list):
            raise PublicMarketDataUnavailable("Binance returned invalid 24-hour ticker data")
        return [item for item in payload if isinstance(item, dict)]

    async def klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: int | None = None,
    ) -> pd.DataFrame:
        params: dict[str, str | int] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(max(limit, 1), 1000),
        }
        if start_time is not None:
            params["startTime"] = start_time

        payload = await self._get_json("/klines", params=params)
        if not isinstance(payload, list):
            raise PublicMarketDataUnavailable("Binance returned invalid candlestick data")

        if not payload:
            return pd.DataFrame(
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trade_count",
                ]
            )

        try:
            frame = pd.DataFrame(
                payload,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trade_count",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
            frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
            for column in ("open", "high", "low", "close", "volume", "quote_volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce").fillna(0).astype(int)
        except (TypeError, ValueError) as exc:
            raise PublicMarketDataUnavailable("Binance returned malformed candlestick data") from exc

        return frame[
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
            ]
        ].dropna()

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> Any:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self._base_url}{path}", params=params)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("public_binance_market_data_unavailable", path=path, error=str(exc))
            raise PublicMarketDataUnavailable("Public Binance market data is temporarily unavailable") from exc


public_binance_market_data = PublicBinanceMarketData()

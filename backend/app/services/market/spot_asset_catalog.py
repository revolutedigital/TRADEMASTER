"""Eligible Spot USDT asset catalog for the single-asset trading workflow."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.services.market.public_binance_data import public_binance_market_data


CATALOG_TTL = timedelta(minutes=10)
MIN_QUOTE_VOLUME_24H = 100_000.0
EXCLUDED_BASE_ASSETS = frozenset(
    {
        "USDT",
        "USDC",
        "FDUSD",
        "TUSD",
        "USDP",
        "DAI",
        "BUSD",
        "USD1",
        "RLUSD",
        "USDE",
        "USDS",
        "GUSD",
        "LUSD",
        "FRAX",
        "PYUSD",
    }
)


@dataclass(frozen=True)
class SpotAsset:
    symbol: str
    base_asset: str
    quote_asset: str
    quote_volume_24h: float
    price_change_pct_24h: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "quote_volume_24h": self.quote_volume_24h,
            "price_change_pct_24h": self.price_change_pct_24h,
        }


class SpotAssetCatalog:
    """Caches an exchange-derived catalog without ever using account credentials."""

    def __init__(self) -> None:
        self._assets: tuple[SpotAsset, ...] = ()
        self._generated_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def list(self, *, search: str = "", limit: int = 100) -> tuple[list[SpotAsset], datetime]:
        assets, generated_at = await self._load()
        normalized_search = search.strip().upper()
        if normalized_search:
            assets = tuple(
                asset
                for asset in assets
                if normalized_search in asset.symbol or normalized_search in asset.base_asset
            )
        return list(assets[: min(max(limit, 1), 500)]), generated_at

    async def require(self, symbol: str) -> SpotAsset:
        normalized_symbol = symbol.upper().strip()
        assets, _generated_at = await self._load()
        for asset in assets:
            if asset.symbol == normalized_symbol:
                return asset
        raise ValueError(
            "The selected pair is not an eligible liquid Binance Spot USDT asset"
        )

    async def _load(self) -> tuple[tuple[SpotAsset, ...], datetime]:
        now = datetime.now(UTC)
        if self._generated_at and now - self._generated_at < CATALOG_TTL:
            return self._assets, self._generated_at

        async with self._lock:
            now = datetime.now(UTC)
            if self._generated_at and now - self._generated_at < CATALOG_TTL:
                return self._assets, self._generated_at

            exchange_info, tickers = await asyncio.gather(
                public_binance_market_data.exchange_info(),
                public_binance_market_data.ticker_24h(),
            )
            ticker_by_symbol = {
                str(ticker.get("symbol", "")).upper(): ticker for ticker in tickers
            }
            assets = tuple(
                sorted(
                    self._eligible_assets(exchange_info.get("symbols", []), ticker_by_symbol),
                    key=lambda asset: asset.quote_volume_24h,
                    reverse=True,
                )
            )
            self._assets = assets
            self._generated_at = now
            return assets, now

    @staticmethod
    def _eligible_assets(
        symbols: list[dict[str, Any]],
        ticker_by_symbol: dict[str, dict[str, Any]],
    ) -> list[SpotAsset]:
        assets: list[SpotAsset] = []
        for item in symbols:
            if not isinstance(item, dict) or item.get("status") != "TRADING":
                continue
            if item.get("quoteAsset") != "USDT":
                continue
            base_asset = str(item.get("baseAsset", "")).upper()
            symbol = str(item.get("symbol", "")).upper()
            if not symbol or base_asset in EXCLUDED_BASE_ASSETS or not _spot_enabled(item):
                continue

            ticker = ticker_by_symbol.get(symbol)
            if ticker is None:
                continue
            quote_volume = _finite_number(ticker.get("quoteVolume"))
            change_pct = _finite_number(ticker.get("priceChangePercent"))
            if quote_volume is None or quote_volume < MIN_QUOTE_VOLUME_24H or change_pct is None:
                continue
            assets.append(
                SpotAsset(
                    symbol=symbol,
                    base_asset=base_asset,
                    quote_asset="USDT",
                    quote_volume_24h=quote_volume,
                    price_change_pct_24h=change_pct,
                )
            )
        return assets


def _spot_enabled(symbol: dict[str, Any]) -> bool:
    permissions = symbol.get("permissions")
    if isinstance(permissions, list) and permissions:
        return "SPOT" in permissions

    permission_sets = symbol.get("permissionSets")
    if isinstance(permission_sets, list) and permission_sets:
        return any(
            isinstance(permission_set, list) and "SPOT" in permission_set
            for permission_set in permission_sets
        )

    return bool(symbol.get("isSpotTradingAllowed", True))


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


spot_asset_catalog = SpotAssetCatalog()

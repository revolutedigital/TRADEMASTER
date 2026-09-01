from unittest.mock import AsyncMock, patch

import pytest

from app.services.market.spot_asset_catalog import SpotAssetCatalog, _spot_enabled


@pytest.mark.asyncio
async def test_catalog_keeps_only_liquid_tradeable_spot_usdt_assets() -> None:
    catalog = SpotAssetCatalog()
    exchange_info = {
        "symbols": [
            {
                "symbol": "SOLUSDT",
                "baseAsset": "SOL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "permissions": ["SPOT"],
            },
            {
                "symbol": "LOWUSDT",
                "baseAsset": "LOW",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "permissions": ["SPOT"],
            },
            {
                "symbol": "USDCUSDT",
                "baseAsset": "USDC",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "permissions": ["SPOT"],
            },
            {
                "symbol": "USD1USDT",
                "baseAsset": "USD1",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "permissions": ["SPOT"],
            },
            {
                "symbol": "SOLBTC",
                "baseAsset": "SOL",
                "quoteAsset": "BTC",
                "status": "TRADING",
                "permissions": ["SPOT"],
            },
            {
                "symbol": "PAUSEDUSDT",
                "baseAsset": "PAUSED",
                "quoteAsset": "USDT",
                "status": "BREAK",
                "permissions": ["SPOT"],
            },
        ]
    }
    tickers = [
        {"symbol": "SOLUSDT", "quoteVolume": "2100000", "priceChangePercent": "3.4"},
        {"symbol": "LOWUSDT", "quoteVolume": "99999", "priceChangePercent": "2.1"},
        {"symbol": "USDCUSDT", "quoteVolume": "9000000", "priceChangePercent": "0"},
        {"symbol": "USD1USDT", "quoteVolume": "9000000", "priceChangePercent": "0"},
    ]

    with (
        patch(
            "app.services.market.spot_asset_catalog.public_binance_market_data.exchange_info",
            new=AsyncMock(return_value=exchange_info),
        ),
        patch(
            "app.services.market.spot_asset_catalog.public_binance_market_data.ticker_24h",
            new=AsyncMock(return_value=tickers),
        ),
    ):
        assets, _generated_at = await catalog.list(limit=100)

    assert [asset.symbol for asset in assets] == ["SOLUSDT"]
    assert assets[0].quote_volume_24h == 2_100_000.0


@pytest.mark.asyncio
async def test_catalog_filters_by_search_and_serves_cached_result() -> None:
    catalog = SpotAssetCatalog()
    exchange_info = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "permissions": ["SPOT"],
            },
            {
                "symbol": "ETHUSDT",
                "baseAsset": "ETH",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "permissions": ["SPOT"],
            },
        ]
    }
    exchange_info_mock = AsyncMock(return_value=exchange_info)
    ticker_mock = AsyncMock(
        return_value=[
            {"symbol": "BTCUSDT", "quoteVolume": "3000000", "priceChangePercent": "1"},
            {"symbol": "ETHUSDT", "quoteVolume": "2000000", "priceChangePercent": "2"},
        ]
    )

    with (
        patch(
            "app.services.market.spot_asset_catalog.public_binance_market_data.exchange_info",
            new=exchange_info_mock,
        ),
        patch(
            "app.services.market.spot_asset_catalog.public_binance_market_data.ticker_24h",
            new=ticker_mock,
        ),
    ):
        assets, _generated_at = await catalog.list(search="eth", limit=100)
        again, _generated_at = await catalog.list(limit=100)

    assert [asset.symbol for asset in assets] == ["ETHUSDT"]
    assert [asset.symbol for asset in again] == ["BTCUSDT", "ETHUSDT"]
    exchange_info_mock.assert_awaited_once()
    ticker_mock.assert_awaited_once()


def test_catalog_supports_current_binance_permission_sets_format() -> None:
    assert _spot_enabled({"permissions": [], "permissionSets": [["SPOT", "MARGIN"]]})
    assert not _spot_enabled({"permissions": [], "permissionSets": [["MARGIN"]]})

"""Tests for market-wide screening priorities and non-execution boundaries."""

import numpy as np
import pandas as pd
import pytest

from app.services.asset_intelligence import AssetIntelligenceError
from app.services.market.spot_asset_catalog import SpotAsset
from app.services.market_opportunity_scans import screen_asset


def _candles(closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC"),
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(len(closes), 1000.0),
            "close_time": pd.date_range("2025-01-01 00:59", periods=len(closes), freq="h", tz="UTC"),
            "quote_volume": np.full(len(closes), 100_000.0),
            "trade_count": np.full(len(closes), 100, dtype=int),
        }
    )


def test_screening_prioritizes_liquid_uptrends_without_creating_a_trade() -> None:
    asset = SpotAsset(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        quote_volume_24h=15_000_000.0,
        price_change_pct_24h=2.4,
    )

    result = screen_asset(asset, _candles(np.linspace(100, 150, 240)))

    assert result.asset == asset
    assert result.market_trend == "UPTREND"
    assert 0 < result.score <= 100


def test_screening_requires_enough_public_history() -> None:
    asset = SpotAsset(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        quote_volume_24h=15_000_000.0,
        price_change_pct_24h=2.4,
    )

    with pytest.raises(AssetIntelligenceError, match="insufficient"):
        screen_asset(asset, _candles(np.linspace(100, 120, 199)))

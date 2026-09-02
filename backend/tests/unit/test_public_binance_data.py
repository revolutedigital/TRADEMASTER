"""Closed-candle public-market data contract tests."""

from unittest.mock import AsyncMock

import pytest

from app.services.market.public_binance_data import PublicBinanceMarketData


@pytest.mark.asyncio
async def test_klines_retains_taker_buy_flow_and_discards_unclosed_candles() -> None:
    client = PublicBinanceMarketData()
    client._get_json = AsyncMock(
        return_value=[
            [
                1_577_836_800_000,
                "100",
                "102",
                "99",
                "101",
                "20",
                1_577_840_399_999,
                "2020",
                12,
                "11",
                "1111",
                "0",
            ],
            [
                4_102_444_800_000,
                "101",
                "103",
                "100",
                "102",
                "21",
                4_102_448_399_999,
                "2142",
                13,
                "12",
                "1212",
                "0",
            ],
        ]
    )

    frame = await client.klines(symbol="BTCUSDT", interval="1h")

    assert frame.columns.tolist()[-2:] == ["taker_buy_base", "taker_buy_quote"]
    assert len(frame) == 1
    assert frame.iloc[0]["taker_buy_base"] == 11
    assert frame.iloc[0]["taker_buy_quote"] == 1111

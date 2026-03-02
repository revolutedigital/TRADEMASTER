"""Integration tests for market data endpoints."""

import pytest


class TestKlines:
    async def test_get_klines_default(self, async_client):
        response = await async_client.get("/api/v1/market/klines/BTCUSDT")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_klines_with_params(self, async_client):
        response = await async_client.get(
            "/api/v1/market/klines/BTCUSDT?interval=1h&limit=10"
        )
        assert response.status_code == 200

    async def test_get_klines_invalid_interval(self, async_client):
        response = await async_client.get(
            "/api/v1/market/klines/BTCUSDT?interval=99x"
        )
        assert response.status_code == 422

    async def test_get_klines_limit_bounds(self, async_client):
        response = await async_client.get(
            "/api/v1/market/klines/BTCUSDT?limit=0"
        )
        assert response.status_code == 422

        response = await async_client.get(
            "/api/v1/market/klines/BTCUSDT?limit=1001"
        )
        assert response.status_code == 422


class TestSymbols:
    async def test_get_symbols(self, async_client):
        response = await async_client.get("/api/v1/market/symbols")
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert isinstance(data["symbols"], list)
        assert len(data["symbols"]) > 0


class TestTickers:
    async def test_get_tickers(self, async_client):
        response = await async_client.get("/api/v1/market/tickers")
        assert response.status_code == 200
        tickers = response.json()
        assert isinstance(tickers, list)
        for ticker in tickers:
            assert "symbol" in ticker
            assert "price" in ticker

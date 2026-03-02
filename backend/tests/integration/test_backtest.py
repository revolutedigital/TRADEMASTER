"""Integration tests for backtest endpoints."""

import pytest


class TestBacktestRun:
    async def test_backtest_requires_auth(self):
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/backtest/run",
                json={
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "initial_capital": 10000,
                },
            )
        assert response.status_code in (401, 403)

    async def test_backtest_run_no_data(self, async_client):
        """Backtest with no data should return zero-metric response."""
        response = await async_client.post(
            "/api/v1/backtest/run",
            json={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "initial_capital": 10000,
                "signal_threshold": 0.3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_trades" in data
        assert "sharpe_ratio" in data
        assert "equity_curve" in data


class TestBacktestHistory:
    async def test_get_history(self, async_client):
        response = await async_client.get("/api/v1/backtest/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_history_requires_auth(self):
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/backtest/history")
        assert response.status_code in (401, 403)

    async def test_get_nonexistent_backtest(self, async_client):
        response = await async_client.get("/api/v1/backtest/99999")
        assert response.status_code == 404

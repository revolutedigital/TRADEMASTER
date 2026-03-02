"""Contract tests: validate API response schemas match expected structures."""

import pytest


class TestPortfolioContracts:
    @pytest.mark.asyncio
    async def test_portfolio_summary_contract(self, async_client):
        response = await async_client.get("/api/v1/portfolio/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_equity" in data
        assert isinstance(data["total_equity"], (int, float))
        assert "daily_pnl" in data
        assert "open_positions" in data

    @pytest.mark.asyncio
    async def test_risk_status_contract(self, async_client):
        response = await async_client.get("/api/v1/portfolio/risk-status")
        assert response.status_code == 200
        data = response.json()
        assert "circuit_breaker_state" in data
        assert data["circuit_breaker_state"] in ("NORMAL", "REDUCED", "PAUSED", "HALTED")
        assert "can_trade" in data
        assert isinstance(data["can_trade"], bool)


class TestSystemContracts:
    @pytest.mark.asyncio
    async def test_health_contract(self, async_client):
        response = await async_client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    @pytest.mark.asyncio
    async def test_health_detailed_contract(self, async_client):
        response = await async_client.get("/api/v1/system/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data


class TestTradingContracts:
    @pytest.mark.asyncio
    async def test_engine_status_contract(self, async_client):
        response = await async_client.get("/api/v1/trading/engine/status")
        assert response.status_code == 200
        data = response.json()
        assert "engine_running" in data
        assert isinstance(data["engine_running"], bool)

    @pytest.mark.asyncio
    async def test_orders_contract(self, async_client):
        response = await async_client.get("/api/v1/trading/orders")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestSignalsContracts:
    @pytest.mark.asyncio
    async def test_signals_history_contract(self, async_client):
        response = await async_client.get("/api/v1/signals/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

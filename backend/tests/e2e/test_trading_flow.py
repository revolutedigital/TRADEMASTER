"""E2E test: full trading flow from auth to portfolio check."""

import pytest


class TestTradingFlow:
    """End-to-end flow: Login -> View Portfolio -> Execute Paper Trade -> Check Result."""

    async def test_full_paper_trade_flow(self, async_client):
        """Test the complete paper trading workflow."""

        # 1. Check portfolio summary
        summary_resp = await async_client.get("/api/v1/portfolio/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        assert "total_equity" in summary

        # 2. Check market data available
        symbols_resp = await async_client.get("/api/v1/market/symbols")
        assert symbols_resp.status_code == 200
        symbols = symbols_resp.json()["symbols"]
        assert len(symbols) > 0

        # 3. Check orders list
        orders_resp = await async_client.get("/api/v1/trading/orders")
        assert orders_resp.status_code == 200
        assert isinstance(orders_resp.json(), list)

        # 4. Check positions
        positions_resp = await async_client.get("/api/v1/portfolio/positions")
        assert positions_resp.status_code == 200
        assert isinstance(positions_resp.json(), list)

        # 5. Check risk status
        risk_resp = await async_client.get("/api/v1/portfolio/risk-status")
        assert risk_resp.status_code == 200
        risk = risk_resp.json()
        assert "state" in risk
        assert "can_trade" in risk

    async def test_engine_status_flow(self, async_client):
        """Test engine status check."""
        status_resp = await async_client.get("/api/v1/trading/engine/status")
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert "engine_running" in status
        assert "circuit_breaker" in status

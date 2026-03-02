"""Integration tests for portfolio endpoints."""

import pytest


class TestPositions:
    async def test_get_positions_empty(self, async_client):
        response = await async_client.get("/api/v1/portfolio/positions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_positions_with_symbol_filter(self, async_client):
        response = await async_client.get("/api/v1/portfolio/positions?symbol=BTCUSDT")
        assert response.status_code == 200

    async def test_get_closed_positions(self, async_client):
        response = await async_client.get("/api/v1/portfolio/positions?is_open=false")
        assert response.status_code == 200


class TestPortfolioSummary:
    async def test_get_summary(self, async_client):
        response = await async_client.get("/api/v1/portfolio/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_equity" in data
        assert "available_balance" in data
        assert "open_positions" in data


class TestRiskStatus:
    async def test_get_risk_status(self, async_client):
        response = await async_client.get("/api/v1/portfolio/risk-status")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "can_trade" in data

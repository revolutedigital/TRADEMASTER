"""Integration tests for trading endpoints."""

import pytest


class TestOrders:
    async def test_get_orders(self, async_client):
        response = await async_client.get("/api/v1/trading/orders")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_orders_with_filter(self, async_client):
        response = await async_client.get("/api/v1/trading/orders?symbol=BTCUSDT")
        assert response.status_code == 200

    async def test_orders_require_auth(self):
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/trading/orders")
        assert response.status_code in (401, 403)


class TestPaperOrder:
    async def test_paper_order_requires_auth(self):
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/trading/paper-order",
                json={"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.001},
            )
        assert response.status_code in (401, 403)

    async def test_paper_order_invalid_side(self, async_client):
        response = await async_client.post(
            "/api/v1/trading/paper-order",
            json={"symbol": "BTCUSDT", "side": "INVALID", "quantity": 0.001},
        )
        assert response.status_code == 400


class TestEngineControl:
    async def test_engine_status(self, async_client):
        response = await async_client.get("/api/v1/trading/engine/status")
        assert response.status_code == 200
        data = response.json()
        assert "engine_running" in data

    async def test_engine_control_requires_auth(self):
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/trading/engine/start")
        assert response.status_code in (401, 403)

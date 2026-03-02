"""Integration tests for signals endpoints."""

import pytest


class TestSignalHistory:
    async def test_get_signal_history(self, async_client):
        response = await async_client.get("/api/v1/signals/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_signal_history_with_filter(self, async_client):
        response = await async_client.get("/api/v1/signals/history?symbol=BTCUSDT&limit=10")
        assert response.status_code == 200

    async def test_signal_history_requires_auth(self):
        """Signals endpoint should require authentication."""
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/signals/history")
        assert response.status_code in (401, 403)

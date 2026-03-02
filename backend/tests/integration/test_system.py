"""Integration tests for system endpoints."""

import pytest


class TestHealth:
    async def test_health_endpoint_no_auth(self):
        """Health endpoint should be accessible without authentication."""
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")


class TestSystemStatus:
    async def test_status_requires_auth(self):
        """Status endpoint should require authentication."""
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/system/status")
        assert response.status_code in (401, 403)

    async def test_status_with_auth(self, async_client):
        response = await async_client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data or "version" in data or "status" in data


class TestMetrics:
    async def test_metrics_requires_auth(self):
        """Metrics endpoint should require authentication."""
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/system/metrics")
        assert response.status_code in (401, 403)

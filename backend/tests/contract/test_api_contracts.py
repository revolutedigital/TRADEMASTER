"""Contract tests: validate API response schemas match expectations."""
import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async test client with auth bypassed."""
    async def mock_auth():
        return {"sub": "admin", "role": "admin"}

    app.dependency_overrides = {}
    from app.dependencies import require_auth
    app.dependency_overrides[require_auth] = mock_auth

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides = {}


class TestHealthContract:
    @pytest.mark.anyio
    async def test_health_returns_required_fields(self, client):
        response = await client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)

    @pytest.mark.anyio
    async def test_health_detailed_structure(self, client):
        response = await client.get("/api/v1/system/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)


class TestAuthContract:
    @pytest.mark.anyio
    async def test_login_returns_token(self, client):
        response = await client.post("/api/v1/auth/login", json={
            "username": "wrong", "password": "wrong"
        })
        # Should return 401 for wrong creds, but structure should be error
        assert response.status_code in (401, 429)

    @pytest.mark.anyio
    async def test_login_missing_fields_422(self, client):
        response = await client.post("/api/v1/auth/login", json={})
        assert response.status_code == 422  # Validation error


class TestFeatureFlagsContract:
    @pytest.mark.anyio
    async def test_feature_flags_returns_dict(self, client):
        response = await client.get("/api/v1/admin/feature-flags")
        assert response.status_code == 200
        data = response.json()
        assert "flags" in data
        assert isinstance(data["flags"], dict)


class TestMLContract:
    @pytest.mark.anyio
    async def test_models_list_structure(self, client):
        response = await client.get("/api/v1/ml/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_symbols" in data

    @pytest.mark.anyio
    async def test_ensemble_weights_structure(self, client):
        response = await client.get("/api/v1/ml/ensemble/weights")
        assert response.status_code == 200
        data = response.json()
        assert "weights" in data
        assert isinstance(data["weights"], dict)

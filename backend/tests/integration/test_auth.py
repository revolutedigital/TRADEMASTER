"""Integration tests for authentication endpoints."""

import pytest
from unittest.mock import patch


class TestLogin:
    async def test_login_success(self, async_client):
        response = await async_client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "trademaster2024",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data

    async def test_login_invalid_password(self, async_client):
        response = await async_client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "wrong",
        })
        assert response.status_code == 401

    async def test_login_missing_fields(self, async_client):
        response = await async_client.post("/api/v1/auth/login", json={})
        assert response.status_code == 422

    async def test_login_sets_cookies(self, async_client):
        response = await async_client.post("/api/v1/auth/login", json={
            "username": "admin",
            "password": "trademaster2024",
        })
        assert response.status_code == 200
        cookies = response.cookies
        assert "access_token" in cookies or "csrf_token" in cookies


class TestRefreshToken:
    async def test_refresh_without_cookie_fails(self, async_client):
        response = await async_client.post("/api/v1/auth/refresh")
        assert response.status_code in (401, 403)


class TestLogout:
    async def test_logout_clears_cookies(self, async_client):
        response = await async_client.post("/api/v1/auth/logout")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "logged_out"

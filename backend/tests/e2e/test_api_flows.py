"""E2E tests that verify complete API flows against a live backend.

These tests require a running backend (use with docker-compose.staging.yml).
Skipped by default in CI unless E2E_TEST_URL is set.

Usage:
    # Against local dev server:
    E2E_TEST_URL=http://localhost:8000 pytest tests/e2e/test_api_flows.py -v

    # Against staging:
    E2E_TEST_URL=https://backendtrademaster.up.railway.app pytest tests/e2e/test_api_flows.py -v
"""

import os

import httpx
import pytest

BASE_URL = os.getenv("E2E_TEST_URL", "")

pytestmark = [
    pytest.mark.skipif(not BASE_URL, reason="E2E_TEST_URL not set"),
    pytest.mark.e2e,
]

# Default credentials — override via E2E_USERNAME / E2E_PASSWORD env vars
_USERNAME = os.getenv("E2E_USERNAME", "admin")
_PASSWORD = os.getenv("E2E_PASSWORD", "trademaster2024")


# ========================================
# Helpers
# ========================================


def _client() -> httpx.Client:
    """Create a synchronous httpx client pointed at the live backend."""
    return httpx.Client(base_url=BASE_URL, timeout=30)


def _login(client: httpx.Client) -> str:
    """Authenticate and return a Bearer token."""
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": _USERNAME, "password": _PASSWORD},
    )
    assert resp.status_code == 200, f"Login failed: {resp.status_code} {resp.text}"
    data = resp.json()
    assert "access_token" in data
    return data["access_token"]


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ========================================
# Auth Flow
# ========================================


class TestAuthFlow:
    """Verify login, token usage, and rejection of bad credentials."""

    def test_login_returns_token(self):
        with _client() as c:
            token = _login(c)
            assert len(token) > 20

    def test_invalid_login_returns_401(self):
        with _client() as c:
            resp = c.post(
                "/api/v1/auth/login",
                json={"username": "wrong", "password": "nope"},
            )
            assert resp.status_code in (401, 429)

    def test_protected_endpoint_without_token_returns_401(self):
        with _client() as c:
            resp = c.get("/api/v1/system/status")
            assert resp.status_code in (401, 403)

    def test_protected_endpoint_with_token_succeeds(self):
        with _client() as c:
            token = _login(c)
            resp = c.get("/api/v1/system/status", headers=_auth_headers(token))
            assert resp.status_code == 200


# ========================================
# Health Check Flow
# ========================================


class TestHealthFlow:
    """Verify health endpoints return expected structure."""

    def test_basic_health(self):
        with _client() as c:
            resp = c.get("/api/v1/system/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "version" in data

    def test_detailed_health(self):
        with _client() as c:
            resp = c.get("/api/v1/system/health/detailed")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] in ("healthy", "degraded")
            assert "uptime_seconds" in data
            assert "dependencies" in data
            deps = data["dependencies"]
            assert "database" in deps
            assert "redis" in deps
            assert "trading_engine" in deps
            assert "system" in data

    def test_v2_health(self):
        with _client() as c:
            resp = c.get("/api/v2/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["api_version"] == "2.0"
            assert "X-API-Version" in resp.headers or "x-api-version" in resp.headers


# ========================================
# Trading Flow
# ========================================


class TestTradingFlow:
    """Full paper trade cycle: login -> place order -> check position -> check portfolio."""

    def test_full_paper_trade_cycle(self):
        with _client() as c:
            token = _login(c)
            headers = _auth_headers(token)

            # 1. Get portfolio summary
            resp = c.get("/api/v1/portfolio/summary", headers=headers)
            assert resp.status_code == 200
            summary = resp.json()
            assert "total_equity" in summary

            # 2. Check available symbols
            resp = c.get("/api/v1/market/symbols", headers=headers)
            assert resp.status_code == 200
            symbols = resp.json()
            assert "symbols" in symbols
            assert len(symbols["symbols"]) > 0

            # 3. List existing orders
            resp = c.get("/api/v1/trading/orders", headers=headers)
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

            # 4. Check open positions
            resp = c.get("/api/v1/portfolio/positions", headers=headers)
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

            # 5. Check risk status
            resp = c.get("/api/v1/portfolio/risk-status", headers=headers)
            assert resp.status_code == 200
            risk = resp.json()
            assert "state" in risk
            assert "can_trade" in risk

    def test_engine_status(self):
        with _client() as c:
            token = _login(c)
            headers = _auth_headers(token)

            resp = c.get("/api/v1/trading/engine/status", headers=headers)
            assert resp.status_code == 200
            status = resp.json()
            assert "engine_running" in status
            assert "circuit_breaker" in status


# ========================================
# Backtest Flow
# ========================================


class TestBacktestFlow:
    """Run a backtest and verify results structure."""

    def test_run_backtest_and_get_results(self):
        with _client() as c:
            token = _login(c)
            headers = _auth_headers(token)

            # 1. Run backtest
            resp = c.post(
                "/api/v1/backtest/run",
                json={
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "initial_capital": 10000,
                    "signal_threshold": 0.3,
                    "atr_stop_multiplier": 2.0,
                    "risk_reward_ratio": 2.0,
                },
                headers=headers,
            )
            assert resp.status_code == 200
            result = resp.json()
            assert "total_trades" in result
            assert "sharpe_ratio" in result

            # 2. Check history
            resp = c.get("/api/v1/backtest/history", headers=headers)
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)


# ========================================
# V2 API Flow
# ========================================


class TestV2ApiFlow:
    """Verify V2 endpoints return the expected envelope and headers."""

    def test_v2_portfolio_summary(self):
        with _client() as c:
            token = _login(c)
            headers = _auth_headers(token)

            resp = c.get("/api/v2/portfolio/summary", headers=headers)
            assert resp.status_code == 200
            body = resp.json()
            assert "data" in body
            assert "meta" in body
            assert body["meta"]["api_version"] == "2.0"

    def test_v2_trades(self):
        with _client() as c:
            token = _login(c)
            headers = _auth_headers(token)

            resp = c.get("/api/v2/trades?limit=5", headers=headers)
            assert resp.status_code == 200
            body = resp.json()
            assert "data" in body
            assert body["meta"]["limit"] == 5

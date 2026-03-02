"""Shared test fixtures for TradeMaster test suite."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd


# ========================================
# Auth fixtures
# ========================================


@pytest.fixture
def auth_token():
    """Valid JWT token for authenticated test requests."""
    from app.core.security import create_access_token
    return create_access_token({"sub": "admin", "type": "access"})


# ========================================
# HTTP client fixture
# ========================================


@pytest.fixture
async def async_client(auth_token):
    """Async HTTP client with mocked auth for API testing."""
    from httpx import ASGITransport, AsyncClient
    from app.main import create_app
    from app.dependencies import require_auth

    app = create_app()

    async def override_require_auth():
        return {"sub": "admin", "type": "access"}

    app.dependency_overrides[require_auth] = override_require_auth

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


# ========================================
# Mock services
# ========================================


@pytest.fixture
def mock_binance_client():
    """Mock Binance async client."""
    client = AsyncMock()
    client.get_server_time = AsyncMock(return_value={"serverTime": 1700000000000})
    client.get_klines = AsyncMock(return_value=[])
    client.get_symbol_ticker = AsyncMock(return_value={"symbol": "BTCUSDT", "price": "85000.00"})
    client.get_ticker_price = AsyncMock(return_value=85000.0)
    client.get_balance = AsyncMock(return_value=10000.0)
    client.get_account = AsyncMock(return_value={
        "balances": [
            {"asset": "USDT", "free": "10000.00", "locked": "0.00"},
            {"asset": "BTC", "free": "0.1", "locked": "0.0"},
        ]
    })
    client.place_market_order = AsyncMock(return_value={
        "orderId": 12345,
        "status": "FILLED",
        "executedQty": "0.001",
        "avgPrice": "85000.00",
        "fills": [{"commission": "0.085", "commissionAsset": "USDT"}],
    })
    client.create_order = AsyncMock(return_value={
        "orderId": 12345,
        "status": "FILLED",
        "executedQty": "0.001",
        "avgPrice": "85000.00",
        "fills": [{"commission": "0.085", "commissionAsset": "USDT"}],
    })
    client.cancel_order = AsyncMock(return_value={"orderId": 12345, "status": "CANCELLED"})
    client.close_connection = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing event publishing."""
    bus = AsyncMock()
    bus.connect = AsyncMock()
    bus.disconnect = AsyncMock()
    bus.publish = AsyncMock(return_value="msg-123")
    bus.subscribe = AsyncMock(return_value=[])
    return bus


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_client = AsyncMock()
    redis_client.ping = AsyncMock(return_value=True)
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock()
    redis_client.setex = AsyncMock()
    redis_client.delete = AsyncMock()
    redis_client.aclose = AsyncMock()
    return redis_client


# ========================================
# Sample data fixtures
# ========================================


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV DataFrame for ML/indicator testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    base_price = 85000
    prices = base_price + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "open_time": dates,
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 200),
        "low": prices - np.abs(np.random.randn(n) * 200),
        "close": prices + np.random.randn(n) * 50,
        "volume": np.abs(np.random.randn(n) * 100) + 10,
    })
    # Ensure OHLCV consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 10
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 10
    return df


@pytest.fixture
def sample_trade_proposal():
    """Sample TradeProposal for risk management testing."""
    from app.services.risk.manager import TradeProposal
    return TradeProposal(
        symbol="BTCUSDT",
        side="BUY",
        signal_strength=0.65,
        entry_price=85000,
        atr=1200,
        current_equity=10000,
        current_exposure=0,
        symbol_exposure=0,
    )

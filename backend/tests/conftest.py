"""Shared test fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_binance_client():
    """Mock Binance async client."""
    client = AsyncMock()
    client.get_server_time = AsyncMock(return_value={"serverTime": 1700000000000})
    client.get_klines = AsyncMock(return_value=[])
    client.get_symbol_ticker = AsyncMock(return_value={"symbol": "BTCUSDT", "price": "85000.00"})
    client.get_account = AsyncMock(return_value={
        "balances": [
            {"asset": "USDT", "free": "10000.00", "locked": "0.00"},
            {"asset": "BTC", "free": "0.1", "locked": "0.0"},
        ]
    })
    client.create_order = AsyncMock(return_value={
        "orderId": 12345,
        "status": "FILLED",
        "executedQty": "0.001",
        "avgPrice": "85000.00",
        "fills": [{"commission": "0.085", "commissionAsset": "USDT"}],
    })
    client.close_connection = AsyncMock()
    return client

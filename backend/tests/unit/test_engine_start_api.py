"""The execution engine must never use synthetic price candles in exchange modes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.trading import start_engine
from app.config import TradingExecutionMode


@pytest.mark.asyncio
async def test_exchange_engine_start_requires_an_active_binance_stream() -> None:
    engine = SimpleNamespace(reserve_start=MagicMock(return_value=True))
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET)
    websocket_manager = SimpleNamespace(_running=False, _tasks=[])

    with (
        patch("app.api.v1.trading.settings", runtime_settings),
        patch(
            "app.services.exchange.binance_ws.binance_ws_manager",
            websocket_manager,
        ),
        pytest.raises(HTTPException, match="synthetic candles are restricted to PAPER") as error,
    ):
        await start_engine(_user={"sub": "operator"}, engine=engine)

    assert error.value.status_code == 409
    engine.reserve_start.assert_not_called()

"""Dashboard reads must default to the same execution ledger as the engine."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.api.v1.portfolio import get_portfolio_summary, get_positions
from app.api.v1.trading import get_orders
from app.config import TradingExecutionMode


@pytest.mark.asyncio
async def test_positions_default_to_the_active_execution_ledger():
    database = AsyncMock()
    repository = SimpleNamespace(get_open=AsyncMock(return_value=[]))
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.LIVE)

    with patch("app.api.v1.portfolio.settings", runtime_settings):
        positions = await get_positions(
            db=database,
            repo=repository,
            _user={"sub": "operator"},
        )

    assert positions == []
    repository.get_open.assert_awaited_once_with(
        database,
        None,
        execution_mode="LIVE",
    )


@pytest.mark.asyncio
async def test_positions_accept_an_explicit_non_active_ledger_for_auditing():
    database = AsyncMock()
    repository = SimpleNamespace(get_open=AsyncMock(return_value=[]))
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.LIVE)

    with patch("app.api.v1.portfolio.settings", runtime_settings):
        await get_positions(
            execution_mode=TradingExecutionMode.PAPER,
            db=database,
            repo=repository,
            _user={"sub": "operator"},
        )

    repository.get_open.assert_awaited_once_with(
        database,
        None,
        execution_mode="PAPER",
    )


@pytest.mark.asyncio
async def test_summary_only_uses_the_active_execution_ledger():
    database = AsyncMock()
    repository = SimpleNamespace(
        get_open=AsyncMock(return_value=[]),
        get_closed=AsyncMock(return_value=[]),
    )
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET)
    exchange = SimpleNamespace(get_balance=AsyncMock(return_value=1_000.0))

    with (
        patch("app.api.v1.portfolio.settings", runtime_settings),
        patch("app.api.v1.portfolio.get_binance_client", return_value=exchange),
    ):
        summary = await get_portfolio_summary(
            db=database,
            repo=repository,
            _user={"sub": "operator"},
        )

    assert summary.execution_mode == "TESTNET"
    repository.get_open.assert_awaited_once_with(database, execution_mode="TESTNET")
    repository.get_closed.assert_awaited_once_with(
        database,
        limit=1_000,
        execution_mode="TESTNET",
    )


@pytest.mark.asyncio
async def test_orders_default_to_the_active_execution_ledger():
    database = AsyncMock()
    repository = SimpleNamespace(list_filtered=AsyncMock(return_value=[]))
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET)

    with patch("app.api.v1.trading.settings", runtime_settings):
        orders = await get_orders(
            db=database,
            repo=repository,
            _user={"sub": "operator"},
        )

    assert orders == []
    assert repository.list_filtered.await_args.kwargs["execution_mode"] == "TESTNET"

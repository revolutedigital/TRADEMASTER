"""Paper endpoints must never mutate a ledger while an exchange mode is active."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.trading import (
    ClosePositionRequest,
    PaperOrderRequest,
    close_position_manually,
    create_paper_order,
)
from app.config import TradingExecutionMode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "execution_mode",
    [TradingExecutionMode.TESTNET, TradingExecutionMode.LIVE],
)
async def test_paper_order_is_rejected_outside_the_paper_ledger(execution_mode):
    database = AsyncMock()
    runtime_settings = SimpleNamespace(execution_mode=execution_mode)

    with (
        patch("app.api.v1.trading.settings", runtime_settings),
        pytest.raises(HTTPException, match="only available while execution mode is PAPER") as error,
    ):
        await create_paper_order(
            PaperOrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.001),
            database,
            {"sub": "operator"},
        )

    assert error.value.status_code == 409
    database.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_paper_sell_without_a_long_is_rejected_instead_of_opening_a_short():
    database = AsyncMock()
    database.add = MagicMock()
    database.execute = AsyncMock(
        return_value=SimpleNamespace(scalar_one_or_none=MagicMock(return_value=None))
    )
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.PAPER)

    with (
        patch("app.api.v1.trading.settings", runtime_settings),
        pytest.raises(HTTPException, match="can only reduce an existing LONG") as error,
    ):
        await create_paper_order(
            PaperOrderRequest(symbol="BTCUSDT", side="SELL", quantity=0.001, price=100_000),
            database,
            {"sub": "operator"},
        )

    assert error.value.status_code == 409
    database.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_paper_sell_partially_reduces_the_existing_long():
    long_position = SimpleNamespace(
        id=8,
        entry_price=90_000.0,
        quantity=1.0,
        current_price=90_000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        is_open=True,
        closed_at=None,
    )
    database = AsyncMock()
    database.add = MagicMock()
    database.execute = AsyncMock(
        return_value=SimpleNamespace(scalar_one_or_none=MagicMock(return_value=long_position))
    )
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.PAPER)

    with patch("app.api.v1.trading.settings", runtime_settings):
        response = await create_paper_order(
            PaperOrderRequest(symbol="BTCUSDT", side="SELL", quantity=0.4, price=100_000),
            database,
            {"sub": "operator"},
        )

    assert response["status"] == "position_reduced"
    assert response["remaining_quantity"] == pytest.approx(0.6)
    assert long_position.quantity == pytest.approx(0.6)
    assert long_position.is_open is True
    assert long_position.realized_pnl == pytest.approx(3_960.0)
    database.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "execution_mode",
    [TradingExecutionMode.TESTNET, TradingExecutionMode.LIVE],
)
async def test_virtual_paper_close_is_rejected_outside_the_paper_ledger(execution_mode):
    database = AsyncMock()
    runtime_settings = SimpleNamespace(execution_mode=execution_mode)

    with (
        patch("app.api.v1.trading.settings", runtime_settings),
        pytest.raises(HTTPException, match="only available while execution mode is PAPER") as error,
    ):
        await close_position_manually(
            position_id=7,
            req=ClosePositionRequest(),
            db=database,
            _user={"sub": "operator"},
        )

    assert error.value.status_code == 409
    database.execute.assert_not_awaited()

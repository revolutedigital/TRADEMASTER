"""Tests for order manager service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.exchange.order_manager import OrderManager


class TestOrderManager:
    def test_instantiation(self):
        manager = OrderManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_paper_order_execution(self):
        manager = OrderManager()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        with (
            patch("app.services.exchange.order_manager.settings") as mock_settings,
            patch("app.services.exchange.order_manager.binance_client") as mock_binance,
            patch("app.services.exchange.order_manager.event_bus") as mock_bus,
        ):
            mock_settings.paper_mode = True
            mock_binance.get_ticker_price = AsyncMock(return_value=85000.0)
            mock_bus.publish = AsyncMock()

            order = await manager.execute_market_order(
                db=mock_db,
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.001,
            )

        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.status == "FILLED"
        assert float(order.filled_quantity) == 0.001
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_paper_order_applies_slippage(self):
        manager = OrderManager()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        with (
            patch("app.services.exchange.order_manager.settings") as mock_settings,
            patch("app.services.exchange.order_manager.binance_client") as mock_binance,
            patch("app.services.exchange.order_manager.event_bus") as mock_bus,
        ):
            mock_settings.paper_mode = True
            mock_binance.get_ticker_price = AsyncMock(return_value=85000.0)
            mock_bus.publish = AsyncMock()

            order = await manager.execute_market_order(
                db=mock_db, symbol="BTCUSDT", side="BUY", quantity=0.001,
            )

        # BUY slippage should increase price
        assert float(order.avg_fill_price) >= 85000.0
        assert float(order.commission) > 0

    @pytest.mark.asyncio
    async def test_paper_order_sell_slippage(self):
        manager = OrderManager()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        with (
            patch("app.services.exchange.order_manager.settings") as mock_settings,
            patch("app.services.exchange.order_manager.binance_client") as mock_binance,
            patch("app.services.exchange.order_manager.event_bus") as mock_bus,
        ):
            mock_settings.paper_mode = True
            mock_binance.get_ticker_price = AsyncMock(return_value=85000.0)
            mock_bus.publish = AsyncMock()

            order = await manager.execute_market_order(
                db=mock_db, symbol="BTCUSDT", side="SELL", quantity=0.001,
            )

        # SELL slippage should decrease price
        assert float(order.avg_fill_price) <= 85000.0

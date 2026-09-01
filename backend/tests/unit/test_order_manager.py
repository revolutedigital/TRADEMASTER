"""Tests for order manager service."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import TradingExecutionMode
from app.core.exceptions import OrderExecutionError
from app.models.portfolio import Position
from app.models.trade import Order, OrderSide, OrderStatus, OrderType
from app.services.exchange.order_manager import OrderManager, SpotLongProtectionPlan


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
        assert order.execution_mode == TradingExecutionMode.PAPER.value
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
                db=mock_db,
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.001,
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
                db=mock_db,
                symbol="BTCUSDT",
                side="SELL",
                quantity=0.001,
            )

        # SELL slippage should decrease price
        assert float(order.avg_fill_price) <= 85000.0

    @pytest.mark.asyncio
    async def test_live_protection_failure_persists_the_unprotected_remainder(self):
        manager = OrderManager()
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        entry_order = Order(
            id=101,
            exchange_order_id="entry-101",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=1.0,
            price=100.0,
            filled_quantity=1.0,
            avg_fill_price=100.0,
            commission=0,
            execution_mode=TradingExecutionMode.LIVE.value,
        )
        protection = SpotLongProtectionPlan(
            stop_loss_price=Decimal("95"),
            take_profit_price=Decimal("105"),
        )

        with (
            patch("app.services.exchange.order_manager.binance_client") as exchange,
            patch("app.services.exchange.order_manager.live_trading_guard") as guard,
            pytest.raises(OrderExecutionError, match="native protection failed"),
        ):
            exchange.get_ticker_price = AsyncMock(return_value=Decimal("100"))
            exchange.place_spot_long_exit_oco = AsyncMock(side_effect=RuntimeError("OCO rejected"))
            exchange.place_market_order = AsyncMock(
                return_value={"orderId": 102, "status": "FILLED", "executedQty": "0.4"}
            )
            exchange.get_order_status = AsyncMock(
                return_value={
                    "orderId": 102,
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "status": "FILLED",
                    "executedQty": "0.4",
                    "cummulativeQuoteQty": "40",
                }
            )
            await manager._attach_native_spot_protection(
                db=database,
                order=entry_order,
                filled_quantity=Decimal("1"),
                fallback_price=Decimal("100"),
                protection=protection,
            )

        persisted_positions = [
            call.args[0]
            for call in database.add.call_args_list
            if isinstance(call.args[0], Position)
        ]
        assert len(persisted_positions) == 1
        assert float(persisted_positions[0].quantity) == 0.6
        assert persisted_positions[0].protection_status == "MISSING"
        guard.disarm.assert_called_once_with("native protection placement failed")

    @pytest.mark.asyncio
    async def test_partial_oco_never_sends_emergency_sell_before_signed_cancellation_confirmation(self):
        manager = OrderManager()
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        entry_order = Order(
            id=101,
            exchange_order_id="entry-101",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=1.0,
            price=100.0,
            filled_quantity=1.0,
            avg_fill_price=100.0,
            commission=0,
            execution_mode=TradingExecutionMode.LIVE.value,
        )
        protection = SpotLongProtectionPlan(
            stop_loss_price=Decimal("95"),
            take_profit_price=Decimal("105"),
        )
        partial_oco = MagicMock(order_list_id=9001, protected_quantity=Decimal("0.5"))

        with (
            patch("app.services.exchange.order_manager.binance_client") as exchange,
            patch("app.services.exchange.order_manager.live_trading_guard"),
            pytest.raises(OrderExecutionError, match="could not be cancelled"),
        ):
            exchange.get_ticker_price = AsyncMock(return_value=Decimal("100"))
            exchange.place_spot_long_exit_oco = AsyncMock(return_value=partial_oco)
            exchange.cancel_spot_order_list = AsyncMock(return_value={"orderListId": 9001})
            exchange.get_spot_order_list = AsyncMock(
                return_value={
                    "orderListId": 9001,
                    "symbol": "BTCUSDT",
                    "contingencyType": "OCO",
                    "listStatusType": "EXEC_STARTED",
                    "listOrderStatus": "EXECUTING",
                    "orders": [{"orderId": 11}, {"orderId": 12}],
                }
            )
            exchange.place_market_order = AsyncMock()

            await manager._attach_native_spot_protection(
                db=database,
                order=entry_order,
                filled_quantity=Decimal("1"),
                fallback_price=Decimal("100"),
                protection=protection,
            )

        exchange.place_market_order.assert_not_awaited()
        persisted_positions = [
            call.args[0]
            for call in database.add.call_args_list
            if isinstance(call.args[0], Position)
        ]
        assert len(persisted_positions) == 1
        assert float(persisted_positions[0].quantity) == 1.0
        assert persisted_positions[0].protection_status == "MISSING"

    @pytest.mark.asyncio
    async def test_live_post_submission_failure_commits_exchange_audit_records(self):
        manager = OrderManager()
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        rules = MagicMock()
        rules.normalize_oco_quantity.return_value = Decimal("1")
        rules.validate_notional.return_value = Decimal("100")
        config = SimpleNamespace(execution_mode=TradingExecutionMode.LIVE)
        protection = SpotLongProtectionPlan(
            stop_loss_price=Decimal("95"),
            take_profit_price=Decimal("105"),
        )

        with (
            patch("app.services.exchange.order_manager.settings", config),
            patch("app.services.exchange.order_manager.binance_client") as exchange,
            patch("app.services.exchange.order_manager.live_trading_guard"),
            patch.object(manager, "_reserve_live_notional", new=AsyncMock()),
            pytest.raises(OrderExecutionError, match="native protection failed"),
        ):
            exchange.get_ticker_price = AsyncMock(return_value=Decimal("100"))
            exchange.get_spot_symbol_rules = AsyncMock(return_value=rules)
            exchange.place_market_order = AsyncMock(
                side_effect=[
                    {"orderId": 101, "status": "FILLED", "executedQty": "1"},
                    {"orderId": 102, "status": "FILLED", "executedQty": "1"},
                ]
            )
            exchange.get_order_status = AsyncMock(
                side_effect=[
                    {
                        "orderId": 101,
                        "symbol": "BTCUSDT",
                        "side": "BUY",
                        "status": "FILLED",
                        "executedQty": "1",
                        "cummulativeQuoteQty": "100",
                    },
                    {
                        "orderId": 102,
                        "symbol": "BTCUSDT",
                        "side": "SELL",
                        "status": "FILLED",
                        "executedQty": "1",
                        "cummulativeQuoteQty": "100",
                    },
                ]
            )
            exchange.place_spot_long_exit_oco = AsyncMock(side_effect=RuntimeError("OCO rejected"))
            await manager._execute_live_order(
                db=database,
                symbol="BTCUSDT",
                side="BUY",
                quantity=1,
                protective_exit=protection,
            )

        database.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_testnet_long_entry_requires_and_attaches_native_oco_protection(self):
        manager = OrderManager()
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        rules = MagicMock()
        rules.normalize_oco_quantity.return_value = Decimal("1")
        rules.validate_notional.return_value = Decimal("100")
        protection_result = MagicMock(order_list_id=9001, protected_quantity=Decimal("1"))
        config = SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET)
        protection = SpotLongProtectionPlan(
            stop_loss_price=Decimal("95"),
            take_profit_price=Decimal("105"),
        )

        with (
            patch("app.services.exchange.order_manager.settings", config),
            patch("app.services.exchange.order_manager.binance_client") as exchange,
            patch("app.services.exchange.order_manager.event_bus") as event_bus,
        ):
            exchange.get_ticker_price = AsyncMock(return_value=Decimal("100"))
            exchange.get_spot_symbol_rules = AsyncMock(return_value=rules)
            exchange.place_market_order = AsyncMock(
                return_value={
                    "orderId": 101,
                    "status": "FILLED",
                    "executedQty": "1",
                    "avgPrice": "100",
                }
            )
            exchange.get_order_status = AsyncMock(
                side_effect=[
                    {
                        "orderId": 101,
                        "symbol": "BTCUSDT",
                        "side": "BUY",
                        "status": "FILLED",
                        "executedQty": "1",
                        "cummulativeQuoteQty": "100",
                    },
                    {
                        "orderId": 11,
                        "orderListId": 9001,
                        "symbol": "BTCUSDT",
                        "side": "SELL",
                        "status": "NEW",
                        "executedQty": "0",
                    },
                    {
                        "orderId": 12,
                        "orderListId": 9001,
                        "symbol": "BTCUSDT",
                        "side": "SELL",
                        "status": "NEW",
                        "executedQty": "0",
                    },
                ]
            )
            exchange.place_spot_long_exit_oco = AsyncMock(return_value=protection_result)
            exchange.get_spot_order_list = AsyncMock(
                return_value={
                    "orderListId": 9001,
                    "symbol": "BTCUSDT",
                    "contingencyType": "OCO",
                    "listStatusType": "EXEC_STARTED",
                    "listOrderStatus": "EXECUTING",
                    "orders": [{"orderId": 11}, {"orderId": 12}],
                }
            )
            event_bus.publish = AsyncMock()

            order = await manager._execute_live_order(
                db=database,
                symbol="BTCUSDT",
                side="BUY",
                quantity=1,
                protective_exit=protection,
            )

        assert order.execution_mode == TradingExecutionMode.TESTNET.value
        assert order.protective_order_list_id == 9001
        exchange.place_spot_long_exit_oco.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_active_oco_requires_two_matching_unfilled_signed_sell_orders(self):
        manager = OrderManager()

        with patch("app.services.exchange.order_manager.binance_client") as exchange:
            exchange.get_spot_order_list = AsyncMock(
                return_value={
                    "orderListId": 9001,
                    "symbol": "BTCUSDT",
                    "contingencyType": "OCO",
                    "listStatusType": "EXEC_STARTED",
                    "listOrderStatus": "EXECUTING",
                    "orders": [{"orderId": 11}, {"orderId": 12}],
                }
            )
            exchange.get_order_status = AsyncMock(
                side_effect=[
                    {
                        "orderId": 11,
                        "orderListId": 9001,
                        "symbol": "BTCUSDT",
                        "side": "SELL",
                        "status": "NEW",
                        "executedQty": "0",
                    },
                    {
                        "orderId": 12,
                        "orderListId": 9001,
                        "symbol": "BTCUSDT",
                        "side": "BUY",
                        "status": "NEW",
                        "executedQty": "0",
                    },
                ]
            )

            active = await manager._is_native_oco_active(
                symbol="BTCUSDT",
                order_list_id=9001,
            )

        assert active is False

    @pytest.mark.asyncio
    async def test_unconfirmed_active_oco_records_missing_position_without_emergency_sell(self):
        manager = OrderManager()
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        database.commit = AsyncMock()
        rules = MagicMock()
        rules.normalize_oco_quantity.return_value = Decimal("1")
        rules.validate_notional.return_value = Decimal("100")
        protection_result = MagicMock(order_list_id=9001, protected_quantity=Decimal("1"))
        config = SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET)
        protection = SpotLongProtectionPlan(
            stop_loss_price=Decimal("95"),
            take_profit_price=Decimal("105"),
        )

        with (
            patch("app.services.exchange.order_manager.settings", config),
            patch("app.services.exchange.order_manager.binance_client") as exchange,
            pytest.raises(OrderExecutionError, match="did not confirm active full protection"),
        ):
            exchange.get_ticker_price = AsyncMock(return_value=Decimal("100"))
            exchange.get_spot_symbol_rules = AsyncMock(return_value=rules)
            exchange.place_market_order = AsyncMock(
                return_value={"orderId": 101, "status": "FILLED", "executedQty": "1"}
            )
            exchange.get_order_status = AsyncMock(
                return_value={
                    "orderId": 101,
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "status": "FILLED",
                    "executedQty": "1",
                    "cummulativeQuoteQty": "100",
                }
            )
            exchange.place_spot_long_exit_oco = AsyncMock(return_value=protection_result)
            exchange.get_spot_order_list = AsyncMock(
                return_value={
                    "orderListId": 9001,
                    "symbol": "BTCUSDT",
                    "contingencyType": "OCO",
                    "listStatusType": "ALL_DONE",
                    "listOrderStatus": "ALL_DONE",
                    "orders": [{"orderId": 11}, {"orderId": 12}],
                }
            )

            await manager._execute_live_order(
                db=database,
                symbol="BTCUSDT",
                side="BUY",
                quantity=1,
                protective_exit=protection,
            )

        exchange.place_market_order.assert_awaited_once()
        persisted_positions = [
            call.args[0]
            for call in database.add.call_args_list
            if isinstance(call.args[0], Position)
        ]
        assert len(persisted_positions) == 1
        position = persisted_positions[0]
        assert position.protection_status == "MISSING"
        assert position.protective_order_list_id == 9001
        assert float(position.protective_quantity) == 1.0

    @pytest.mark.asyncio
    async def test_unconfirmed_entry_signed_read_records_a_missing_position_and_skips_oco(self):
        manager = OrderManager()
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        database.commit = AsyncMock()
        rules = MagicMock()
        rules.normalize_oco_quantity.return_value = Decimal("1")
        rules.validate_notional.return_value = Decimal("100")
        config = SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET)
        protection = SpotLongProtectionPlan(
            stop_loss_price=Decimal("95"),
            take_profit_price=Decimal("105"),
        )

        with (
            patch("app.services.exchange.order_manager.settings", config),
            patch("app.services.exchange.order_manager.binance_client") as exchange,
            pytest.raises(OrderExecutionError, match="signed read did not match"),
        ):
            exchange.get_ticker_price = AsyncMock(return_value=Decimal("100"))
            exchange.get_spot_symbol_rules = AsyncMock(return_value=rules)
            exchange.place_market_order = AsyncMock(
                return_value={"orderId": 101, "status": "FILLED", "executedQty": "1"}
            )
            exchange.get_order_status = AsyncMock(
                return_value={
                    "orderId": 101,
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "status": "FILLED",
                    "executedQty": "1",
                }
            )
            exchange.place_spot_long_exit_oco = AsyncMock()

            await manager._execute_live_order(
                db=database,
                symbol="BTCUSDT",
                side="BUY",
                quantity=1,
                protective_exit=protection,
            )

        exchange.place_spot_long_exit_oco.assert_not_awaited()
        persisted_positions = [
            call.args[0]
            for call in database.add.call_args_list
            if isinstance(call.args[0], Position)
        ]
        assert len(persisted_positions) == 1
        assert float(persisted_positions[0].quantity) == 1.0
        assert persisted_positions[0].protection_status == "MISSING"

    @pytest.mark.asyncio
    async def test_partially_filled_entry_is_recorded_as_missing_without_oco(self):
        manager = OrderManager()
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        database.commit = AsyncMock()
        rules = MagicMock()
        rules.normalize_oco_quantity.return_value = Decimal("1")
        rules.validate_notional.return_value = Decimal("100")
        config = SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET)
        protection = SpotLongProtectionPlan(
            stop_loss_price=Decimal("95"),
            take_profit_price=Decimal("105"),
        )

        with (
            patch("app.services.exchange.order_manager.settings", config),
            patch("app.services.exchange.order_manager.binance_client") as exchange,
            pytest.raises(OrderExecutionError, match="not a fully filled terminal order"),
        ):
            exchange.get_ticker_price = AsyncMock(return_value=Decimal("100"))
            exchange.get_spot_symbol_rules = AsyncMock(return_value=rules)
            exchange.place_market_order = AsyncMock(
                return_value={"orderId": 101, "status": "PARTIALLY_FILLED", "executedQty": "0.5"}
            )
            exchange.get_order_status = AsyncMock(
                return_value={
                    "orderId": 101,
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "0.5",
                    "cummulativeQuoteQty": "50",
                }
            )
            exchange.place_spot_long_exit_oco = AsyncMock()

            await manager._execute_live_order(
                db=database,
                symbol="BTCUSDT",
                side="BUY",
                quantity=1,
                protective_exit=protection,
            )

        exchange.place_spot_long_exit_oco.assert_not_awaited()
        persisted_positions = [
            call.args[0]
            for call in database.add.call_args_list
            if isinstance(call.args[0], Position)
        ]
        assert len(persisted_positions) == 1
        assert float(persisted_positions[0].quantity) == 1.0
        assert persisted_positions[0].protection_status == "MISSING"

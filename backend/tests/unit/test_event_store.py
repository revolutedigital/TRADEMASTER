"""Unit tests for event store and plugin system."""
import pytest
from app.core.event_store import EventStore
from app.core.plugin_system import PluginManager, StrategyPlugin, PluginInfo
from app.services.exchange.advanced_orders import AdvancedOrderManager, BracketConfig, TrailingStopConfig
from decimal import Decimal


class TestEventStore:
    def test_append_and_query(self):
        store = EventStore()
        event = store.append("OrderPlaced", "Order", "order-1", {"symbol": "BTCUSDT", "qty": 1})
        assert event.version == 1
        assert event.event_type == "OrderPlaced"

        events = store.get_events(aggregate_id="order-1")
        assert len(events) == 1

    def test_versioning(self):
        store = EventStore()
        store.append("Created", "Order", "o1", {})
        store.append("Filled", "Order", "o1", {})
        store.append("Created", "Order", "o2", {})

        events = store.get_events(aggregate_id="o1")
        assert len(events) == 2
        assert events[0].version == 1
        assert events[1].version == 2

    def test_subscribe(self):
        store = EventStore()
        received = []
        store.subscribe("OrderFilled", lambda e: received.append(e))

        store.append("OrderPlaced", "Order", "o1", {})
        assert len(received) == 0

        store.append("OrderFilled", "Order", "o1", {})
        assert len(received) == 1

    def test_stats(self):
        store = EventStore()
        store.append("A", "X", "1", {})
        store.append("B", "X", "1", {})
        store.append("A", "X", "2", {})

        stats = store.get_stats()
        assert stats["total_events"] == 3
        assert stats["unique_aggregates"] == 2


class TestPluginManager:
    def test_register_and_list(self):
        pm = PluginManager()
        status = pm.get_status()
        assert status["loaded"] == 0

    def test_discover_nonexistent(self):
        pm = PluginManager(plugins_dir="/nonexistent")
        import asyncio
        discovered = asyncio.get_event_loop().run_until_complete(pm.discover_plugins())
        assert len(discovered) == 0


class TestAdvancedOrders:
    def test_bracket_order(self):
        aom = AdvancedOrderManager()
        config = BracketConfig(
            symbol="BTCUSDT", side="BUY", quantity=Decimal("0.1"),
            entry_price=Decimal("50000"), take_profit_price=Decimal("55000"),
            stop_loss_price=Decimal("48000"),
        )
        result = aom.calculate_bracket_orders(config)
        assert result["entry"]["side"] == "BUY"
        assert result["take_profit"]["side"] == "SELL"
        assert result["stop_loss"]["side"] == "SELL"
        assert result["risk_reward_ratio"] == 2.5  # 5000/2000

    def test_trailing_stop_not_triggered(self):
        aom = AdvancedOrderManager()
        config = TrailingStopConfig(
            symbol="BTCUSDT", side="BUY", quantity=Decimal("1"),
            trail_pct=0.05,
        )
        result = aom.calculate_trailing_stop(config, current_price=53000, peak_price=55000)
        # Stop at 55000 * 0.95 = 52250, price 53000 > 52250 = not triggered
        assert result["stop_price"] == 52250
        assert result["triggered"] is False

    def test_trailing_stop_triggered(self):
        aom = AdvancedOrderManager()
        config = TrailingStopConfig(
            symbol="BTCUSDT", side="BUY", quantity=Decimal("1"),
            trail_pct=0.05,
        )
        result = aom.calculate_trailing_stop(config, current_price=50000, peak_price=55000)
        # Stop at 52250, price at 50000 = triggered
        assert result["triggered"] is True

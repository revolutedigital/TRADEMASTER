"""Tests for multi-exchange adapter infrastructure.

Covers:
- ExchangeAdapter ABC contract enforcement
- Factory: correct adapter creation, caching, error handling
- BinanceAdapter: proper delegation to BinanceClientWrapper
- BybitAdapter: paper trading, NotImplementedError for live methods
- OKXAdapter: stub behavior
- Interface completeness (all abstract methods implemented)
"""

import inspect
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.exchange.adapters.base import ExchangeAdapter, IExchangeAdapter
from app.services.exchange.adapters.binance_adapter import BinanceAdapter
from app.services.exchange.adapters.bybit_adapter import BybitAdapter
from app.services.exchange.adapters.okx import OKXAdapter
from app.services.exchange.factory import (
    get_exchange_adapter,
    list_exchanges,
    register_adapter,
    _ADAPTERS,
    _instances,
)


# =========================================================================
# Test: Abstract interface contract
# =========================================================================


class TestExchangeAdapterInterface:
    """Verify the ExchangeAdapter ABC defines the correct abstract methods."""

    def test_cannot_instantiate_abstract_class(self):
        """ExchangeAdapter is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExchangeAdapter()

    def test_backward_compat_alias(self):
        """IExchangeAdapter is an alias for ExchangeAdapter."""
        assert IExchangeAdapter is ExchangeAdapter

    def test_required_abstract_methods(self):
        """All expected abstract methods are defined in the interface."""
        expected_methods = {
            "connect",
            "disconnect",
            "get_ticker_price",
            "get_klines",
            "get_balance",
            "place_order",
            "place_market_order",
            "cancel_order",
            "get_open_orders",
        }
        # Abstract methods include property getters
        abstract_names = set()
        for name, _ in inspect.getmembers(ExchangeAdapter):
            if getattr(getattr(ExchangeAdapter, name, None), "__isabstractmethod__", False):
                abstract_names.add(name)

        for method in expected_methods:
            assert method in abstract_names, f"Missing abstract method: {method}"

    def test_required_abstract_properties(self):
        """name and is_connected must be abstract properties."""
        for prop_name in ("name", "is_connected"):
            prop = getattr(ExchangeAdapter, prop_name)
            assert isinstance(prop, property), f"{prop_name} should be a property"
            assert getattr(prop.fget, "__isabstractmethod__", False), (
                f"{prop_name} should be abstract"
            )

    def test_get_exchange_name_delegates_to_name(self):
        """get_exchange_name() should delegate to the `name` property."""
        adapter = BinanceAdapter.__new__(BinanceAdapter)
        adapter._client = MagicMock()
        assert adapter.get_exchange_name() == "binance"

    def test_supports_paper_trading_default_false(self):
        """Default supports_paper_trading is False unless overridden."""
        # OKX doesn't override, so it should be False
        okx = OKXAdapter()
        assert okx.supports_paper_trading is False


# =========================================================================
# Test: Factory
# =========================================================================


class TestExchangeFactory:
    """Test the exchange adapter factory."""

    def setup_method(self):
        """Clear cached instances before each test."""
        _instances.clear()

    def test_factory_returns_binance_adapter(self):
        """get_exchange_adapter('binance') returns BinanceAdapter."""
        adapter = get_exchange_adapter("binance", cached=False)
        assert isinstance(adapter, BinanceAdapter)
        assert adapter.name == "binance"

    def test_factory_returns_bybit_adapter(self):
        """get_exchange_adapter('bybit') returns BybitAdapter."""
        adapter = get_exchange_adapter("bybit", cached=False)
        assert isinstance(adapter, BybitAdapter)
        assert adapter.name == "bybit"

    def test_factory_returns_okx_adapter(self):
        """get_exchange_adapter('okx') returns OKXAdapter."""
        adapter = get_exchange_adapter("okx", cached=False)
        assert isinstance(adapter, OKXAdapter)
        assert adapter.name == "okx"

    def test_factory_case_insensitive(self):
        """Factory handles mixed-case exchange names."""
        adapter = get_exchange_adapter("BINANCE", cached=False)
        assert isinstance(adapter, BinanceAdapter)

        adapter2 = get_exchange_adapter("Bybit", cached=False)
        assert isinstance(adapter2, BybitAdapter)

    def test_factory_unknown_exchange_raises(self):
        """Unknown exchange name raises ValueError with available list."""
        with pytest.raises(ValueError, match="Unknown exchange: kraken"):
            get_exchange_adapter("kraken")

    def test_factory_caching(self):
        """Cached mode returns the same instance."""
        adapter1 = get_exchange_adapter("bybit", cached=True)
        adapter2 = get_exchange_adapter("bybit", cached=True)
        assert adapter1 is adapter2

    def test_factory_no_caching(self):
        """Non-cached mode returns different instances."""
        adapter1 = get_exchange_adapter("bybit", cached=False)
        adapter2 = get_exchange_adapter("bybit", cached=False)
        assert adapter1 is not adapter2

    def test_list_exchanges(self):
        """list_exchanges returns sorted list of available exchanges."""
        exchanges = list_exchanges()
        assert "binance" in exchanges
        assert "bybit" in exchanges
        assert "okx" in exchanges
        assert exchanges == sorted(exchanges)

    def test_register_adapter(self):
        """register_adapter adds a new exchange to the factory."""

        class FakeAdapter(ExchangeAdapter):
            async def connect(self): pass
            async def disconnect(self): pass
            @property
            def is_connected(self): return False
            async def get_ticker_price(self, symbol): return Decimal("0")
            async def get_klines(self, symbol, interval="1h", limit=500): pass
            async def get_balance(self, asset="USDT"): return Decimal("0")
            async def place_order(self, symbol, side, quantity, order_type="MARKET", price=None): return {}
            async def place_market_order(self, symbol, side, quantity): return {}
            async def cancel_order(self, symbol, order_id): return {}
            async def get_open_orders(self, symbol=None): return []
            @property
            def name(self): return "fake"

        register_adapter("fake", FakeAdapter)
        assert "fake" in list_exchanges()

        adapter = get_exchange_adapter("fake", cached=False)
        assert isinstance(adapter, FakeAdapter)
        assert adapter.name == "fake"

        # Cleanup
        _ADAPTERS.pop("fake", None)


# =========================================================================
# Test: BinanceAdapter delegation
# =========================================================================


class TestBinanceAdapter:
    """Verify BinanceAdapter delegates correctly to BinanceClientWrapper."""

    def _make_adapter(self):
        """Create a BinanceAdapter with a mock client."""
        mock_client = MagicMock()
        mock_client._client = MagicMock()  # Simulates connected state
        return BinanceAdapter(client=mock_client), mock_client

    def test_name(self):
        adapter, _ = self._make_adapter()
        assert adapter.name == "binance"
        assert adapter.get_exchange_name() == "binance"

    def test_supports_paper_trading(self):
        adapter, _ = self._make_adapter()
        assert adapter.supports_paper_trading is True

    def test_is_connected_true(self):
        adapter, mock = self._make_adapter()
        mock._client = MagicMock()  # Not None = connected
        assert adapter.is_connected is True

    def test_is_connected_false(self):
        adapter, mock = self._make_adapter()
        mock._client = None
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_delegates(self):
        adapter, mock = self._make_adapter()
        mock.connect = AsyncMock()
        await adapter.connect()
        mock.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_delegates(self):
        adapter, mock = self._make_adapter()
        mock.disconnect = AsyncMock()
        await adapter.disconnect()
        mock.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_ticker_price_delegates(self):
        adapter, mock = self._make_adapter()
        mock.get_ticker_price = AsyncMock(return_value=Decimal("85000.50"))
        result = await adapter.get_ticker_price("BTCUSDT")
        assert result == Decimal("85000.50")
        mock.get_ticker_price.assert_awaited_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_klines_delegates(self):
        adapter, mock = self._make_adapter()
        mock.get_klines = AsyncMock(return_value="mock_df")
        result = await adapter.get_klines("BTCUSDT", "1h", 100)
        assert result == "mock_df"
        mock.get_klines.assert_awaited_once_with("BTCUSDT", "1h", 100)

    @pytest.mark.asyncio
    async def test_get_balance_delegates(self):
        adapter, mock = self._make_adapter()
        mock.get_balance = AsyncMock(return_value=Decimal("10000"))
        result = await adapter.get_balance("USDT")
        assert result == Decimal("10000")
        mock.get_balance.assert_awaited_once_with("USDT")

    @pytest.mark.asyncio
    async def test_place_market_order_delegates(self):
        adapter, mock = self._make_adapter()
        expected = {"orderId": 123, "status": "FILLED"}
        mock.place_market_order = AsyncMock(return_value=expected)
        result = await adapter.place_market_order("BTCUSDT", "BUY", 0.001)
        assert result == expected
        mock.place_market_order.assert_awaited_once_with("BTCUSDT", "BUY", 0.001)

    @pytest.mark.asyncio
    async def test_place_order_market_delegates(self):
        adapter, mock = self._make_adapter()
        expected = {"orderId": 123, "status": "FILLED"}
        mock.place_market_order = AsyncMock(return_value=expected)
        result = await adapter.place_order("BTCUSDT", "BUY", 0.001, "MARKET")
        assert result == expected
        mock.place_market_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_place_order_limit_delegates(self):
        adapter, mock = self._make_adapter()
        expected = {"orderId": 456, "status": "NEW"}
        mock.place_limit_order = AsyncMock(return_value=expected)
        result = await adapter.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", 84000.0)
        assert result == expected
        mock.place_limit_order.assert_awaited_once_with("BTCUSDT", "BUY", 0.001, 84000.0)

    @pytest.mark.asyncio
    async def test_place_order_limit_requires_price(self):
        adapter, mock = self._make_adapter()
        with pytest.raises(ValueError, match="Price is required"):
            await adapter.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", None)

    @pytest.mark.asyncio
    async def test_cancel_order_delegates(self):
        adapter, mock = self._make_adapter()
        mock.cancel_order = AsyncMock(return_value={"status": "CANCELLED"})
        result = await adapter.cancel_order("BTCUSDT", 12345)
        assert result["status"] == "CANCELLED"
        mock.cancel_order.assert_awaited_once_with("BTCUSDT", 12345)

    @pytest.mark.asyncio
    async def test_cancel_order_string_id_cast_to_int(self):
        adapter, mock = self._make_adapter()
        mock.cancel_order = AsyncMock(return_value={"status": "CANCELLED"})
        await adapter.cancel_order("BTCUSDT", "12345")
        mock.cancel_order.assert_awaited_once_with("BTCUSDT", 12345)

    @pytest.mark.asyncio
    async def test_get_open_orders_delegates(self):
        adapter, mock = self._make_adapter()
        mock.get_open_orders = AsyncMock(return_value=[{"orderId": 1}])
        result = await adapter.get_open_orders("BTCUSDT")
        assert len(result) == 1
        mock.get_open_orders.assert_awaited_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_open_orders_no_symbol(self):
        adapter, mock = self._make_adapter()
        mock.get_open_orders = AsyncMock(return_value=[])
        result = await adapter.get_open_orders()
        assert result == []
        mock.get_open_orders.assert_awaited_once_with(None)


# =========================================================================
# Test: BybitAdapter
# =========================================================================


class TestBybitAdapter:
    """Verify BybitAdapter paper trading and NotImplementedError for live."""

    def test_name(self):
        adapter = BybitAdapter()
        assert adapter.name == "bybit"
        assert adapter.get_exchange_name() == "bybit"

    def test_supports_paper_trading(self):
        adapter = BybitAdapter(paper_mode=True)
        assert adapter.supports_paper_trading is True

    def test_initial_state(self):
        adapter = BybitAdapter()
        assert adapter.is_connected is False
        assert adapter._paper_mode is True
        assert adapter._paper_balance["USDT"] == Decimal("10000")

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        adapter = BybitAdapter()
        assert adapter.is_connected is False
        await adapter.connect()
        assert adapter.is_connected is True
        await adapter.disconnect()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_get_balance_paper(self):
        adapter = BybitAdapter(paper_mode=True)
        balance = await adapter.get_balance("USDT")
        assert balance == Decimal("10000")

    @pytest.mark.asyncio
    async def test_get_balance_unknown_asset(self):
        adapter = BybitAdapter(paper_mode=True)
        balance = await adapter.get_balance("XRP")
        assert balance == Decimal("0")

    # --- Live mode raises NotImplementedError ---

    @pytest.mark.asyncio
    async def test_live_get_ticker_price_raises(self):
        adapter = BybitAdapter(paper_mode=False)
        with pytest.raises(NotImplementedError, match="Bybit live"):
            await adapter.get_ticker_price("BTCUSDT")

    @pytest.mark.asyncio
    async def test_live_get_klines_raises(self):
        adapter = BybitAdapter(paper_mode=False)
        with pytest.raises(NotImplementedError, match="Bybit kline"):
            await adapter.get_klines("BTCUSDT")

    @pytest.mark.asyncio
    async def test_live_get_balance_raises(self):
        adapter = BybitAdapter(paper_mode=False)
        with pytest.raises(NotImplementedError, match="Bybit live"):
            await adapter.get_balance("USDT")

    @pytest.mark.asyncio
    async def test_live_place_order_raises(self):
        adapter = BybitAdapter(paper_mode=False)
        with pytest.raises(NotImplementedError, match="Bybit live"):
            await adapter.place_order("BTCUSDT", "BUY", 0.001)

    @pytest.mark.asyncio
    async def test_live_cancel_order_raises(self):
        adapter = BybitAdapter(paper_mode=False)
        with pytest.raises(NotImplementedError, match="Bybit live"):
            await adapter.cancel_order("BTCUSDT", "123")

    @pytest.mark.asyncio
    async def test_live_get_open_orders_raises(self):
        adapter = BybitAdapter(paper_mode=False)
        with pytest.raises(NotImplementedError, match="Bybit live"):
            await adapter.get_open_orders()

    # --- Paper trading order execution ---

    @pytest.mark.asyncio
    async def test_paper_market_order_buy(self):
        adapter = BybitAdapter(paper_mode=True)
        # Mock Redis to return a price
        with patch("app.services.exchange.adapters.bybit_adapter.BybitAdapter.get_ticker_price") as mock_price:
            mock_price.return_value = Decimal("85000")
            result = await adapter.place_market_order("BTCUSDT", "BUY", 0.001)

        assert result["status"] == "FILLED"
        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "BUY"
        assert result["paper_mode"] is True
        assert result["exchange"] == "bybit"
        assert "orderId" in result
        assert result["orderId"].startswith("BYBIT-PAPER-")

    @pytest.mark.asyncio
    async def test_paper_market_order_sell(self):
        adapter = BybitAdapter(paper_mode=True)
        # Give some BTC first
        adapter._paper_balance["BTC"] = Decimal("1.0")
        with patch("app.services.exchange.adapters.bybit_adapter.BybitAdapter.get_ticker_price") as mock_price:
            mock_price.return_value = Decimal("85000")
            result = await adapter.place_order("BTCUSDT", "SELL", 0.001)

        assert result["status"] == "FILLED"
        assert result["side"] == "SELL"

    @pytest.mark.asyncio
    async def test_paper_limit_order_requires_price(self):
        adapter = BybitAdapter(paper_mode=True)
        with pytest.raises(ValueError, match="Price is required"):
            await adapter.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", None)

    @pytest.mark.asyncio
    async def test_paper_limit_order_uses_specified_price(self):
        adapter = BybitAdapter(paper_mode=True)
        result = await adapter.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", 80000.0)
        assert result["status"] == "FILLED"
        assert result["avgPrice"] == "80000.00"

    @pytest.mark.asyncio
    async def test_paper_order_updates_balance(self):
        adapter = BybitAdapter(paper_mode=True)
        initial_usdt = adapter._paper_balance["USDT"]

        result = await adapter.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", 80000.0)

        # USDT should decrease (cost + commission)
        assert adapter._paper_balance["USDT"] < initial_usdt
        # BTC should increase
        assert adapter._paper_balance.get("BTC", Decimal("0")) > Decimal("0")

    @pytest.mark.asyncio
    async def test_paper_cancel_order(self):
        adapter = BybitAdapter(paper_mode=True)
        # Add a fake open order
        adapter._paper_orders.append({
            "orderId": "BYBIT-PAPER-99",
            "symbol": "BTCUSDT",
            "status": "NEW",
        })

        result = await adapter.cancel_order("BTCUSDT", "BYBIT-PAPER-99")
        assert result["status"] == "CANCELLED"

    @pytest.mark.asyncio
    async def test_paper_cancel_nonexistent_order(self):
        adapter = BybitAdapter(paper_mode=True)
        result = await adapter.cancel_order("BTCUSDT", "doesnt-exist")
        assert result["status"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_paper_get_open_orders(self):
        adapter = BybitAdapter(paper_mode=True)
        adapter._paper_orders = [
            {"orderId": "1", "symbol": "BTCUSDT", "status": "NEW"},
            {"orderId": "2", "symbol": "ETHUSDT", "status": "NEW"},
            {"orderId": "3", "symbol": "BTCUSDT", "status": "FILLED"},
        ]
        orders = await adapter.get_open_orders("BTCUSDT")
        assert len(orders) == 1
        assert orders[0]["orderId"] == "1"

    @pytest.mark.asyncio
    async def test_paper_get_open_orders_all(self):
        adapter = BybitAdapter(paper_mode=True)
        adapter._paper_orders = [
            {"orderId": "1", "symbol": "BTCUSDT", "status": "NEW"},
            {"orderId": "2", "symbol": "ETHUSDT", "status": "NEW"},
        ]
        orders = await adapter.get_open_orders()
        assert len(orders) == 2

    @pytest.mark.asyncio
    async def test_paper_order_counter_increments(self):
        adapter = BybitAdapter(paper_mode=True)
        result1 = await adapter.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", 80000.0)
        result2 = await adapter.place_order("BTCUSDT", "BUY", 0.001, "LIMIT", 80000.0)
        assert result1["orderId"] != result2["orderId"]
        assert adapter._paper_order_counter == 2


# =========================================================================
# Test: OKXAdapter
# =========================================================================


class TestOKXAdapter:
    """Verify OKXAdapter stub raises NotImplementedError on all live methods."""

    def test_name(self):
        adapter = OKXAdapter()
        assert adapter.name == "okx"
        assert adapter.get_exchange_name() == "okx"

    def test_supports_paper_trading_false(self):
        adapter = OKXAdapter()
        assert adapter.supports_paper_trading is False

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        adapter = OKXAdapter()
        await adapter.connect()
        assert adapter.is_connected is True
        await adapter.disconnect()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_get_ticker_price_raises(self):
        adapter = OKXAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.get_ticker_price("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_klines_raises(self):
        adapter = OKXAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.get_klines("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_balance_raises(self):
        adapter = OKXAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.get_balance()

    @pytest.mark.asyncio
    async def test_place_order_raises(self):
        adapter = OKXAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.place_order("BTCUSDT", "BUY", 0.001)

    @pytest.mark.asyncio
    async def test_place_market_order_raises(self):
        adapter = OKXAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.place_market_order("BTCUSDT", "BUY", 0.001)

    @pytest.mark.asyncio
    async def test_cancel_order_raises(self):
        adapter = OKXAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.cancel_order("BTCUSDT", "123")

    @pytest.mark.asyncio
    async def test_get_open_orders_raises(self):
        adapter = OKXAdapter()
        with pytest.raises(NotImplementedError):
            await adapter.get_open_orders()


# =========================================================================
# Test: Interface completeness across all adapters
# =========================================================================


class TestAdapterCompleteness:
    """Ensure all adapters implement every method from ExchangeAdapter."""

    @pytest.mark.parametrize("adapter_cls", [BinanceAdapter, BybitAdapter, OKXAdapter])
    def test_adapter_is_subclass(self, adapter_cls):
        """All adapters are subclasses of ExchangeAdapter."""
        assert issubclass(adapter_cls, ExchangeAdapter)

    @pytest.mark.parametrize("adapter_cls", [BinanceAdapter, BybitAdapter, OKXAdapter])
    def test_adapter_instantiable(self, adapter_cls):
        """All adapters can be instantiated (no missing abstract methods)."""
        # BinanceAdapter needs special handling since it uses the global singleton
        if adapter_cls is BinanceAdapter:
            adapter = adapter_cls(client=MagicMock())
        else:
            adapter = adapter_cls()
        assert adapter is not None

    @pytest.mark.parametrize("adapter_cls", [BinanceAdapter, BybitAdapter, OKXAdapter])
    def test_adapter_has_all_interface_methods(self, adapter_cls):
        """All abstract methods from ExchangeAdapter are implemented."""
        abstract_methods = set()
        for name, _ in inspect.getmembers(ExchangeAdapter):
            if getattr(getattr(ExchangeAdapter, name, None), "__isabstractmethod__", False):
                abstract_methods.add(name)

        for method in abstract_methods:
            assert hasattr(adapter_cls, method), (
                f"{adapter_cls.__name__} missing method: {method}"
            )
            # Ensure it's not still abstract
            impl = getattr(adapter_cls, method)
            if isinstance(impl, property):
                assert not getattr(impl.fget, "__isabstractmethod__", False), (
                    f"{adapter_cls.__name__}.{method} is still abstract"
                )
            else:
                assert not getattr(impl, "__isabstractmethod__", False), (
                    f"{adapter_cls.__name__}.{method} is still abstract"
                )

    def test_all_adapters_registered_in_factory(self):
        """Every adapter class has a corresponding factory registration."""
        assert "binance" in _ADAPTERS
        assert "bybit" in _ADAPTERS
        assert "okx" in _ADAPTERS
        assert _ADAPTERS["binance"] is BinanceAdapter
        assert _ADAPTERS["bybit"] is BybitAdapter
        assert _ADAPTERS["okx"] is OKXAdapter

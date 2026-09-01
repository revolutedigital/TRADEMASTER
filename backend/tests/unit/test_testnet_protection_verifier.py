"""Tests for the explicit, Testnet-only native OCO release proof."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import TradingExecutionMode
from app.services.exchange.binance_client import NativeSpotOcoProtection
from app.services.exchange.live_execution_readiness import testnet_protection_readiness
from app.services.exchange.spot_rules import SpotSymbolRules
from app.services.exchange.testnet_protection_verifier import (
    TestnetProtectionVerificationError as NativeOcoVerificationError,
)
from app.services.exchange.testnet_protection_verifier import (
    TestnetProtectionVerifier as NativeOcoVerifier,
)


def _rules() -> SpotSymbolRules:
    return SpotSymbolRules.from_exchange_info(
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "minQty": "0.00010",
                    "maxQty": "100.00000",
                    "stepSize": "0.00010",
                },
                {"filterType": "NOTIONAL", "minNotional": "10.00"},
            ],
        }
    )


class _TestnetExchange:
    async def get_account(self) -> dict:
        return {"canTrade": True}

    async def get_spot_symbol_rules(self, symbol: str) -> SpotSymbolRules:
        assert symbol == "BTCUSDT"
        return _rules()

    async def get_ticker_price(self, symbol: str) -> Decimal:
        assert symbol == "BTCUSDT"
        return Decimal("90000")

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict:
        assert symbol == "BTCUSDT"
        assert quantity == Decimal("0.00020")
        assert client_order_id.startswith("TM-TV-")
        if side == "BUY":
            return {"orderId": 101, "executedQty": "0.00020"}
        return {"orderId": 102, "executedQty": "0.00020"}

    async def place_spot_long_exit_oco(self, **kwargs) -> NativeSpotOcoProtection:
        assert kwargs["quantity"] == Decimal("0.00020")
        return NativeSpotOcoProtection(
            order_list_id=77,
            protected_quantity=Decimal("0.00020"),
            response={"orderListId": 77},
        )

    async def get_open_spot_order_lists(self) -> list[dict]:
        return [
            {
                "orderListId": 77,
                "symbol": "BTCUSDT",
                "contingencyType": "OCO",
                "listStatusType": "EXEC_STARTED",
                "listOrderStatus": "EXECUTING",
            }
        ]

    async def cancel_spot_order_list(self, symbol: str, order_list_id: int) -> dict:
        assert symbol == "BTCUSDT"
        assert order_list_id == 77
        return {"orderListId": 77}

    async def get_spot_order_list(self, order_list_id: int) -> dict:
        assert order_list_id == 77
        return {
            "orderListId": 77,
            "symbol": "BTCUSDT",
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "orders": [{"orderId": 701}, {"orderId": 702}],
        }

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        assert symbol == "BTCUSDT"
        if order_id == 101:
            return {
                "orderId": 101,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "status": "FILLED",
                "executedQty": "0.00020",
            }
        if order_id == 102:
            return {
                "orderId": 102,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "status": "FILLED",
                "executedQty": "0.00020",
            }
        assert order_id in {701, 702}
        return {
            "orderId": order_id,
            "orderListId": 77,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "status": "CANCELED",
            "executedQty": "0",
        }


class _FailedProofCleanupExchange(_TestnetExchange):
    def __init__(self) -> None:
        self.exit_attempted = False

    async def get_open_spot_order_lists(self) -> list[dict]:
        return []

    async def cancel_spot_order_list(self, symbol: str, order_list_id: int) -> dict:
        raise RuntimeError("request timed out after Binance accepted cancellation")

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict:
        result = await super().place_market_order(symbol, side, quantity, client_order_id)
        if side == "SELL" and client_order_id.startswith("TM-TV-C-"):
            self.exit_attempted = True
        return result


class _FilledOcoExchange(_TestnetExchange):
    def __init__(self) -> None:
        self.sell_attempted = False

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        if order_id in {101, 102}:
            return await super().get_order_status(symbol, order_id)
        if order_id == 702:
            return await super().get_order_status(symbol, order_id)
        return {
            "orderId": order_id,
            "orderListId": 77,
            "symbol": symbol,
            "side": "SELL",
            "status": "FILLED",
            "executedQty": "0.00020",
        }

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict:
        if side == "SELL":
            self.sell_attempted = True
        return await super().place_market_order(symbol, side, quantity, client_order_id)


class _UnconfirmedEntryExchange(_TestnetExchange):
    def __init__(self) -> None:
        self.cleanup_sell_attempted = False

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        if order_id == 101:
            return {
                "orderId": 101,
                "symbol": symbol,
                "side": "BUY",
                "status": "NEW",
                "executedQty": "0",
            }
        return await super().get_order_status(symbol, order_id)

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict:
        if side == "SELL" and client_order_id.startswith("TM-TV-C-"):
            self.cleanup_sell_attempted = True
        return await super().place_market_order(symbol, side, quantity, client_order_id)


class _UnconfirmedCleanupExitExchange(_TestnetExchange):
    def __init__(self) -> None:
        self.sell_calls = 0

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        if order_id == 102:
            return {
                "orderId": 102,
                "symbol": symbol,
                "side": "SELL",
                "status": "NEW",
                "executedQty": "0",
            }
        return await super().get_order_status(symbol, order_id)

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict:
        if side == "SELL":
            self.sell_calls += 1
        return await super().place_market_order(symbol, side, quantity, client_order_id)


class _PartialOcoCoverageExchange(_TestnetExchange):
    def __init__(self) -> None:
        self.cancel_calls = 0

    async def place_spot_long_exit_oco(self, **kwargs) -> NativeSpotOcoProtection:
        assert kwargs["quantity"] == Decimal("0.00020")
        return NativeSpotOcoProtection(
            order_list_id=77,
            protected_quantity=Decimal("0.00010"),
            response={"orderListId": 77},
        )

    async def cancel_spot_order_list(self, symbol: str, order_list_id: int) -> dict:
        self.cancel_calls += 1
        return await super().cancel_spot_order_list(symbol, order_list_id)


@pytest.fixture(autouse=True)
def reset_testnet_readiness():
    testnet_protection_readiness.reset()
    yield
    testnet_protection_readiness.reset()


@pytest.mark.asyncio
async def test_verifier_records_only_a_complete_testnet_oco_lifecycle() -> None:
    database = AsyncMock()
    database.add = MagicMock()
    database.flush = AsyncMock()
    settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.TESTNET,
        symbols_list=["BTCUSDT"],
        live_trading_testnet_verification_max_age_days=30,
    )

    with patch("app.services.exchange.testnet_protection_verifier.settings", settings):
        report = await NativeOcoVerifier(_TestnetExchange()).verify(database, "BTCUSDT")

    assert report.symbol == "BTCUSDT"
    assert report.order_list_id == 77
    assert report.exit_order_id == "102"
    database.add.assert_called_once()
    database.flush.assert_awaited_once()
    assert testnet_protection_readiness.is_ready(30 * 86_400) is True


@pytest.mark.asyncio
async def test_verifier_refuses_any_environment_other_than_testnet() -> None:
    database = AsyncMock()
    settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.LIVE,
        symbols_list=["BTCUSDT"],
        live_trading_testnet_verification_max_age_days=30,
    )

    with (
        patch("app.services.exchange.testnet_protection_verifier.settings", settings),
        pytest.raises(
            NativeOcoVerificationError,
            match="only runs while execution mode is TESTNET",
        ),
    ):
        await NativeOcoVerifier(_TestnetExchange()).verify(database, "BTCUSDT")


@pytest.mark.asyncio
async def test_verifier_cleans_up_when_cancel_response_fails_after_terminal_cancellation() -> None:
    database = AsyncMock()
    exchange = _FailedProofCleanupExchange()
    settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.TESTNET,
        symbols_list=["BTCUSDT"],
        live_trading_testnet_verification_max_age_days=30,
    )

    with (
        patch("app.services.exchange.testnet_protection_verifier.settings", settings),
        pytest.raises(
            NativeOcoVerificationError,
            match="not visible as an active Binance order list",
        ),
    ):
        await NativeOcoVerifier(exchange).verify(database, "BTCUSDT")

    assert exchange.exit_attempted is True


@pytest.mark.asyncio
async def test_verifier_never_sells_again_when_the_oco_already_filled() -> None:
    database = AsyncMock()
    exchange = _FilledOcoExchange()
    settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.TESTNET,
        symbols_list=["BTCUSDT"],
        live_trading_testnet_verification_max_age_days=30,
    )

    with (
        patch("app.services.exchange.testnet_protection_verifier.settings", settings),
        pytest.raises(
            NativeOcoVerificationError,
            match="filled before the controlled cleanup market exit",
        ),
    ):
        await NativeOcoVerifier(exchange).verify(database, "BTCUSDT")

    assert exchange.sell_attempted is False


@pytest.mark.asyncio
async def test_verifier_attempts_cleanup_when_the_buy_signed_read_is_not_confirmed() -> None:
    database = AsyncMock()
    exchange = _UnconfirmedEntryExchange()
    settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.TESTNET,
        symbols_list=["BTCUSDT"],
        live_trading_testnet_verification_max_age_days=30,
    )

    with (
        patch("app.services.exchange.testnet_protection_verifier.settings", settings),
        pytest.raises(NativeOcoVerificationError, match="signed read did not confirm"),
    ):
        await NativeOcoVerifier(exchange).verify(database, "BTCUSDT")

    assert exchange.cleanup_sell_attempted is True


@pytest.mark.asyncio
async def test_verifier_never_sends_a_second_cleanup_sell_after_an_unconfirmed_exit_submission() -> None:
    database = AsyncMock()
    exchange = _UnconfirmedCleanupExitExchange()
    settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.TESTNET,
        symbols_list=["BTCUSDT"],
        live_trading_testnet_verification_max_age_days=30,
    )

    with (
        patch("app.services.exchange.testnet_protection_verifier.settings", settings),
        pytest.raises(NativeOcoVerificationError, match="signed read did not confirm"),
    ):
        await NativeOcoVerifier(exchange).verify(database, "BTCUSDT")

    assert exchange.sell_calls == 1


@pytest.mark.asyncio
async def test_partial_oco_coverage_is_cancelled_before_testnet_cleanup_sell() -> None:
    database = AsyncMock()
    exchange = _PartialOcoCoverageExchange()
    settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.TESTNET,
        symbols_list=["BTCUSDT"],
        live_trading_testnet_verification_max_age_days=30,
    )

    with (
        patch("app.services.exchange.testnet_protection_verifier.settings", settings),
        pytest.raises(NativeOcoVerificationError, match="did not cover the complete market fill"),
    ):
        await NativeOcoVerifier(exchange).verify(database, "BTCUSDT")

    assert exchange.cancel_calls == 1

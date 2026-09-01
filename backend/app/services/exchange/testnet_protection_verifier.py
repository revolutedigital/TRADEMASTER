"""Controlled Binance Spot Testnet proof for the native OCO protection path."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode, settings
from app.core.logging import get_logger
from app.models.execution_release import ExecutionReleaseCheck
from app.services.exchange.binance_client import NativeSpotOcoProtection, binance_client
from app.services.exchange.live_execution_readiness import testnet_protection_readiness

logger = get_logger(__name__)

TESTNET_OCO_RELEASE_CHECK = "BINANCE_SPOT_NATIVE_OCO_TESTNET"


class TestnetProtectionVerificationError(Exception):
    """Raised when the controlled Testnet OCO proof cannot complete safely."""


class TestnetVerificationExchange(Protocol):
    async def get_account(self) -> dict[str, Any]: ...

    async def get_spot_symbol_rules(self, symbol: str): ...

    async def get_ticker_price(self, symbol: str) -> Decimal: ...

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal, client_order_id: str
    ) -> dict[str, Any]: ...

    async def place_spot_long_exit_oco(
        self,
        *,
        symbol: str,
        last_price: Decimal,
        quantity: Decimal,
        take_profit_price: Decimal,
        stop_loss_price: Decimal,
        client_order_id: str,
    ) -> NativeSpotOcoProtection: ...

    async def get_open_spot_order_lists(self) -> list[dict[str, Any]]: ...

    async def cancel_spot_order_list(self, symbol: str, order_list_id: int) -> dict[str, Any]: ...

    async def get_spot_order_list(self, order_list_id: int) -> dict[str, Any]: ...

    async def get_order_status(self, symbol: str, order_id: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TestnetProtectionVerificationReport:
    symbol: str
    entry_order_id: str
    order_list_id: int
    exit_order_id: str
    verified_at: datetime

    def as_dict(self) -> dict[str, str | int]:
        return {
            "status": "PASSED",
            "environment": TradingExecutionMode.TESTNET.value,
            "symbol": self.symbol,
            "entry_order_id": self.entry_order_id,
            "order_list_id": self.order_list_id,
            "exit_order_id": self.exit_order_id,
            "verified_at": self.verified_at.isoformat(),
        }


@dataclass(frozen=True)
class _TerminalOcoState:
    kind: str
    filled_quantity: Decimal = Decimal("0")


class TestnetProtectionVerifier:
    """Prove BUY -> native OCO -> signed read -> cancel -> SELL only on Testnet."""

    def __init__(self, exchange: TestnetVerificationExchange | None = None) -> None:
        self._exchange = exchange or binance_client

    async def verify(
        self, db: AsyncSession, symbol: str
    ) -> TestnetProtectionVerificationReport:
        normalized_symbol = symbol.upper()
        self._require_testnet(normalized_symbol)
        entry_order_id: str | None = None
        order_list_id: int | None = None
        remaining_quantity = Decimal("0")
        oco_active = False
        market_exit_submitted = False
        try:
            account = await self._exchange.get_account()
            if not account.get("canTrade", False):
                raise TestnetProtectionVerificationError(
                    "Binance Testnet credentials do not have Spot trading permission"
                )
            rules = await self._exchange.get_spot_symbol_rules(normalized_symbol)
            entry_price = await self._exchange.get_ticker_price(normalized_symbol)
            entry_quantity = rules.minimum_market_quantity_for_notional(entry_price)
            entry = await self._exchange.place_market_order(
                normalized_symbol,
                "BUY",
                entry_quantity,
                client_order_id=f"TM-TV-E-{_token()}",
            )
            # A rejected or unreadable response is not proof that the exchange
            # did not fill the BUY. Cleanup must conservatively account for the
            # full requested Testnet quantity until its signed status is known.
            remaining_quantity = entry_quantity
            entry_order_id = await self._verify_full_market_fill(
                symbol=normalized_symbol,
                submitted_order=entry,
                expected_side="BUY",
                expected_quantity=entry_quantity,
            )
            remaining_quantity = entry_quantity

            last_price = await self._exchange.get_ticker_price(normalized_symbol)
            protection = await self._exchange.place_spot_long_exit_oco(
                symbol=normalized_symbol,
                last_price=last_price,
                quantity=remaining_quantity,
                take_profit_price=last_price * Decimal("1.05"),
                stop_loss_price=last_price * Decimal("0.95"),
                client_order_id=f"TM-P-{_token()}",
            )
            order_list_id = protection.order_list_id
            # The exchange may already have accepted a partial OCO. From this
            # point cleanup must cancel the exact list before it ever considers
            # a reducing market sell.
            oco_active = True
            if protection.protected_quantity != remaining_quantity:
                raise TestnetProtectionVerificationError(
                    "Testnet native OCO did not cover the complete market fill"
                )
            # The OCO exists and is visible as an active native exit. Any later
            # failure must cancel it before it attempts a market cleanup sell.
            open_lists = await self._exchange.get_open_spot_order_lists()
            if not _has_active_oco(open_lists, order_list_id, normalized_symbol):
                raise TestnetProtectionVerificationError(
                    "Testnet native OCO was not visible as an active Binance order list"
                )

            await self._exchange.cancel_spot_order_list(normalized_symbol, order_list_id)
            terminal_state = await self._terminal_oco_state(normalized_symbol, order_list_id)
            if terminal_state.kind == "SELL_FILLED":
                remaining_quantity = max(
                    Decimal("0"), remaining_quantity - terminal_state.filled_quantity
                )
                if remaining_quantity > 0:
                    submitted_cleanup = await self._exchange.place_market_order(
                        normalized_symbol,
                        "SELL",
                        remaining_quantity,
                        client_order_id=f"TM-TV-C-{_token()}",
                    )
                    market_exit_submitted = True
                    await self._verify_full_market_fill(
                        symbol=normalized_symbol,
                        submitted_order=submitted_cleanup,
                        expected_side="SELL",
                        expected_quantity=remaining_quantity,
                    )
                    remaining_quantity = Decimal("0")
                raise TestnetProtectionVerificationError(
                    "Testnet OCO filled before the controlled cleanup market exit"
                )
            if terminal_state.kind != "CANCELLED":
                raise TestnetProtectionVerificationError(
                    "Testnet OCO cancellation could not be confirmed without a sell fill"
                )
            oco_active = False
            submitted_exit = await self._exchange.place_market_order(
                normalized_symbol,
                "SELL",
                remaining_quantity,
                client_order_id=f"TM-TV-X-{_token()}",
            )
            market_exit_submitted = True
            exit_order_id = await self._verify_full_market_fill(
                symbol=normalized_symbol,
                submitted_order=submitted_exit,
                expected_side="SELL",
                expected_quantity=remaining_quantity,
            )
            remaining_quantity = Decimal("0")
        except Exception as exc:
            if market_exit_submitted:
                logger.critical(
                    "testnet_native_oco_cleanup_market_exit_unverified",
                    symbol=normalized_symbol,
                    entry_order_id=entry_order_id,
                    order_list_id=order_list_id,
                    reason="a Testnet market exit was submitted but its signed state was not confirmed; no second sell was sent",
                )
            else:
                await self._best_effort_cleanup(
                    symbol=normalized_symbol,
                    order_list_id=order_list_id if oco_active else None,
                    remaining_quantity=remaining_quantity,
                )
            message = str(exc)
            testnet_protection_readiness.mark_error(message)
            logger.critical(
                "testnet_native_oco_verification_failed",
                symbol=normalized_symbol,
                entry_order_id=entry_order_id,
                order_list_id=order_list_id,
                error=message,
                exc_info=True,
            )
            if isinstance(exc, TestnetProtectionVerificationError):
                raise
            raise TestnetProtectionVerificationError(message) from exc

        verified_at = datetime.now(UTC)
        release_check = ExecutionReleaseCheck(
            check_name=TESTNET_OCO_RELEASE_CHECK,
            environment=TradingExecutionMode.TESTNET.value,
            status="PASSED",
            symbol=normalized_symbol,
            entry_exchange_order_id=entry_order_id,
            protective_order_list_id=order_list_id,
            exit_exchange_order_id=exit_order_id,
            verified_at=verified_at,
        )
        db.add(release_check)
        await db.flush()
        testnet_protection_readiness.mark_ready(verified_at)
        logger.info(
            "testnet_native_oco_verification_passed",
            symbol=normalized_symbol,
            entry_order_id=entry_order_id,
            order_list_id=order_list_id,
            exit_order_id=exit_order_id,
        )
        return TestnetProtectionVerificationReport(
            symbol=normalized_symbol,
            entry_order_id=entry_order_id,
            order_list_id=order_list_id,
            exit_order_id=exit_order_id,
            verified_at=verified_at,
        )

    async def load_live_readiness(self, db: AsyncSession) -> bool:
        """Load recent persistent Testnet proof before a live process can be armed."""
        cutoff = datetime.now(UTC) - timedelta(
            days=settings.live_trading_testnet_verification_max_age_days
        )
        result = await db.execute(
            select(ExecutionReleaseCheck)
            .where(
                ExecutionReleaseCheck.check_name == TESTNET_OCO_RELEASE_CHECK,
                ExecutionReleaseCheck.environment == TradingExecutionMode.TESTNET.value,
                ExecutionReleaseCheck.status == "PASSED",
                ExecutionReleaseCheck.verified_at >= cutoff,
            )
            .order_by(ExecutionReleaseCheck.verified_at.desc())
            .limit(1)
        )
        check = result.scalar_one_or_none()
        if check is None:
            testnet_protection_readiness.mark_unresolved(
                ["no recent successful Binance Spot Testnet OCO verification"]
            )
            return False
        testnet_protection_readiness.mark_ready(check.verified_at)
        return True

    async def _verify_full_market_fill(
        self,
        *,
        symbol: str,
        submitted_order: dict[str, Any],
        expected_side: str,
        expected_quantity: Decimal,
    ) -> str:
        """Verify the signed exchange state of a fully filled Testnet market order."""
        exchange_order_id = str(submitted_order.get("orderId", ""))
        try:
            numeric_order_id = int(exchange_order_id)
        except ValueError as exc:
            raise TestnetProtectionVerificationError(
                "Testnet market order did not return a valid order id"
            ) from exc
        verified_order = await self._exchange.get_order_status(symbol, numeric_order_id)
        if (
            str(verified_order.get("orderId", "")) != exchange_order_id
            or verified_order.get("symbol") != symbol
            or verified_order.get("side") != expected_side
            or verified_order.get("status") != "FILLED"
            or _decimal(verified_order.get("executedQty")) != expected_quantity
        ):
            raise TestnetProtectionVerificationError(
                "Testnet market order signed read did not confirm the complete expected fill"
            )
        return exchange_order_id

    async def _submit_and_verify_cleanup_sell(
        self,
        *,
        symbol: str,
        quantity: Decimal,
        client_order_id: str,
    ) -> str:
        """Submit a reducing Testnet sell and only accept a signed full-fill confirmation."""
        submitted_order = await self._exchange.place_market_order(
            symbol,
            "SELL",
            quantity,
            client_order_id=client_order_id,
        )
        return await self._verify_full_market_fill(
            symbol=symbol,
            submitted_order=submitted_order,
            expected_side="SELL",
            expected_quantity=quantity,
        )

    async def _terminal_oco_state(
        self, symbol: str, order_list_id: int
    ) -> _TerminalOcoState:
        """Confirm every OCO child through signed reads before allowing a naked cleanup sell."""
        order_list = await self._exchange.get_spot_order_list(order_list_id)
        if (
            _order_list_id(order_list) != order_list_id
            or order_list.get("symbol") != symbol
            or order_list.get("contingencyType") != "OCO"
            or not _is_terminal_oco(order_list)
        ):
            return _TerminalOcoState("AMBIGUOUS")
        order_ids = _child_order_ids(order_list)
        if len(order_ids) != 2:
            return _TerminalOcoState("AMBIGUOUS")
        child_orders = [
            await self._exchange.get_order_status(symbol, order_id) for order_id in order_ids
        ]
        filled_orders = [
            order
            for order in child_orders
            if order.get("side") == "SELL" and _decimal(order.get("executedQty")) > 0
        ]
        if filled_orders:
            if (
                len(filled_orders) != 1
                or int(filled_orders[0].get("orderListId", -1)) != order_list_id
            ):
                return _TerminalOcoState("AMBIGUOUS")
            return _TerminalOcoState(
                "SELL_FILLED",
                filled_quantity=_decimal(filled_orders[0].get("executedQty")),
            )

        if any(
            order.get("side") != "SELL"
            or int(order.get("orderListId", -1)) != order_list_id
            or order.get("status") not in {"CANCELED", "EXPIRED"}
            or _decimal(order.get("executedQty")) != 0
            for order in child_orders
        ):
            return _TerminalOcoState("AMBIGUOUS")
        return _TerminalOcoState("CANCELLED")

    @staticmethod
    def _require_testnet(symbol: str) -> None:
        if settings.execution_mode != TradingExecutionMode.TESTNET:
            raise TestnetProtectionVerificationError(
                "native OCO verification only runs while execution mode is TESTNET"
            )
        if symbol not in settings.symbols_list:
            raise TestnetProtectionVerificationError(
                "verification symbol must be included in TRADING_SYMBOLS"
            )

    async def _best_effort_cleanup(
        self,
        *,
        symbol: str,
        order_list_id: int | None,
        remaining_quantity: Decimal,
    ) -> None:
        if order_list_id is not None:
            try:
                await self._exchange.cancel_spot_order_list(symbol, order_list_id)
            except Exception as exc:
                logger.critical(
                    "testnet_native_oco_cleanup_cancel_failed",
                    symbol=symbol,
                    order_list_id=order_list_id,
                    error=str(exc),
                )
            try:
                terminal_state = await self._terminal_oco_state(symbol, order_list_id)
            except Exception as lookup_exc:
                logger.critical(
                    "testnet_native_oco_cleanup_state_lookup_failed",
                    symbol=symbol,
                    order_list_id=order_list_id,
                    error=str(lookup_exc),
                )
                return
            if terminal_state.kind == "SELL_FILLED":
                logger.critical(
                    "testnet_native_oco_cleanup_already_exited",
                    symbol=symbol,
                    order_list_id=order_list_id,
                )
                return
            if terminal_state.kind != "CANCELLED":
                logger.critical(
                    "testnet_native_oco_cleanup_not_cancelled",
                    symbol=symbol,
                    order_list_id=order_list_id,
                )
                return
        if remaining_quantity <= 0:
            return
        try:
            await self._submit_and_verify_cleanup_sell(
                symbol=symbol,
                quantity=remaining_quantity,
                client_order_id=f"TM-TV-C-{_token()}",
            )
        except Exception as exc:
            logger.critical(
                "testnet_native_oco_cleanup_exit_failed",
                symbol=symbol,
                quantity=str(remaining_quantity),
                error=str(exc),
            )


def _has_active_oco(order_lists: list[dict[str, Any]], order_list_id: int, symbol: str) -> bool:
    return any(
        _order_list_id(order_list) == order_list_id
        and order_list.get("symbol") == symbol
        and order_list.get("contingencyType") == "OCO"
        and order_list.get("listStatusType") == "EXEC_STARTED"
        and order_list.get("listOrderStatus") == "EXECUTING"
        for order_list in order_lists
    )


def _is_terminal_oco(order_list: dict[str, Any]) -> bool:
    return (
        order_list.get("listStatusType") == "ALL_DONE"
        and order_list.get("listOrderStatus") == "ALL_DONE"
    )


def _child_order_ids(order_list: dict[str, Any]) -> list[int]:
    order_ids: list[int] = []
    for order in order_list.get("orders", []):
        try:
            order_ids.append(int(order["orderId"]))
        except (KeyError, TypeError, ValueError):
            return []
    return order_ids


def _order_list_id(order_list: dict[str, Any]) -> int | None:
    try:
        return int(order_list["orderListId"])
    except (KeyError, TypeError, ValueError):
        return None


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value or 0))
    except Exception:
        return Decimal("0")


def _token() -> str:
    import uuid

    return uuid.uuid4().hex[:20]


testnet_protection_verifier = TestnetProtectionVerifier()

"""Order lifecycle management: signal to execution.

Supports both live (Binance) and paper trading modes.
Paper mode simulates fills with realistic slippage and commission.

All financial calculations use Decimal for precision — IEEE 754 float
rounding errors are unacceptable in a trading system.
"""

import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode, settings
from app.core.events import Event, EventType, event_bus
from app.core.exceptions import OrderExecutionError
from app.core.logging import get_logger
from app.models.portfolio import Position
from app.models.trade import Order, OrderStatus, OrderType
from app.services.exchange.binance_client import binance_client
from app.services.exchange.live_trading_guard import (
    LiveTradingSafetyError,
    live_trading_guard,
)

logger = get_logger(__name__)

# Paper trading simulation parameters (Decimal for precision)
PAPER_COMMISSION_RATE = Decimal("0.001")  # 0.1% taker fee (Binance standard)
PAPER_SLIPPAGE_BPS = 5  # 5 basis points max (0.05%) slippage


@dataclass(frozen=True)
class SpotLongProtectionPlan:
    """Native exit levels that must exist before a live Spot long is opened."""

    stop_loss_price: Decimal
    take_profit_price: Decimal


@dataclass(frozen=True)
class EmergencySpotExitResult:
    """What the exchange confirmed after an emergency Spot exit attempt."""

    filled_quantity: Decimal
    exchange_order_id: str | None


class OrderManager:
    """Manages the full order lifecycle from signal to fill."""

    async def execute_market_order(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
        protective_exit: SpotLongProtectionPlan | None = None,
    ) -> Order:
        """Execute a market order. Routes to paper or live mode based on config."""
        if settings.paper_mode:
            return await self._execute_paper_order(db, symbol, side, quantity, signal_id)
        return await self._execute_live_order(
            db,
            symbol,
            side,
            quantity,
            signal_id,
            protective_exit,
        )

    async def _reserve_live_notional(
        self,
        db: AsyncSession,
        estimated_notional: Decimal,
    ) -> None:
        """Reserve daily live notional inside a PostgreSQL transaction lock.

        The advisory lock serializes the cap check across application workers.
        The pending order is flushed before this transaction is released, so a
        concurrent request cannot oversubscribe the daily live budget.
        """
        try:
            await db.execute(
                text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": 729_451_003},
            )
        except Exception as exc:
            raise LiveTradingSafetyError("could not acquire the live execution lock") from exc

        day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        result = await db.execute(
            select(func.coalesce(func.sum(Order.price * Order.quantity), 0)).where(
                Order.execution_mode == TradingExecutionMode.LIVE.value,
                Order.created_at >= day_start,
                Order.status.in_(
                    [
                        OrderStatus.PENDING,
                        OrderStatus.SUBMITTED,
                        OrderStatus.PARTIALLY_FILLED,
                        OrderStatus.FILLED,
                    ]
                ),
            )
        )
        used_notional = Decimal(str(result.scalar_one() or 0))
        daily_limit = Decimal(str(settings.live_trading_max_daily_notional))
        if used_notional + estimated_notional > daily_limit:
            raise LiveTradingSafetyError("order would exceed LIVE_TRADING_MAX_DAILY_NOTIONAL")

    async def _execute_paper_order(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
    ) -> Order:
        """Simulate a market order with realistic slippage and fees.

        All price/commission arithmetic uses Decimal to avoid float rounding
        errors that can accumulate over many trades.
        """
        qty = Decimal(str(quantity))

        # Get current price from Binance (already returns Decimal)
        try:
            current_price = await binance_client.get_ticker_price(symbol)
        except Exception:
            # Fallback: get from DB
            from sqlalchemy import select

            from app.models.market import OHLCV

            result = await db.execute(
                select(OHLCV)
                .where(OHLCV.symbol == symbol, OHLCV.interval == "15m")
                .order_by(OHLCV.open_time.desc())
                .limit(1)
            )
            candle = result.scalar_one_or_none()
            if not candle:
                raise OrderExecutionError(f"No price data for {symbol}") from None
            current_price = Decimal(str(candle.close))

        # Adapters and test doubles may return a float even though the Binance
        # wrapper returns Decimal. Normalize at the boundary so no money
        # calculation ever mixes binary floats with Decimal.
        current_price = Decimal(str(current_price))

        # Paper fills must be conservative: buys pay above and sells receive
        # below the observed price. Giving the simulator favorable fills would
        # inflate strategy results and make its paper performance misleading.
        slippage_bps = Decimal(str(random.uniform(0, PAPER_SLIPPAGE_BPS)))  # noqa: S311
        slippage_pct = slippage_bps / Decimal("10000")
        if side == "BUY":
            fill_price = current_price * (Decimal("1") + slippage_pct)
        else:
            fill_price = current_price * (Decimal("1") - slippage_pct)

        # Round to 2 decimal places (USDT precision)
        fill_price = fill_price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        commission = (fill_price * qty * PAPER_COMMISSION_RATE).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        order = Order(
            exchange_order_id=f"PAPER-{int(datetime.now(UTC).timestamp() * 1000)}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=float(qty),
            price=float(fill_price),
            filled_quantity=float(qty),
            avg_fill_price=float(fill_price),
            commission=float(commission),
            signal_id=signal_id,
            execution_mode=TradingExecutionMode.PAPER.value,
            notes="Paper trade (simulated with slippage)",
        )
        db.add(order)
        await db.flush()

        # Publish event
        await event_bus.publish(
            Event(
                type=EventType.ORDER_FILLED,
                data={
                    "order_id": order.id,
                    "exchange_order_id": order.exchange_order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": float(qty),
                    "avg_price": float(fill_price),
                    "commission": float(commission),
                    "paper_mode": True,
                },
            )
        )

        # Record execution analytics
        from app.services.exchange.execution_analytics import execution_analytics

        execution_analytics.record_execution(
            order_id=order.exchange_order_id,
            symbol=symbol,
            side=side,
            intended_price=current_price,
            fill_price=fill_price,
            quantity=qty,
            latency_ms=0,  # Paper = instant
        )

        logger.info(
            "paper_order_executed",
            order_id=order.id,
            symbol=symbol,
            side=side,
            filled_qty=float(qty),
            fill_price=float(fill_price),
            slippage_bps=float(slippage_bps),
            commission=float(commission),
        )
        return order

    async def _execute_live_order(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        quantity: float,
        signal_id: int | None = None,
        protective_exit: SpotLongProtectionPlan | None = None,
    ) -> Order:
        """Execute a Binance market order in testnet or armed live mode."""
        execution_mode = settings.execution_mode
        if execution_mode == TradingExecutionMode.PAPER:
            raise LiveTradingSafetyError("paper orders must use the paper execution path")

        try:
            estimated_price = await binance_client.get_ticker_price(symbol)
        except Exception as exc:
            raise OrderExecutionError(f"Could not get a pre-trade price for {symbol}") from exc

        try:
            rules = await binance_client.get_spot_symbol_rules(symbol)
            normalized_quantity = (
                rules.normalize_oco_quantity(Decimal(str(quantity)))
                if _requires_native_spot_protection(execution_mode, side)
                else rules.normalize_market_quantity(Decimal(str(quantity)))
            )
            estimated_notional = rules.validate_notional(estimated_price, normalized_quantity)
        except Exception as exc:
            raise OrderExecutionError(
                f"Could not validate the order against Binance Spot rules for {symbol}"
            ) from exc

        if execution_mode == TradingExecutionMode.LIVE:
            live_trading_guard.require_live_order(float(estimated_notional))
        if _requires_native_spot_protection(execution_mode, side):
            if protective_exit is None:
                raise LiveTradingSafetyError(
                    f"{execution_mode.value} Spot entry requires an exchange-native protection plan"
                )
        if execution_mode == TradingExecutionMode.LIVE:
            await self._reserve_live_notional(db, estimated_notional)

        client_order_id = f"TM-{uuid.uuid4().hex[:20]}"

        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            quantity=float(normalized_quantity),
            price=float(estimated_price),
            signal_id=signal_id,
            execution_mode=execution_mode.value,
            notes=f"mode={execution_mode.value};coid:{client_order_id}",
        )
        db.add(order)
        await db.flush()

        protection_handling_started = False
        try:
            result = await binance_client.place_market_order(
                symbol, side, normalized_quantity, client_order_id=client_order_id
            )

            order.exchange_order_id = str(result["orderId"])
            verified_order = await self._read_submitted_market_order(
                symbol=symbol,
                side=side,
                exchange_order_id=order.exchange_order_id,
            )
            order.status = OrderStatus(verified_order.get("status", "FILLED"))
            filled_quantity = Decimal(str(verified_order.get("executedQty", 0)))
            order.filled_quantity = float(filled_quantity)
            order.avg_fill_price = _get_average_fill_price(verified_order)

            commission = sum(float(fill.get("commission", 0)) for fill in result.get("fills", []))
            order.commission = commission

            if _requires_native_spot_protection(execution_mode, side):
                assert protective_exit is not None  # checked before the entry is submitted
                # A MARKET submission acknowledgement can be observed while
                # the matching engine still reports a partial/non-terminal
                # state. It is unsafe to infer that an OCO covers any later
                # fills, so the outer fail-closed path records a conservative
                # MISSING exposure instead of attaching protection prematurely.
                if order.status != OrderStatus.FILLED or filled_quantity <= 0:
                    raise OrderExecutionError(
                        "exchange entry signed read was not a fully filled terminal order"
                    )
                protection_handling_started = True
                await self._attach_native_spot_protection(
                    db=db,
                    order=order,
                    filled_quantity=filled_quantity,
                    fallback_price=estimated_price,
                    protection=protective_exit,
                    execution_mode=execution_mode,
                )

            await db.flush()

            await event_bus.publish(
                Event(
                    type=EventType.ORDER_FILLED,
                    data={
                        "order_id": order.id,
                        "exchange_order_id": order.exchange_order_id,
                        "symbol": symbol,
                        "side": side,
                        "quantity": order.filled_quantity,
                        "avg_price": order.avg_fill_price,
                        "commission": commission,
                        "paper_mode": False,
                        "execution_mode": execution_mode.value,
                    },
                )
            )

            # Record execution analytics
            from app.services.exchange.execution_analytics import execution_analytics

            if order.avg_fill_price:
                execution_analytics.record_execution(
                    order_id=order.exchange_order_id or "",
                    symbol=symbol,
                    side=side,
                    intended_price=Decimal(str(order.avg_fill_price or order.price or 0)),
                    fill_price=Decimal(str(order.avg_fill_price)),
                    quantity=Decimal(str(order.filled_quantity)),
                    latency_ms=0,
                )

            logger.info(
                "exchange_order_executed",
                order_id=order.id,
                symbol=symbol,
                side=side,
                execution_mode=execution_mode.value,
                filled_qty=order.filled_quantity,
                avg_price=order.avg_fill_price,
            )
            return order

        except Exception as e:
            if (
                execution_mode != TradingExecutionMode.PAPER
                and side == "BUY"
                and order.exchange_order_id
                and not protection_handling_started
                and protective_exit is not None
            ):
                # The exchange accepted the submission but its signed state
                # could not be trusted. Record the whole requested amount as
                # MISSING rather than assuming no asset was acquired.
                if execution_mode == TradingExecutionMode.LIVE:
                    live_trading_guard.disarm("exchange entry signed read was not confirmed")
                try:
                    await self._record_unprotected_live_position(
                        db=db,
                        order=order,
                        filled_quantity=normalized_quantity,
                        emergency_exit=EmergencySpotExitResult(
                            filled_quantity=Decimal("0"),
                            exchange_order_id=None,
                        ),
                        protection=protective_exit,
                        execution_mode=execution_mode,
                    )
                except Exception as ledger_error:
                    live_trading_guard.disarm("could not record an unverified exchange entry")
                    logger.critical(
                        "unverified_exchange_entry_ledger_record_failed",
                        order_id=order.id,
                        exchange_order_id=order.exchange_order_id,
                        error=str(ledger_error),
                        exc_info=True,
                    )
            if order.exchange_order_id:
                order.notes = f"{order.notes or ''};post_submission_error={str(e)[:180]}"[:500]
            else:
                order.status = OrderStatus.REJECTED
                order.notes = str(e)[:500]
            try:
                await db.flush()
                # Binance may have filled the entry before an OCO or its emergency
                # exit fails. Persist that evidence before propagating the error;
                # otherwise the caller's rollback would erase a real exposure.
                if execution_mode != TradingExecutionMode.PAPER and order.exchange_order_id:
                    await db.commit()
            except Exception as persistence_error:
                live_trading_guard.disarm("could not persist live post-submission failure")
                logger.critical(
                    "live_post_submission_failure_persistence_failed",
                    order_id=order.id,
                    exchange_order_id=order.exchange_order_id,
                    error=str(persistence_error),
                    exc_info=True,
                )
            logger.error("order_execution_failed", order_id=order.id, error=str(e), exc_info=True)
            raise OrderExecutionError(f"Order failed: {e}") from e

    @staticmethod
    async def _read_submitted_market_order(
        *,
        symbol: str,
        side: str,
        exchange_order_id: str,
    ) -> dict:
        """Treat the signed order read, not the submission response, as execution truth."""
        try:
            numeric_order_id = int(exchange_order_id)
        except ValueError as exc:
            raise OrderExecutionError("exchange order submission returned an invalid order id") from exc
        verified_order = await binance_client.get_order_status(symbol, numeric_order_id)
        if (
            str(verified_order.get("orderId", "")) != exchange_order_id
            or verified_order.get("symbol") != symbol
            or verified_order.get("side") != side
        ):
            raise OrderExecutionError("exchange order signed read did not match its submission")
        return verified_order

    async def _attach_native_spot_protection(
        self,
        *,
        db: AsyncSession,
        order: Order,
        filled_quantity: Decimal,
        fallback_price: Decimal,
        protection: SpotLongProtectionPlan,
        execution_mode: TradingExecutionMode = TradingExecutionMode.LIVE,
    ) -> None:
        """Attach an exchange-native OCO or fail closed after an exchange entry fill."""
        if filled_quantity <= 0:
            if execution_mode == TradingExecutionMode.LIVE:
                live_trading_guard.disarm("live entry returned no filled quantity")
            raise OrderExecutionError("exchange entry returned no filled quantity to protect")

        try:
            last_price = await binance_client.get_ticker_price(order.symbol)
        except Exception:
            last_price = fallback_price

        try:
            protection_result = await binance_client.place_spot_long_exit_oco(
                symbol=order.symbol,
                last_price=last_price,
                quantity=filled_quantity,
                take_profit_price=protection.take_profit_price,
                stop_loss_price=protection.stop_loss_price,
                client_order_id=f"TM-P-{uuid.uuid4().hex[:20]}",
            )
        except Exception as protection_error:
            if execution_mode == TradingExecutionMode.LIVE:
                live_trading_guard.disarm("native protection placement failed")
            emergency_exit = await self._attempt_emergency_spot_exit(
                db=db,
                order=order,
                filled_quantity=filled_quantity,
                cause=protection_error,
                execution_mode=execution_mode,
            )
            await self._record_unprotected_live_position(
                db=db,
                order=order,
                filled_quantity=filled_quantity,
                emergency_exit=emergency_exit,
                protection=protection,
                execution_mode=execution_mode,
            )
            raise OrderExecutionError(
                "exchange entry was filled but native protection failed; "
                "emergency exit was requested"
            ) from protection_error

        if protection_result.protected_quantity != filled_quantity:
            if execution_mode == TradingExecutionMode.LIVE:
                live_trading_guard.disarm("native protection does not cover the full entry")
            try:
                await binance_client.cancel_spot_order_list(
                    order.symbol, protection_result.order_list_id
                )
                cancellation_confirmed = await self._is_native_oco_cancelled(
                    symbol=order.symbol,
                    order_list_id=protection_result.order_list_id,
                )
                if not cancellation_confirmed:
                    raise OrderExecutionError(
                        "native OCO cancellation was not confirmed by signed reads"
                    )
            except Exception as cancellation_error:
                await self._record_unprotected_live_position(
                    db=db,
                    order=order,
                    filled_quantity=filled_quantity,
                    emergency_exit=EmergencySpotExitResult(
                        filled_quantity=Decimal("0"),
                        exchange_order_id=None,
                    ),
                    protection=protection,
                    execution_mode=execution_mode,
                )
                logger.critical(
                    "live_protection_partial_oco_cancellation_failed",
                    entry_order_id=order.id,
                    order_list_id=protection_result.order_list_id,
                    error=str(cancellation_error),
                    exc_info=True,
                )
                raise OrderExecutionError(
                    "native OCO covers less than the entry and could not be cancelled; "
                    "manual exchange intervention is required"
                ) from cancellation_error
            coverage_error = OrderExecutionError(
                "native OCO covered less than the filled entry; emergency exit was requested"
            )
            emergency_exit = await self._attempt_emergency_spot_exit(
                db=db,
                order=order,
                filled_quantity=filled_quantity,
                cause=coverage_error,
                execution_mode=execution_mode,
            )
            await self._record_unprotected_live_position(
                db=db,
                order=order,
                filled_quantity=filled_quantity,
                emergency_exit=emergency_exit,
                protection=protection,
                execution_mode=execution_mode,
            )
            raise coverage_error

        # Binance acknowledging the OCO creation is not proof that its two
        # child exits are live. Do not create an ACTIVE local position until a
        # signed read confirms that exact list and both untouched SELL legs.
        try:
            protection_active = await self._is_native_oco_active(
                symbol=order.symbol,
                order_list_id=protection_result.order_list_id,
            )
        except Exception as verification_error:
            protection_active = False
            logger.critical(
                "native_oco_activation_signed_read_failed",
                entry_order_id=order.id,
                order_list_id=protection_result.order_list_id,
                error=str(verification_error),
                exc_info=True,
            )

        if not protection_active:
            if execution_mode == TradingExecutionMode.LIVE:
                live_trading_guard.disarm("native OCO signed read was not confirmed")
            order.protective_order_list_id = protection_result.order_list_id
            order.protective_quantity = float(protection_result.protected_quantity)
            order.notes = (
                f"{order.notes or ''};protective_oco_unconfirmed="
                f"{order.protective_order_list_id}"
            )[:500]
            await self._record_unprotected_live_position(
                db=db,
                order=order,
                filled_quantity=filled_quantity,
                emergency_exit=EmergencySpotExitResult(
                    filled_quantity=Decimal("0"),
                    exchange_order_id=None,
                ),
                protection=protection,
                execution_mode=execution_mode,
                protective_order_list_id=protection_result.order_list_id,
                protective_quantity=protection_result.protected_quantity,
            )
            raise OrderExecutionError(
                "native OCO signed read did not confirm active full protection; "
                "manual exchange reconciliation is required"
            )

        order.protective_order_list_id = protection_result.order_list_id
        order.protective_quantity = float(protection_result.protected_quantity)
        order.notes = (f"{order.notes or ''};protective_oco={order.protective_order_list_id}")[:500]

    @staticmethod
    async def _is_native_oco_active(*, symbol: str, order_list_id: int) -> bool:
        """Confirm a newly created OCO has two untouched, signed SELL exits."""
        order_list = await binance_client.get_spot_order_list(order_list_id)
        if (
            int(order_list.get("orderListId", -1)) != order_list_id
            or order_list.get("symbol") != symbol
            or order_list.get("contingencyType") != "OCO"
            or order_list.get("listStatusType") != "EXEC_STARTED"
            or order_list.get("listOrderStatus") != "EXECUTING"
        ):
            return False
        try:
            order_ids = [int(child["orderId"]) for child in order_list.get("orders", [])]
        except (KeyError, TypeError, ValueError):
            return False
        if len(order_ids) != 2 or len(set(order_ids)) != 2:
            return False

        child_orders = [
            await binance_client.get_order_status(symbol, order_id) for order_id in order_ids
        ]
        try:
            return all(
                str(child_order.get("orderId", "")) == str(order_id)
                and child_order.get("symbol") == symbol
                and child_order.get("side") == "SELL"
                and int(child_order.get("orderListId", -1)) == order_list_id
                and child_order.get("status") in {"NEW", "PENDING_NEW"}
                and _decimal(child_order.get("executedQty")) == 0
                for order_id, child_order in zip(order_ids, child_orders, strict=True)
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    async def _is_native_oco_cancelled(*, symbol: str, order_list_id: int) -> bool:
        """Accept a naked emergency sell only after exact OCO cancellation is proven."""
        order_list = await binance_client.get_spot_order_list(order_list_id)
        if (
            int(order_list.get("orderListId", -1)) != order_list_id
            or order_list.get("symbol") != symbol
            or order_list.get("contingencyType") != "OCO"
            or order_list.get("listStatusType") != "ALL_DONE"
            or order_list.get("listOrderStatus") != "ALL_DONE"
        ):
            return False
        try:
            order_ids = [int(order["orderId"]) for order in order_list.get("orders", [])]
        except (KeyError, TypeError, ValueError):
            return False
        if len(order_ids) != 2:
            return False
        child_orders = [
            await binance_client.get_order_status(symbol, order_id) for order_id in order_ids
        ]
        return all(
            child_order.get("side") == "SELL"
            and int(child_order.get("orderListId", -1)) == order_list_id
            and child_order.get("status") in {"CANCELED", "EXPIRED"}
            and _decimal(child_order.get("executedQty")) == 0
            for child_order in child_orders
        )

    async def _attempt_emergency_spot_exit(
        self,
        *,
        db: AsyncSession,
        order: Order,
        filled_quantity: Decimal,
        cause: Exception,
        execution_mode: TradingExecutionMode = TradingExecutionMode.LIVE,
    ) -> EmergencySpotExitResult:
        """Best-effort emergency close when a post-fill OCO cannot be placed."""
        order.notes = f"{order.notes or ''};protection_failed={str(cause)[:180]}"[:500]
        try:
            submitted_exit = await binance_client.place_market_order(
                order.symbol,
                "SELL",
                filled_quantity,
                client_order_id=f"TM-E-{uuid.uuid4().hex[:20]}",
            )
            exchange_order_id = str(submitted_exit.get("orderId", ""))
            try:
                numeric_order_id = int(exchange_order_id)
            except ValueError as exc:
                raise OrderExecutionError(
                    "emergency Spot exit did not return a valid exchange order id"
                ) from exc
            verified_exit = await binance_client.get_order_status(order.symbol, numeric_order_id)
            if (
                str(verified_exit.get("orderId", "")) != exchange_order_id
                or verified_exit.get("symbol") != order.symbol
                or verified_exit.get("side") != "SELL"
            ):
                raise OrderExecutionError(
                    "emergency Spot exit signed read did not match its submission"
                )
            emergency_filled_quantity = _decimal(verified_exit.get("executedQty"))
            emergency_order = Order(
                exchange_order_id=exchange_order_id,
                symbol=order.symbol,
                side="SELL",
                order_type=OrderType.MARKET,
                status=OrderStatus(verified_exit.get("status", "FILLED")),
                quantity=float(filled_quantity),
                price=float(order.avg_fill_price or order.price or 0),
                filled_quantity=float(emergency_filled_quantity),
                avg_fill_price=_get_average_fill_price(verified_exit),
                commission=sum(
                    float(fill.get("commission", 0)) for fill in verified_exit.get("fills", [])
                ),
                execution_mode=execution_mode.value,
                notes=f"Emergency exit after protection failure for order #{order.id}",
            )
            db.add(emergency_order)
            await db.flush()
            logger.critical(
                "spot_protection_failed_emergency_exit_submitted",
                execution_mode=execution_mode.value,
                entry_order_id=order.id,
                emergency_exchange_order_id=emergency_order.exchange_order_id,
                filled_quantity=str(emergency_filled_quantity),
            )
            return EmergencySpotExitResult(
                filled_quantity=min(filled_quantity, emergency_filled_quantity),
                exchange_order_id=emergency_order.exchange_order_id,
            )
        except Exception as emergency_error:
            logger.critical(
                "live_protection_failed_emergency_exit_failed",
                entry_order_id=order.id,
                protection_error=str(cause),
                emergency_error=str(emergency_error),
                exc_info=True,
            )
            return EmergencySpotExitResult(filled_quantity=Decimal("0"), exchange_order_id=None)

    async def _record_unprotected_live_position(
        self,
        *,
        db: AsyncSession,
        order: Order,
        filled_quantity: Decimal,
        emergency_exit: EmergencySpotExitResult,
        protection: SpotLongProtectionPlan,
        execution_mode: TradingExecutionMode = TradingExecutionMode.LIVE,
        protective_order_list_id: int | None = None,
        protective_quantity: Decimal | None = None,
    ) -> None:
        """Persist any real base asset left after a failed protection sequence."""
        remaining_quantity = max(Decimal("0"), filled_quantity - emergency_exit.filled_quantity)
        if remaining_quantity <= 0:
            return
        position = Position(
            symbol=order.symbol,
            side="LONG",
            entry_price=float(order.avg_fill_price or order.price or 0),
            quantity=float(remaining_quantity),
            current_price=float(order.avg_fill_price or order.price or 0),
            unrealized_pnl=0,
            realized_pnl=0,
            stop_loss_price=float(protection.stop_loss_price),
            take_profit_price=float(protection.take_profit_price),
            execution_mode=execution_mode.value,
            entry_exchange_order_id=order.exchange_order_id,
            protective_order_list_id=protective_order_list_id,
            protective_quantity=(
                float(protective_quantity) if protective_quantity is not None else None
            ),
            protection_status="MISSING",
            protection_updated_at=datetime.now(UTC),
            is_open=True,
            opened_at=datetime.now(UTC),
        )
        db.add(position)
        await db.flush()
        logger.critical(
            "spot_unprotected_position_recorded",
            execution_mode=execution_mode.value,
            position_id=position.id,
            entry_order_id=order.id,
            entry_exchange_order_id=order.exchange_order_id,
            remaining_quantity=str(remaining_quantity),
            emergency_exit_order_id=emergency_exit.exchange_order_id,
        )

    async def cancel_order(self, db: AsyncSession, order: Order) -> Order:
        """Cancel an open order."""
        if not settings.paper_mode and order.exchange_order_id:
            await binance_client.cancel_order(order.symbol, int(order.exchange_order_id))

        order.status = OrderStatus.CANCELLED
        await db.flush()

        await event_bus.publish(
            Event(
                type=EventType.ORDER_CANCELLED,
                data={"order_id": order.id, "symbol": order.symbol},
            )
        )
        return order


def _requires_native_spot_protection(
    execution_mode: TradingExecutionMode,
    side: str,
) -> bool:
    """Both exchange environments require native exits for a Spot long entry."""
    return execution_mode != TradingExecutionMode.PAPER and side == "BUY"


def _get_average_fill_price(result: dict) -> float | None:
    """Calculate the actual average price from common Binance Spot responses."""
    direct_average = Decimal(str(result.get("avgPrice", 0)))
    if direct_average > 0:
        return float(direct_average)

    executed_quantity = Decimal(str(result.get("executedQty", 0)))
    cumulative_quote = Decimal(str(result.get("cummulativeQuoteQty", 0)))
    if executed_quantity > 0 and cumulative_quote > 0:
        return float(cumulative_quote / executed_quantity)

    fills = result.get("fills", [])
    fill_quantity = sum((Decimal(str(fill.get("qty", 0))) for fill in fills), Decimal("0"))
    fill_quote = sum(
        (Decimal(str(fill.get("price", 0))) * Decimal(str(fill.get("qty", 0))) for fill in fills),
        Decimal("0"),
    )
    return float(fill_quote / fill_quantity) if fill_quantity > 0 else None


def _decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value or 0))
    except Exception:
        return Decimal("0")


order_manager = OrderManager()

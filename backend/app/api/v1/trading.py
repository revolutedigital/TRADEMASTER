"""Trading API endpoints: orders, engine control, manual actions, paper trading."""

from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode, settings
from app.core.logging import get_logger
from app.dependencies import (
    get_db,
    get_order_repository,
    get_trading_engine,
    require_auth,
)
from app.models.portfolio import Position
from app.models.trade import Order
from app.repositories.order_repo import OrderRepository
from app.schemas.trading import OrderResponse
from app.services.exchange.binance_client import binance_client
from app.services.exchange.live_trading_guard import (
    LiveTradingSafetyError,
    live_trading_guard,
)
from app.services.exchange.spot_position_closer import (
    SpotPositionCloseError,
    spot_position_closer,
)
from app.services.exchange.spot_protection_reconciler import spot_protection_reconciler
from app.services.exchange.testnet_protection_verifier import (
    TestnetProtectionVerificationError,
    testnet_protection_verifier,
)

logger = get_logger(__name__)

router = APIRouter()


class PaperOrderRequest(BaseModel):
    symbol: str = "BTCUSDT"
    side: str = "BUY"  # BUY opens/adds a LONG; SELL only reduces a LONG
    quantity: float = Field(default=0.001, gt=0)  # Amount of the asset
    stop_loss_pct: float | None = Field(default=0.02, gt=0, lt=1)
    take_profit_pct: float | None = Field(default=0.04, gt=0, lt=1)
    price: float | None = Field(default=None, gt=0)  # Live price from frontend (Binance WS)


LIVE_ARM_CONFIRMATION_PHRASE = "ARM LIVE TRADING"


def _require_paper_execution_mode() -> None:
    """Keep legacy virtual order endpoints out of exchange-mode ledgers."""
    if settings.execution_mode != TradingExecutionMode.PAPER:
        raise HTTPException(
            status_code=409,
            detail="Paper trading endpoints are only available while execution mode is PAPER",
        )


class ArmLiveTradingRequest(BaseModel):
    confirmation_phrase: str = Field(min_length=1, max_length=100)
    arm_code: str = Field(min_length=20, max_length=200)
    totp_code: str = Field(pattern=r"^\d{6}$")

    @model_validator(mode="after")
    def validate_confirmation_phrase(self) -> "ArmLiveTradingRequest":
        if self.confirmation_phrase != LIVE_ARM_CONFIRMATION_PHRASE:
            raise ValueError(f"confirmation_phrase must be {LIVE_ARM_CONFIRMATION_PHRASE!r}")
        return self


class DisarmLiveTradingRequest(BaseModel):
    reason: str = Field(default="operator disarm", min_length=3, max_length=200)


class LiveProtectionReadinessResponse(BaseModel):
    ready: bool
    state: str
    checked_at: str | None
    max_age_seconds: int
    issues: list[str]


class LiveProtectionReconciliationResponse(LiveProtectionReadinessResponse):
    checked_positions: int
    active_protections: int
    closed_positions: int


class TestnetProtectionVerificationRequest(BaseModel):
    confirmation_phrase: str = Field(min_length=1, max_length=100)
    symbol: str = Field(pattern=r"^[A-Z0-9]{5,20}$")

    @model_validator(mode="after")
    def validate_confirmation_phrase(self) -> "TestnetProtectionVerificationRequest":
        if self.confirmation_phrase != "VERIFY TESTNET OCO":
            raise ValueError("confirmation_phrase must be 'VERIFY TESTNET OCO'")
        return self


class TestnetProtectionVerificationResponse(BaseModel):
    status: str
    environment: str
    symbol: str
    entry_order_id: str
    order_list_id: int
    exit_order_id: str
    verified_at: str


class ExchangePositionCloseRequest(BaseModel):
    confirmation_phrase: str = Field(min_length=1, max_length=100)
    totp_code: str = Field(pattern=r"^\d{6}$")

    @model_validator(mode="after")
    def validate_confirmation_phrase(self) -> "ExchangePositionCloseRequest":
        if self.confirmation_phrase != "CLOSE SPOT POSITION":
            raise ValueError("confirmation_phrase must be 'CLOSE SPOT POSITION'")
        return self


class ExchangePositionCloseResponse(BaseModel):
    status: str
    position_id: int
    symbol: str
    exit_order_id: str
    exit_price: float
    closed_at: str


class LiveTradingStatusResponse(BaseModel):
    execution_mode: str
    live_enabled: bool
    armed: bool
    armed_until: str | None
    armable: bool
    blockers: list[str]
    max_notional_per_order: float
    max_daily_notional: float
    reconciliation: LiveProtectionReadinessResponse
    testnet_verification: LiveProtectionReadinessResponse


@router.get("/live/status", response_model=LiveTradingStatusResponse)
async def get_live_trading_status(
    _user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Read the effective execution mode and all live-execution safety gates."""
    return live_trading_guard.status()


@router.post("/live/arm", response_model=LiveTradingStatusResponse)
async def arm_live_trading(
    body: ArmLiveTradingRequest,
    request: Request,
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Arm real-capital execution temporarily; this endpoint never places an order."""
    try:
        status = live_trading_guard.arm(
            actor=str(user.get("sub", "admin")),
            arm_code=body.arm_code,
            totp_code=body.totp_code,
        )
    except LiveTradingSafetyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="LIVE_TRADING_ARMED",
        user_id=str(user.get("sub", "admin")),
        resource="trading:live-control",
        details={"armed_until": status["armed_until"]},
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return status


@router.post("/live/reconcile", response_model=LiveProtectionReconciliationResponse)
async def reconcile_live_spot_protection(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Read and reconcile Binance Spot OCO protection; this endpoint never trades."""
    if settings.execution_mode.value != "LIVE":
        raise HTTPException(
            status_code=409,
            detail="Spot OCO reconciliation is only available when execution mode is LIVE",
        )
    try:
        report = await spot_protection_reconciler.reconcile(db)
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail="Live Spot protection reconciliation failed; execution remains disarmed",
        ) from exc

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="LIVE_SPOT_PROTECTION_RECONCILED",
        user_id=str(user.get("sub", "admin")),
        resource="trading:live-control",
        details={
            "checked_positions": report.checked_positions,
            "active_protections": report.active_protections,
            "closed_positions": report.closed_positions,
            "issues": list(report.issues),
        },
    )
    return report.as_dict(settings.live_trading_reconciliation_max_age_seconds)


@router.post(
    "/live/testnet-protection-verification",
    response_model=TestnetProtectionVerificationResponse,
)
async def verify_testnet_native_spot_protection(
    body: TestnetProtectionVerificationRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_auth),
    engine=Depends(get_trading_engine),
) -> dict[str, str | int]:
    """Run the explicit Testnet BUY -> OCO -> signed read -> cleanup release proof."""
    if engine._running:
        raise HTTPException(
            status_code=409,
            detail="stop the trading engine before running the isolated Testnet protection proof",
        )
    try:
        report = await testnet_protection_verifier.verify(db, body.symbol)
    except TestnetProtectionVerificationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="TESTNET_NATIVE_OCO_VERIFIED",
        user_id=str(user.get("sub", "admin")),
        resource="trading:live-control",
        details={"symbol": report.symbol, "order_list_id": report.order_list_id},
    )
    return report.as_dict()


@router.post(
    "/positions/{position_id}/close-exchange",
    response_model=ExchangePositionCloseResponse,
)
async def close_live_spot_position_on_exchange(
    position_id: int,
    body: ExchangePositionCloseRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(require_auth),
) -> dict[str, str | int | float]:
    """Cancel the native OCO, prove its terminal state, then sell once on Binance Spot."""
    try:
        report = await spot_position_closer.close(
            db=db,
            position_id=position_id,
            totp_code=body.totp_code,
        )
    except (LiveTradingSafetyError, SpotPositionCloseError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="LIVE_SPOT_POSITION_CLOSED_ON_EXCHANGE",
        user_id=str(user.get("sub", "admin")),
        resource=f"trading:position:{report.position_id}",
        details={
            "symbol": report.symbol,
            "status": report.status,
            "exit_order_id": report.exit_order_id,
            "exit_price": float(report.exit_price),
        },
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return report.as_dict()


@router.post("/live/disarm", response_model=LiveTradingStatusResponse)
async def disarm_live_trading(
    body: DisarmLiveTradingRequest,
    request: Request,
    user: dict = Depends(require_auth),
) -> dict[str, object]:
    """Immediately revoke live-execution permission without touching open positions."""
    status = live_trading_guard.disarm(body.reason)

    from app.core.audit import audit_logger

    await audit_logger.log_event(
        action="LIVE_TRADING_DISARMED",
        user_id=str(user.get("sub", "admin")),
        resource="trading:live-control",
        details={"reason": body.reason},
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return status


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(
    symbol: str | None = None,
    side: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
    offset: int = 0,
    execution_mode: TradingExecutionMode | None = None,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
    repo: OrderRepository = Depends(get_order_repository),
):
    """Get recent orders from the selected ledger, defaulting to the active mode."""
    selected_mode = execution_mode or settings.execution_mode
    orders = await repo.list_filtered(
        db,
        symbol=symbol,
        side=side,
        status=status,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        execution_mode=selected_mode.value,
    )
    return orders


@router.post("/paper-order")
async def create_paper_order(
    req: PaperOrderRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Execute a simulated paper trade using live Binance prices. Requires authentication."""
    _require_paper_execution_mode()
    symbol = req.symbol.upper()
    side = req.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(400, "side must be BUY or SELL")

    # Use frontend live price (from Binance WebSocket) or fetch from Binance API
    if req.price and req.price > 0:
        price = req.price
    else:
        try:
            price = float(await binance_client.get_ticker_price(symbol))
        except Exception as exc:
            logger.error("live_price_fetch_failed", symbol=symbol, error=str(exc))
            raise HTTPException(
                503, f"Não foi possível obter preço em tempo real para {symbol}. Tente novamente."
            ) from exc
    now = datetime.now(timezone.utc)
    commission = price * req.quantity * 0.001  # 0.1% fee

    # Create order record (instantly filled)
    order = Order(
        exchange_order_id=f"PAPER-{int(now.timestamp() * 1000)}",
        symbol=symbol,
        side=side,
        order_type="MARKET",
        status="FILLED",
        quantity=req.quantity,
        price=price,
        filled_quantity=req.quantity,
        avg_fill_price=price,
        commission=commission,
        execution_mode=TradingExecutionMode.PAPER.value,
        notes="Paper trade (simulated)",
    )
    db.add(order)

    # Paper execution must mirror the supported exchange contract: Spot
    # long-only. A SELL can reduce an existing simulated long but it cannot
    # create a synthetic short that would never be executable on Binance Spot.
    if side == "SELL":
        long_result = await db.execute(
            select(Position)
            .where(
                Position.symbol == symbol,
                Position.side == "LONG",
                Position.is_open.is_(True),
                Position.execution_mode == TradingExecutionMode.PAPER.value,
            )
            .with_for_update()
        )
        long_position = long_result.scalar_one_or_none()
        if long_position is None:
            raise HTTPException(
                status_code=409,
                detail="Paper Spot SELL can only reduce an existing LONG position",
            )

        current_quantity = Decimal(str(long_position.quantity))
        requested_quantity = Decimal(str(req.quantity))
        if requested_quantity > current_quantity:
            raise HTTPException(
                status_code=409,
                detail="Paper Spot SELL quantity exceeds the open LONG position",
            )

        realized_pnl = (
            Decimal(str(price)) - Decimal(str(long_position.entry_price))
        ) * requested_quantity - Decimal(str(commission))
        remaining_quantity = current_quantity - requested_quantity
        long_position.current_price = price
        long_position.realized_pnl = float(
            Decimal(str(long_position.realized_pnl)) + realized_pnl
        )

        if remaining_quantity == 0:
            long_position.unrealized_pnl = 0
            long_position.is_open = False
            long_position.closed_at = now
            await db.commit()
            await db.refresh(order)
            return {
                "status": "position_closed",
                "order_id": order.id,
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": req.quantity,
                "closed_position_id": long_position.id,
                "realized_pnl": round(float(long_position.realized_pnl), 2),
            }

        long_position.quantity = float(remaining_quantity)
        long_position.unrealized_pnl = float(
            (Decimal(str(price)) - Decimal(str(long_position.entry_price)))
            * remaining_quantity
        )
        await db.commit()
        await db.refresh(order)
        return {
            "status": "position_reduced",
            "order_id": order.id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": req.quantity,
            "position_id": long_position.id,
            "remaining_quantity": float(remaining_quantity),
            "realized_pnl": round(float(realized_pnl), 2),
        }

    # BUY can close any historical paper short created before the Spot-only
    # rule, but cannot create a new one.
    position_side = "LONG"

    # Check for a historical short that predates the Spot-long-only contract.
    opposite_side = "SHORT"
    existing = await db.execute(
        select(Position).where(
            Position.symbol == symbol,
            Position.side == opposite_side,
            Position.is_open.is_(True),
            Position.execution_mode == TradingExecutionMode.PAPER.value,
        )
    )
    existing_pos = existing.scalar_one_or_none()

    if existing_pos:
        # Close opposite position
        if existing_pos.side == "LONG":
            pnl = (price - float(existing_pos.entry_price)) * float(existing_pos.quantity)
        else:
            pnl = (float(existing_pos.entry_price) - price) * float(existing_pos.quantity)
        pnl -= commission

        existing_pos.current_price = price
        existing_pos.realized_pnl = pnl
        existing_pos.unrealized_pnl = 0
        existing_pos.is_open = False
        existing_pos.closed_at = now
        logger.info(
            "paper_position_closed", symbol=symbol, side=existing_pos.side, pnl=round(pnl, 2)
        )

        await db.commit()
        await db.refresh(order)

        return {
            "status": "position_closed",
            "order_id": order.id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": req.quantity,
            "closed_position_id": existing_pos.id,
            "realized_pnl": round(pnl, 2),
        }

    # Check for existing open position on same side (add to it)
    same_result = await db.execute(
        select(Position).where(
            Position.symbol == symbol,
            Position.side == position_side,
            Position.is_open.is_(True),
            Position.execution_mode == TradingExecutionMode.PAPER.value,
        )
    )
    same_pos = same_result.scalar_one_or_none()

    if same_pos:
        # Average into position
        old_qty = float(same_pos.quantity)
        old_price = float(same_pos.entry_price)
        new_qty = old_qty + req.quantity
        avg_price = (old_price * old_qty + price * req.quantity) / new_qty
        same_pos.entry_price = avg_price
        same_pos.quantity = new_qty
        same_pos.current_price = price
        if position_side == "LONG":
            same_pos.unrealized_pnl = (price - avg_price) * new_qty
        else:
            same_pos.unrealized_pnl = (avg_price - price) * new_qty
        # Update stops
        if req.stop_loss_pct:
            sl = (
                price * (1 - req.stop_loss_pct)
                if position_side == "LONG"
                else price * (1 + req.stop_loss_pct)
            )
            same_pos.stop_loss_price = sl
        if req.take_profit_pct:
            tp = (
                price * (1 + req.take_profit_pct)
                if position_side == "LONG"
                else price * (1 - req.take_profit_pct)
            )
            same_pos.take_profit_price = tp

        await db.commit()
        await db.refresh(order)

        return {
            "status": "position_increased",
            "order_id": order.id,
            "symbol": symbol,
            "side": position_side,
            "price": price,
            "quantity": req.quantity,
            "position_id": same_pos.id,
            "total_quantity": new_qty,
            "avg_entry": round(avg_price, 2),
        }

    # New position
    stop_loss = None
    take_profit = None
    if req.stop_loss_pct:
        stop_loss = (
            price * (1 - req.stop_loss_pct)
            if position_side == "LONG"
            else price * (1 + req.stop_loss_pct)
        )
    if req.take_profit_pct:
        take_profit = (
            price * (1 + req.take_profit_pct)
            if position_side == "LONG"
            else price * (1 - req.take_profit_pct)
        )

    position = Position(
        symbol=symbol,
        side=position_side,
        entry_price=price,
        quantity=req.quantity,
        current_price=price,
        unrealized_pnl=0,
        realized_pnl=0,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        execution_mode=TradingExecutionMode.PAPER.value,
        is_open=True,
        opened_at=now,
    )
    db.add(position)
    await db.commit()
    await db.refresh(order)
    await db.refresh(position)

    logger.info(
        "paper_position_opened",
        symbol=symbol,
        side=position_side,
        price=price,
        qty=req.quantity,
        sl=round(stop_loss, 2) if stop_loss else None,
        tp=round(take_profit, 2) if take_profit else None,
    )

    return {
        "status": "position_opened",
        "order_id": order.id,
        "position_id": position.id,
        "symbol": symbol,
        "side": position_side,
        "entry_price": price,
        "quantity": req.quantity,
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "take_profit": round(take_profit, 2) if take_profit else None,
    }


class ClosePositionRequest(BaseModel):
    price: float | None = None  # Live price from frontend (Binance WS)


@router.post("/close-position/{position_id}")
async def close_position_manually(
    position_id: int,
    req: ClosePositionRequest | None = None,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Manually close a position at current live price. Requires authentication."""
    _require_paper_execution_mode()
    result = await db.execute(
        select(Position).where(
            Position.id == position_id,
            Position.is_open.is_(True),
            Position.execution_mode == TradingExecutionMode.PAPER.value,
        )
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=404, detail="Open position not found")
    # Use frontend-provided live price first, then try Binance API
    exit_price = None
    if req and req.price and req.price > 0:
        exit_price = req.price
    else:
        try:
            exit_price = float(await binance_client.get_ticker_price(position.symbol))
        except Exception as exc:
            logger.error("live_price_fetch_failed", symbol=position.symbol, error=str(exc))
            raise HTTPException(
                503, "Não foi possível obter preço em tempo real. Tente novamente."
            ) from exc

    # Calculate P&L
    if position.side == "LONG":
        pnl = (exit_price - float(position.entry_price)) * float(position.quantity)
    else:
        pnl = (float(position.entry_price) - exit_price) * float(position.quantity)

    commission = exit_price * float(position.quantity) * 0.001
    pnl -= commission

    position.current_price = exit_price
    position.realized_pnl = pnl
    position.unrealized_pnl = 0
    position.is_open = False
    position.closed_at = datetime.now(timezone.utc)

    # Create closing order record
    close_side = "SELL" if position.side == "LONG" else "BUY"
    order = Order(
        exchange_order_id=f"PAPER-CLOSE-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        symbol=position.symbol,
        side=close_side,
        order_type="MARKET",
        status="FILLED",
        quantity=float(position.quantity),
        price=exit_price,
        filled_quantity=float(position.quantity),
        avg_fill_price=exit_price,
        commission=commission,
        execution_mode=TradingExecutionMode.PAPER.value,
        notes=f"Paper close position #{position_id}",
    )
    db.add(order)
    await db.commit()

    logger.info(
        "paper_position_closed", position_id=position_id, exit_price=exit_price, pnl=round(pnl, 2)
    )

    return {
        "status": "closed",
        "position_id": position_id,
        "exit_price": exit_price,
        "pnl": round(pnl, 2),
    }


@router.post("/engine/start")
async def start_engine(
    _user: dict = Depends(require_auth),
    engine=Depends(get_trading_engine),
):
    """Start the engine from one approved market-data source. Requires authentication."""
    if settings.execution_mode != TradingExecutionMode.PAPER:
        # Testnet/LIVE decisions must originate from Binance market events.
        # Never start the offline price/candle generators in an exchange mode.
        from app.services.exchange.binance_ws import binance_ws_manager

        if not binance_ws_manager._running or not binance_ws_manager._tasks:
            raise HTTPException(
                status_code=409,
                detail=(
                    "exchange execution requires active Binance WebSocket market data; "
                    "synthetic candles are restricted to PAPER mode"
                ),
            )

    if not engine.reserve_start():
        return {"status": "already_running"}

    import asyncio

    if settings.execution_mode == TradingExecutionMode.PAPER:
        # Paper mode may use the independent public price feed and derive its
        # candles locally. These sources are explicitly forbidden above for
        # Testnet/LIVE execution.
        from app.services.market.price_fetcher import price_fetcher

        if not price_fetcher._running:
            asyncio.create_task(price_fetcher.start(), name="price_fetcher")

        from app.services.market.synthetic_kline_generator import synthetic_kline_generator

        if not synthetic_kline_generator._running:
            asyncio.create_task(synthetic_kline_generator.start(), name="synthetic_kline_generator")

        from app.services.market.stream_processor import market_stream_processor

        if not market_stream_processor._running:
            asyncio.create_task(market_stream_processor.start(), name="market_stream_processor")

    asyncio.create_task(engine.start(), name="trading_engine")
    return {"status": "started"}


@router.post("/engine/stop")
async def stop_engine(
    _user: dict = Depends(require_auth),
    engine=Depends(get_trading_engine),
):
    """Pause the trading engine. Requires authentication."""
    live_trading_guard.disarm("trading engine stopped by operator")
    if not engine._running:
        return {"status": "already_stopped", "live_trading": live_trading_guard.status()}

    await engine.stop()
    return {"status": "stopped", "live_trading": live_trading_guard.status()}


@router.get("/engine/status")
async def engine_status(_user: dict = Depends(require_auth)):
    """Get trading engine status. Requires authentication."""
    from sqlalchemy import func

    from app.dependencies import get_circuit_breaker, get_trading_engine
    from app.services.exchange.binance_ws import binance_ws_manager
    from app.services.market.price_fetcher import price_fetcher
    from app.services.market.synthetic_kline_generator import synthetic_kline_generator
    from app.services.scheduler import scheduler

    engine = get_trading_engine()
    cb = get_circuit_breaker()

    # Count candles per symbol for monitoring
    candle_counts = {}
    try:
        from app.models.base import async_session_factory as db_session_maker
        from app.models.market import OHLCV

        async with db_session_maker() as sdb:
            for sym in ["BTCUSDT", "ETHUSDT"]:
                result = await sdb.execute(
                    select(func.count())
                    .select_from(OHLCV)
                    .where(
                        OHLCV.symbol == sym,
                        OHLCV.interval == synthetic_kline_generator._interval_label,
                    )
                )
                candle_counts[sym] = result.scalar() or 0
    except Exception:
        pass

    return {
        "engine_running": engine._running,
        "price_fetcher_active": price_fetcher._running,
        "price_fetcher_source": price_fetcher._source,
        "candle_interval": synthetic_kline_generator._interval_label,
        "candle_counts": candle_counts,
        "min_candles_for_signal": 30,
        "websocket_streams": len(binance_ws_manager._tasks),
        "websocket_active": binance_ws_manager._running,
        "synthetic_kline_active": synthetic_kline_generator._running,
        "synthetic_candles_tracking": list(synthetic_kline_generator._candles.keys()),
        "daily_trade_count": dict(engine._daily_trade_count),
        "max_trades_per_day": 6,
        "circuit_breaker": cb.get_status(),
        "scheduled_tasks": scheduler.get_status(),
        "live_trading": live_trading_guard.status(),
    }


@router.post("/engine/train-model")
async def train_bootstrap_model(
    _user: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Train a bootstrap XGBoost model using available historical data."""
    from pathlib import Path

    import numpy as np

    from app.services.market.data_collector import market_data_collector
    from app.services.ml.features import feature_engineer
    from app.services.ml.models.xgboost_model import XGBoostTradingModel
    from app.services.ml.pipeline import ml_pipeline
    from app.services.ml.preprocessor import Preprocessor

    MODELS_DIR = Path("ml_artifacts/models")
    SCALERS_DIR = Path("ml_artifacts/scalers")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    preprocessor = Preprocessor(threshold=0.005)

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        symbol_lower = symbol.lower()

        # Get historical candles
        df = await market_data_collector.get_latest_candles(
            db=db, symbol=symbol, interval="1m", limit=10000
        )

        if df.empty or len(df) < 300:
            # Try 1h interval
            df = await market_data_collector.get_latest_candles(
                db=db, symbol=symbol, interval="1h", limit=10000
            )

        if df.empty or len(df) < 300:
            results[symbol] = {
                "status": "skipped",
                "reason": f"Only {len(df)} candles available (need 300+)",
            }
            continue

        # Feature engineering
        df_features = feature_engineer.build_features(df)
        if df_features.empty:
            results[symbol] = {"status": "skipped", "reason": "Feature engineering failed"}
            continue

        feature_cols = feature_engineer.get_feature_columns(df_features)

        # Create targets and split
        try:
            df_features = preprocessor.create_target(df_features, horizon=5)
            split = preprocessor.prepare_tabular(df_features, feature_cols)
        except Exception as e:
            results[symbol] = {"status": "error", "reason": str(e)}
            continue

        # Train XGBoost
        model = XGBoostTradingModel()
        training_result = model.train(
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            feature_names=split.feature_names,
        )

        # Save model and scaler
        model.save(MODELS_DIR / f"xgboost_{symbol_lower}.json")
        Preprocessor.save_scaler(split.scaler, SCALERS_DIR / f"scaler_{symbol_lower}.joblib")

        results[symbol] = {
            "status": "trained",
            "rows": len(df),
            "features": len(feature_cols),
            "train_accuracy": round(training_result.accuracy, 4),
            "val_accuracy": round(training_result.val_accuracy, 4),
        }

    # Reload models in the pipeline
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        await ml_pipeline.load_models(symbol)

    return {"results": results, "models_reloaded": True}


@router.get("/engine/drift-status")
async def get_drift_status(_user: dict = Depends(require_auth)):
    """Get model drift detection status for all symbols."""
    from app.services.ml.drift_detector import drift_detector

    return drift_detector.get_status()


@router.post("/engine/retrain-if-drifted")
async def trigger_drift_retrain(_user: dict = Depends(require_auth)):
    """Manually trigger drift check and retrain if needed."""
    from app.services.ml.drift_detector import drift_detector

    retrained = await drift_detector.auto_retrain_if_needed()
    return {"retrained": retrained}


@router.get("/engine/execution-analytics")
async def get_execution_analytics(
    symbol: str | None = None,
    _user: dict = Depends(require_auth),
):
    """Get trade execution quality metrics (slippage, latency, fill rate)."""
    from app.services.exchange.execution_analytics import execution_analytics

    return execution_analytics.get_best_execution_report()


@router.get("/engine/regime-status")
async def get_regime_status(_user: dict = Depends(require_auth)):
    """Get current market regime detection for all tracked symbols."""
    from app.services.ml.regime import regime_detector

    return {
        "regimes": regime_detector.get_all(),
        "description": "Adaptive regime: bull/bear/sideways × low/normal/high volatility",
    }


@router.get("/engine/rolling-sharpe")
async def get_rolling_sharpe(_user: dict = Depends(require_auth)):
    """Get rolling Sharpe ratio monitor status (auto-pause indicator)."""
    from app.services.risk.rolling_sharpe import rolling_sharpe_monitor

    return rolling_sharpe_monitor.get_status()


@router.post("/engine/rolling-sharpe/resume")
async def force_resume_sharpe(_user: dict = Depends(require_auth)):
    """Manually resume trading after rolling Sharpe auto-pause."""
    from app.services.risk.rolling_sharpe import rolling_sharpe_monitor

    rolling_sharpe_monitor.force_resume()
    return {"status": "resumed"}


@router.get("/engine/ensemble-status")
async def get_ensemble_status(_user: dict = Depends(require_auth)):
    """Get ensemble voting configuration and regime-adaptive weight info."""
    from app.services.ml.ensemble_voter import _REGIME_BIAS, _REGIME_WEIGHTS, _VOL_ADJUSTMENTS

    return {
        "regime_weights": {
            k: {"technical": v[0], "ml": v[1], "regime": v[2]} for k, v in _REGIME_WEIGHTS.items()
        },
        "volatility_adjustments": {
            k: {"tech": v[0], "ml": v[1], "regime": v[2]} for k, v in _VOL_ADJUSTMENTS.items()
        },
        "regime_bias": _REGIME_BIAS,
    }

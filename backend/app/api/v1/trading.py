"""Trading API endpoints: orders, engine control, manual actions, paper trading."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.dependencies import get_db, require_auth
from app.models.trade import Order
from app.models.portfolio import Position
from app.models.market import OHLCV
from app.schemas.trading import OrderResponse

logger = get_logger(__name__)

router = APIRouter()


class PaperOrderRequest(BaseModel):
    symbol: str = "BTCUSDT"
    side: str = "BUY"  # BUY or SELL
    quantity: float = 0.001  # Amount of the asset
    stop_loss_pct: float | None = 0.02  # 2% stop loss
    take_profit_pct: float | None = 0.04  # 4% take profit


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(
    symbol: str | None = None,
    status: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Get recent orders. Requires authentication."""
    query = select(Order).order_by(Order.created_at.desc()).limit(limit)
    if symbol:
        query = query.where(Order.symbol == symbol.upper())
    if status:
        query = query.where(Order.status == status.upper())
    result = await db.execute(query)
    return list(result.scalars().all())


@router.post("/paper-order")
async def create_paper_order(
    req: PaperOrderRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Execute a simulated paper trade using DB prices. Requires authentication."""
    symbol = req.symbol.upper()
    side = req.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(400, "side must be BUY or SELL")

    # Get current price from latest candle in DB
    result = await db.execute(
        select(OHLCV)
        .where(OHLCV.symbol == symbol, OHLCV.interval == "1h")
        .order_by(OHLCV.open_time.desc())
        .limit(1)
    )
    candle = result.scalar_one_or_none()
    if not candle:
        raise HTTPException(404, f"No price data for {symbol}")

    price = float(candle.close)
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
        notes="Paper trade (simulated)",
    )
    db.add(order)

    # Handle position logic
    position_side = "LONG" if side == "BUY" else "SHORT"

    # Check for existing open position on opposite side (close it)
    opposite_side = "SHORT" if side == "BUY" else "LONG"
    existing = await db.execute(
        select(Position).where(
            Position.symbol == symbol,
            Position.side == opposite_side,
            Position.is_open == True,
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
        logger.info("paper_position_closed", symbol=symbol, side=existing_pos.side, pnl=round(pnl, 2))

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
            Position.is_open == True,
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
            sl = price * (1 - req.stop_loss_pct) if position_side == "LONG" else price * (1 + req.stop_loss_pct)
            same_pos.stop_loss_price = sl
        if req.take_profit_pct:
            tp = price * (1 + req.take_profit_pct) if position_side == "LONG" else price * (1 - req.take_profit_pct)
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
        stop_loss = price * (1 - req.stop_loss_pct) if position_side == "LONG" else price * (1 + req.stop_loss_pct)
    if req.take_profit_pct:
        take_profit = price * (1 + req.take_profit_pct) if position_side == "LONG" else price * (1 - req.take_profit_pct)

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


@router.post("/close-position/{position_id}")
async def close_position_manually(
    position_id: int,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Manually close a position at current DB price. Requires authentication."""
    result = await db.execute(
        select(Position).where(Position.id == position_id, Position.is_open == True)
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=404, detail="Open position not found")

    # Get current price from DB
    candle_result = await db.execute(
        select(OHLCV)
        .where(OHLCV.symbol == position.symbol, OHLCV.interval == "1h")
        .order_by(OHLCV.open_time.desc())
        .limit(1)
    )
    candle = candle_result.scalar_one_or_none()
    exit_price = float(candle.close) if candle else float(position.current_price)

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
        notes=f"Paper close position #{position_id}",
    )
    db.add(order)
    await db.commit()

    logger.info("paper_position_closed", position_id=position_id, exit_price=exit_price, pnl=round(pnl, 2))

    return {
        "status": "closed",
        "position_id": position_id,
        "exit_price": exit_price,
        "pnl": round(pnl, 2),
    }


@router.post("/engine/start")
async def start_engine(_user: dict = Depends(require_auth)):
    """Start the trading engine. Requires authentication."""
    from app.services.trading_engine import trading_engine
    if trading_engine._running:
        return {"status": "already_running"}

    import asyncio
    asyncio.create_task(trading_engine.start(), name="trading_engine")
    return {"status": "started"}


@router.post("/engine/stop")
async def stop_engine(_user: dict = Depends(require_auth)):
    """Pause the trading engine. Requires authentication."""
    from app.services.trading_engine import trading_engine
    if not trading_engine._running:
        return {"status": "already_stopped"}

    await trading_engine.stop()
    return {"status": "stopped"}


@router.get("/engine/status")
async def engine_status(_user: dict = Depends(require_auth)):
    """Get trading engine status. Requires authentication."""
    from app.services.trading_engine import trading_engine
    from app.services.scheduler import scheduler
    from app.services.exchange.binance_ws import binance_ws_manager
    from app.services.risk.drawdown import circuit_breaker

    return {
        "engine_running": trading_engine._running,
        "websocket_streams": len(binance_ws_manager._tasks),
        "websocket_active": binance_ws_manager._running,
        "circuit_breaker": circuit_breaker.get_status(),
        "scheduled_tasks": scheduler.get_status(),
    }

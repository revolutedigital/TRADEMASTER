"""Export API endpoints for downloading data as CSV."""

import csv
import io
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.models.trade import Order
from app.models.portfolio import Position

router = APIRouter()


@router.get("/trades")
async def export_trades(
    format: str = "csv",
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Export trade/order history as CSV."""
    result = await db.execute(
        select(Order).order_by(Order.created_at.desc()).limit(5000)
    )
    orders = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "symbol", "side", "type", "status", "quantity", "price", "filled_qty", "avg_fill_price", "commission", "created_at"])
    for o in orders:
        writer.writerow([
            o.id, o.symbol, o.side, o.order_type, o.status,
            float(o.quantity), float(o.price) if o.price else "",
            float(o.filled_quantity), float(o.avg_fill_price) if o.avg_fill_price else "",
            float(o.commission), o.created_at.isoformat() if o.created_at else "",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{datetime.now().strftime('%Y%m%d')}.csv"},
    )


@router.get("/portfolio")
async def export_portfolio(
    format: str = "csv",
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Export portfolio positions as CSV."""
    result = await db.execute(select(Position).order_by(Position.created_at.desc()).limit(5000))
    positions = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "symbol", "side", "entry_price", "quantity", "current_price", "unrealized_pnl", "realized_pnl", "is_open", "opened_at", "closed_at"])
    for p in positions:
        writer.writerow([
            p.id, p.symbol, p.side, float(p.entry_price), float(p.quantity),
            float(p.current_price), float(p.unrealized_pnl), float(p.realized_pnl),
            p.is_open, p.opened_at.isoformat() if p.opened_at else "",
            p.closed_at.isoformat() if p.closed_at else "",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=portfolio_{datetime.now().strftime('%Y%m%d')}.csv"},
    )

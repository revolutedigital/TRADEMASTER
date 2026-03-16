"""Dashboard aggregation endpoint.

Returns everything the frontend needs in a single API call:
equity, positions, P&L, engine status, alerts, risk, data quality.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db, require_auth
from app.core.logging import get_logger
from app.repositories.position_repo import PositionRepository

logger = get_logger(__name__)

router = APIRouter()


@router.get("/dashboard")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Aggregated dashboard payload -- replaces 5+ separate frontend calls."""

    repo = PositionRepository()

    # ------------------------------------------------------------------
    # 1. Equity & balance
    # ------------------------------------------------------------------
    available_balance = 0.0
    total_equity = 0.0
    try:
        from app.services.exchange.binance_client import binance_client
        balance = await binance_client.get_balance("USDT")
        available_balance = float(balance)
    except Exception:
        available_balance = 0.0

    # ------------------------------------------------------------------
    # 2. Open positions
    # ------------------------------------------------------------------
    open_positions = await repo.get_open(db)
    total_exposure = sum(
        float(p.current_price) * float(p.quantity) for p in open_positions
    )
    total_unrealized = sum(float(p.unrealized_pnl) for p in open_positions)
    total_equity = available_balance + total_exposure

    positions_list = [
        {
            "id": p.id,
            "symbol": p.symbol,
            "side": p.side,
            "quantity": float(p.quantity),
            "entry_price": float(p.entry_price),
            "current_price": float(p.current_price),
            "unrealized_pnl": float(p.unrealized_pnl),
        }
        for p in open_positions
    ]

    # ------------------------------------------------------------------
    # 3. Today's P&L  (closed positions from today + unrealized)
    # ------------------------------------------------------------------
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    closed_today = await repo.get_closed(db, limit=500)
    realized_today = sum(
        float(p.realized_pnl)
        for p in closed_today
        if p.closed_at and p.closed_at >= today_start
    )
    todays_pnl = realized_today + total_unrealized

    # ------------------------------------------------------------------
    # 4. Engine status
    # ------------------------------------------------------------------
    engine_running = False
    try:
        from app.services.trading_engine import trading_engine
        engine_running = trading_engine._running
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. Recent triggered alerts (last 5)
    # ------------------------------------------------------------------
    recent_alerts: list[dict] = []
    try:
        from app.models.alert import PriceAlert
        result = await db.execute(
            select(PriceAlert)
            .where(PriceAlert.is_triggered == True)
            .order_by(PriceAlert.triggered_at.desc())
            .limit(5)
        )
        for a in result.scalars().all():
            recent_alerts.append({
                "id": a.id,
                "symbol": a.symbol,
                "condition": a.condition,
                "target_price": a.target_price,
                "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
                "notes": a.notes,
            })
    except Exception as e:
        logger.warning("dashboard_alerts_failed", error=str(e))

    # ------------------------------------------------------------------
    # 6. Risk status (circuit breaker)
    # ------------------------------------------------------------------
    risk_status: dict = {}
    try:
        from app.services.risk.drawdown import circuit_breaker
        risk_status = circuit_breaker.get_status()
    except Exception as e:
        logger.warning("dashboard_risk_failed", error=str(e))
        risk_status = {"state": "unknown"}

    # ------------------------------------------------------------------
    # 7. Data quality score (average across all symbols)
    # ------------------------------------------------------------------
    data_quality_score = 0.0
    try:
        from app.services.data.quality import data_quality_monitor
        scores: list[float] = []
        for symbol in settings.symbols_list:
            report = await data_quality_monitor.generate_report(db, symbol, "1h", 7)
            scores.append(report.quality_score)
        data_quality_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    except Exception as e:
        logger.warning("dashboard_data_quality_failed", error=str(e))

    return {
        "equity": {
            "total": round(total_equity, 2),
            "available_balance": round(available_balance, 2),
            "total_exposure": round(total_exposure, 2),
            "unrealized_pnl": round(total_unrealized, 2),
        },
        "positions": {
            "open_count": len(open_positions),
            "list": positions_list,
        },
        "todays_pnl": {
            "total": round(todays_pnl, 2),
            "realized": round(realized_today, 2),
            "unrealized": round(total_unrealized, 2),
        },
        "engine": {
            "running": engine_running,
        },
        "recent_alerts": recent_alerts,
        "risk": risk_status,
        "data_quality_score": data_quality_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

"""Price alert API endpoints."""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.models.alert import PriceAlert

router = APIRouter(prefix="/alerts", tags=["alerts"])


class CreateAlertRequest(BaseModel):
    symbol: str
    condition: str  # "above" or "below"
    target_price: float


@router.post("")
async def create_alert(body: CreateAlertRequest, db: AsyncSession = Depends(get_db), _auth: dict = Depends(require_auth)):
    alert = PriceAlert(symbol=body.symbol, condition=body.condition, target_price=body.target_price)
    db.add(alert)
    await db.commit()
    return {"id": alert.id, "symbol": alert.symbol, "condition": alert.condition, "target_price": alert.target_price}


@router.get("")
async def list_alerts(db: AsyncSession = Depends(get_db), _auth: dict = Depends(require_auth)):
    result = await db.execute(select(PriceAlert).where(PriceAlert.is_active == True).order_by(PriceAlert.created_at.desc()))
    alerts = result.scalars().all()
    return [{"id": a.id, "symbol": a.symbol, "condition": a.condition, "target_price": a.target_price, "is_triggered": a.is_triggered, "triggered_at": str(a.triggered_at) if a.triggered_at else None} for a in alerts]


@router.delete("/{alert_id}")
async def delete_alert(alert_id: int, db: AsyncSession = Depends(get_db), _auth: dict = Depends(require_auth)):
    result = await db.execute(select(PriceAlert).where(PriceAlert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_active = False
    await db.commit()
    return {"status": "deleted"}

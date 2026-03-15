"""Price alerts and trading journal endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.dependencies import require_auth
from app.models.base import async_session_factory

router = APIRouter(prefix="/alerts", tags=["alerts"])


# --- Price Alerts ---

class CreateAlertRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    condition: str = Field(pattern="^(above|below)$")
    target_price: float = Field(gt=0)
    notes: str | None = None


@router.post("")
async def create_alert(req: CreateAlertRequest, _user: dict = Depends(require_auth)):
    from app.services.alerts.checker import alert_checker
    async with async_session_factory() as db:
        alert = await alert_checker.create_alert(
            db, req.symbol, req.condition, req.target_price, req.notes,
        )
        return {"id": alert.id, "symbol": alert.symbol, "condition": alert.condition,
                "target_price": alert.target_price, "status": "active"}


@router.get("")
async def list_alerts(active_only: bool = True, _user: dict = Depends(require_auth)):
    from app.services.alerts.checker import alert_checker
    async with async_session_factory() as db:
        alerts = await alert_checker.get_alerts(db, active_only=active_only)
        return [
            {"id": a.id, "symbol": a.symbol, "condition": a.condition,
             "target_price": a.target_price, "is_triggered": a.is_triggered,
             "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
             "notes": a.notes, "is_active": a.is_active}
            for a in alerts
        ]


@router.delete("/{alert_id}")
async def delete_alert(alert_id: int, _user: dict = Depends(require_auth)):
    from app.services.alerts.checker import alert_checker
    async with async_session_factory() as db:
        deleted = await alert_checker.delete_alert(db, alert_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"status": "deleted"}

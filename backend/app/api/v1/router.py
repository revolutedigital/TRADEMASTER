"""Aggregate API v1 router."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.system import router as system_router
from app.api.v1.market import router as market_router
from app.api.v1.trading import router as trading_router
from app.api.v1.portfolio import router as portfolio_router
from app.api.v1.signals import router as signals_router
from app.api.v1.backtest import router as backtest_router
from app.api.v1.export import router as export_router
from app.api.v1.risk import router as risk_router
from app.api.v1.notifications import router as notifications_router
from app.api.v1.settings import router as settings_router
from app.api.v1.audit import router as audit_router
from app.api.v1.api_keys import router as api_keys_router
from app.api.v1.security_txt import router as security_txt_router
from app.api.v1.alerts import router as alerts_router
from app.api.v1.journal import router as journal_router
from app.api.v1.admin import router as admin_router
from app.api.v1.ml import router as ml_router
from app.api.v1.tax import router as tax_router
from app.api.v1.events import router as events_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
api_router.include_router(market_router, prefix="/market", tags=["market"])
api_router.include_router(trading_router, prefix="/trading", tags=["trading"])
api_router.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(signals_router, prefix="/signals", tags=["signals"])
api_router.include_router(backtest_router, prefix="/backtest", tags=["backtest"])
api_router.include_router(export_router, prefix="/export", tags=["export"])
api_router.include_router(risk_router, prefix="/risk", tags=["risk"])
api_router.include_router(notifications_router, prefix="/notifications", tags=["notifications"])
api_router.include_router(settings_router, prefix="/settings", tags=["settings"])
api_router.include_router(audit_router)
api_router.include_router(api_keys_router)
api_router.include_router(security_txt_router)
api_router.include_router(alerts_router)
api_router.include_router(journal_router)
api_router.include_router(admin_router)
api_router.include_router(ml_router, prefix="/ml", tags=["ml"])
api_router.include_router(tax_router)
api_router.include_router(events_router, prefix="/architecture", tags=["architecture"])

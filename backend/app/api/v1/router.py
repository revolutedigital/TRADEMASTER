"""Aggregate API v1 router."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.system import router as system_router
from app.api.v1.market import router as market_router
from app.api.v1.trading import router as trading_router
from app.api.v1.portfolio import router as portfolio_router
from app.api.v1.signals import router as signals_router
from app.api.v1.backtest import router as backtest_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(system_router, prefix="/system", tags=["system"])
api_router.include_router(market_router, prefix="/market", tags=["market"])
api_router.include_router(trading_router, prefix="/trading", tags=["trading"])
api_router.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(signals_router, prefix="/signals", tags=["signals"])
api_router.include_router(backtest_router, prefix="/backtest", tags=["backtest"])

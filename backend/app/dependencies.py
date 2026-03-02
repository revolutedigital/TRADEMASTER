"""FastAPI dependency injection for database sessions, Redis, auth, and services.

Provides factory functions for all major services so that:
- Routers use Depends() instead of importing singletons
- Services can be swapped for testing
- Dependencies are explicit and traceable
"""

from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import event_bus
from app.core.security import verify_token
from app.models.base import async_session_factory
from app.repositories.order_repo import OrderRepository
from app.repositories.position_repo import PositionRepository, SnapshotRepository
from app.repositories.market_repo import MarketDataRepository

_bearer_scheme = HTTPBearer(auto_error=False)


# ========================================
# Infrastructure
# ========================================


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_event_bus():
    return event_bus


# ========================================
# Authentication
# ========================================


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """Verify JWT token from Bearer header OR httpOnly cookie. Returns decoded payload or raises 401."""
    token = None

    # 1. Try Bearer header first
    if credentials is not None:
        token = credentials.credentials

    # 2. Fall back to httpOnly cookie
    if token is None:
        token = request.cookies.get("access_token")

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


# ========================================
# Repositories (stateless, can be created per-request)
# ========================================


def get_order_repository() -> OrderRepository:
    return OrderRepository()


def get_position_repository() -> PositionRepository:
    return PositionRepository()


def get_snapshot_repository() -> SnapshotRepository:
    return SnapshotRepository()


def get_market_repository() -> MarketDataRepository:
    return MarketDataRepository()


# ========================================
# Services (access singletons via DI for testability)
# ========================================


def get_order_manager():
    from app.services.exchange.order_manager import order_manager
    return order_manager


def get_portfolio_tracker():
    from app.services.portfolio.tracker import portfolio_tracker
    return portfolio_tracker


def get_risk_manager():
    from app.services.risk.manager import risk_manager
    return risk_manager


def get_circuit_breaker():
    from app.services.risk.drawdown import circuit_breaker
    return circuit_breaker


def get_binance_client():
    from app.services.exchange.binance_client import binance_client
    return binance_client


def get_ml_pipeline():
    from app.services.ml.pipeline import ml_pipeline
    return ml_pipeline


def get_market_data_collector():
    from app.services.market.data_collector import market_data_collector
    return market_data_collector


def get_trading_engine():
    from app.services.trading_engine import trading_engine
    return trading_engine

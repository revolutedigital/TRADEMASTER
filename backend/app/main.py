"""TradeMaster FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

from app.config import settings
from app.core.events import event_bus
from app.core.exceptions import TradeMasterError, EXCEPTION_STATUS_MAP
from app.core.logging import get_logger, setup_logging
from app.core.metrics import metrics
from app.core.idempotency import IdempotencyMiddleware
from app.core.tracing import setup_tracing
from app.api.v1.router import api_router
from app.api.websocket.streams import router as ws_router
from app.core.rasp import RASPMiddleware
from app.core.honeypot import honeypot_router

logger = get_logger(__name__)

# Global state for trading engine control
_engine_enabled: bool = True
_background_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    setup_logging()
    logger.info(
        "trademaster_starting",
        env=settings.app_env,
        testnet=settings.binance_testnet,
        symbols=settings.symbols_list,
    )

    # --- Phase 0: Create/update database tables ---
    try:
        from app.models.base import engine, Base
        import app.models.market  # noqa: F401
        import app.models.trade  # noqa: F401
        import app.models.portfolio  # noqa: F401
        import app.models.signal  # noqa: F401
        import app.models.audit  # noqa: F401
        import app.models.api_key  # noqa: F401
        import app.models.alert  # noqa: F401
        import app.models.journal  # noqa: F401
        import app.models.lineage  # noqa: F401
        import app.models.user  # noqa: F401
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_ready")
    except Exception as e:
        logger.warning("database_setup_failed", error=str(e))

    # --- Phase 1: Connect infrastructure ---
    redis_ok = False
    try:
        await event_bus.connect()
        redis_ok = True
        logger.info("redis_connected")
    except Exception as e:
        logger.warning("redis_connection_failed", error=str(e))

    # --- Phase 2: Connect to Binance ---
    binance_ok = False
    try:
        from app.services.exchange.binance_client import binance_client
        await binance_client.connect()
        binance_ok = True
        logger.info("binance_connected")
    except Exception as e:
        logger.warning("binance_connection_failed", error=str(e))

    # --- Phase 3: Start background services (never crash the app) ---
    try:
        if redis_ok and binance_ok:
            await _start_background_services()
        elif redis_ok:
            logger.info("starting_partial_services", reason="binance_unavailable")
            from app.services.ws_broadcaster import ws_broadcaster
            await ws_broadcaster.start()
        else:
            logger.warning("background_services_skipped", redis=redis_ok, binance=binance_ok)
    except Exception as e:
        logger.error("background_services_failed", error=str(e))

    logger.info("services_initialized", redis=redis_ok, binance=binance_ok)

    yield

    # --- Shutdown ---
    await _stop_background_services()

    try:
        from app.services.exchange.binance_client import binance_client
        await binance_client.disconnect()
    except Exception:
        pass

    try:
        await event_bus.disconnect()
    except Exception:
        pass

    logger.info("trademaster_shutdown")


async def _start_background_services() -> None:
    """Start all background async tasks: WS streams, processors, trading engine, scheduler."""
    global _background_tasks

    # 1. WebSocket streams from Binance -> Redis
    from app.services.exchange.binance_client import binance_client
    from app.services.exchange.binance_ws import binance_ws_manager
    try:
        await binance_ws_manager.start(binance_client._client)
        logger.info("binance_ws_started")
    except Exception as e:
        logger.error("binance_ws_start_failed", error=str(e))

    # 2. Market stream processor (Redis -> Database)
    from app.services.market.stream_processor import market_stream_processor
    task = asyncio.create_task(market_stream_processor.start(), name="market_stream_processor")
    _background_tasks.append(task)

    # 3. WebSocket broadcaster (Redis -> Dashboard clients)
    from app.services.ws_broadcaster import ws_broadcaster
    await ws_broadcaster.start()

    # 4. Trading engine (consumes kline events, generates signals, executes trades)
    from app.services.trading_engine import trading_engine
    if _engine_enabled:
        task = asyncio.create_task(trading_engine.start(), name="trading_engine")
        _background_tasks.append(task)
        logger.info("trading_engine_task_created")

    # 5. Periodic scheduler
    from app.services.scheduler import scheduler
    from app.services.exchange.binance_client import binance_client as bc

    # Position check every 60 seconds
    async def check_positions():
        from app.services.trading_engine import trading_engine
        await trading_engine.check_positions()

    # Metrics update every 30 seconds
    async def update_metrics():
        try:
            balance = await bc.get_balance("USDT")
            metrics.total_equity.set(float(balance))
        except Exception:
            pass

    # Order reconciliation every 300 seconds
    async def reconcile_orders():
        from app.core.resilience import reconciler
        from app.models.base import async_session
        from app.repositories.position_repo import PositionRepository
        try:
            async with async_session() as session:
                repo = PositionRepository()
                open_positions = await repo.get_open(session)
                # Gather local open order IDs
                local_orders = [
                    {"exchange_order_id": str(p.exchange_order_id), "status": "OPEN"}
                    for p in open_positions if hasattr(p, "exchange_order_id") and p.exchange_order_id
                ]
                exchange_orders = await bc.get_open_orders()
                await reconciler.reconcile_orders(local_orders, exchange_orders)
        except Exception as e:
            logger.warning("order_reconciliation_failed", error=str(e))

    scheduler.add_task("position_check", check_positions, interval_seconds=5, run_immediately=True)
    scheduler.add_task("metrics_update", update_metrics, interval_seconds=15, run_immediately=True)
    scheduler.add_task("order_reconciliation", reconcile_orders, interval_seconds=300, run_immediately=False)
    scheduler.start_all()

    logger.info("all_background_services_started")


async def _stop_background_services() -> None:
    """Gracefully stop all background services."""
    from app.services.scheduler import scheduler
    await scheduler.stop_all()

    from app.services.ws_broadcaster import ws_broadcaster
    await ws_broadcaster.stop()

    from app.services.market.stream_processor import market_stream_processor
    await market_stream_processor.stop()

    from app.services.trading_engine import trading_engine
    await trading_engine.stop()

    from app.services.exchange.binance_ws import binance_ws_manager
    await binance_ws_manager.stop()

    # Cancel any remaining tasks
    for task in _background_tasks:
        task.cancel()
    await asyncio.gather(*_background_tasks, return_exceptions=True)
    _background_tasks.clear()

    logger.info("all_background_services_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TradeMaster",
        description="AI-powered cryptocurrency trading platform for BTC & ETH via Binance",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware order: last added = outermost (processes request first).
    # CORS must be outermost so preflight OPTIONS are handled before other middleware.

    # Idempotency middleware (safe POST retries)
    app.add_middleware(IdempotencyMiddleware)

    # Request logging middleware
    from app.core.logging import RequestLoggingMiddleware
    app.add_middleware(RequestLoggingMiddleware)

    # RASP - Runtime Application Self-Protection (SQLi, XSS, path traversal detection)
    app.add_middleware(RASPMiddleware)

    # CSRF validation middleware
    from app.core.security import CSRFMiddleware
    app.add_middleware(CSRFMiddleware)

    # Security headers middleware
    from app.core.security import SecurityHeadersMiddleware
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS - MUST be outermost (added last) so preflight OPTIONS gets CORS headers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
    )

    # Global exception handler: maps domain exceptions to proper HTTP responses
    @app.exception_handler(TradeMasterError)
    async def trademaster_exception_handler(_request: StarletteRequest, exc: TradeMasterError) -> JSONResponse:
        status_code = EXCEPTION_STATUS_MAP.get(type(exc), 500)
        return JSONResponse(
            status_code=status_code,
            content={"error": exc.code, "message": exc.message},
        )

    # API routes
    app.include_router(api_router, prefix="/api/v1")

    # API v2 routes
    from app.api.v2.router import v2_router
    app.include_router(v2_router, prefix="/api/v2", tags=["v2"])

    # WebSocket routes
    app.include_router(ws_router)

    # Honeypot endpoints (hidden from OpenAPI docs, trap attacker probing)
    app.include_router(honeypot_router)

    # OpenTelemetry tracing (if available)
    setup_tracing(app)

    return app


app = create_app()

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
from app.api.v1 import data_quality as data_quality_router
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
        import app.models.event  # noqa: F401
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_ready")
    except Exception as e:
        logger.warning("database_setup_failed", error=str(e))

    # --- Phase 0b: Rehydrate event store from PostgreSQL ---
    try:
        from app.core.event_store import event_store
        loaded = await event_store.load_from_db()
        logger.info("event_store_rehydrated", events_loaded=loaded)
    except Exception as e:
        logger.warning("event_store_rehydrate_failed", error=str(e))

    # --- Phase 1: Connect infrastructure ---
    redis_ok = False
    try:
        await event_bus.connect()
        redis_ok = True
        logger.info("redis_connected")
    except Exception as e:
        logger.warning("redis_connection_failed", error=str(e))

    # --- Phase 1b: Initialize feature flags (Redis-backed) ---
    try:
        from app.core.feature_flags import feature_flags
        await feature_flags.init()
        logger.info("feature_flags_initialized")
    except Exception as e:
        logger.warning("feature_flags_init_failed", error=str(e))

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
            await _start_background_services_offline()
        else:
            logger.warning("background_services_skipped", redis=redis_ok, binance=binance_ok)
    except Exception as e:
        logger.error("background_services_failed", error=str(e))

    logger.info("services_initialized", redis=redis_ok, binance=binance_ok)

    yield

    # --- Graceful Shutdown ---
    logger.info("graceful_shutdown_started")

    # 1. Stop trading engine first (no new signals)
    try:
        from app.services.trading_engine import trading_engine
        await trading_engine.stop()
        logger.info("trading_engine_stopped_for_shutdown")
    except Exception as e:
        logger.warning("trading_engine_stop_failed", error=str(e))

    # 2. Wait for pending orders to complete (max 30s)
    try:
        from app.models.base import async_session_factory
        from app.models.trade import Order, OrderStatus
        from sqlalchemy import select

        pending_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        deadline = asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < deadline:
            async with async_session_factory() as db:
                result = await db.execute(
                    select(Order).where(Order.status.in_(pending_statuses))
                )
                pending = list(result.scalars().all())
            if not pending:
                logger.info("all_pending_orders_resolved")
                break
            logger.info("waiting_for_pending_orders", count=len(pending))
            await asyncio.sleep(2)
        else:
            logger.warning("pending_orders_timeout", message="30s deadline exceeded")
    except Exception as e:
        logger.warning("pending_orders_check_failed", error=str(e))

    # 3. Take final portfolio snapshot
    try:
        from app.models.base import async_session_factory
        from app.services.portfolio.tracker import portfolio_tracker
        from app.services.exchange.binance_client import binance_client as bc

        async with async_session_factory() as db:
            try:
                balance = float(await bc.get_balance("USDT"))
            except Exception:
                balance = 0.0
            positions = await portfolio_tracker.get_open_positions(db)
            unrealized = sum(float(p.unrealized_pnl) for p in positions)
            equity = balance + unrealized
            await portfolio_tracker.take_snapshot(db, equity=equity, balance=balance)
            await db.commit()
        logger.info("final_portfolio_snapshot_saved", equity=equity, balance=balance)
    except Exception as e:
        logger.warning("final_snapshot_failed", error=str(e))

    # 4. Stop remaining background services
    await _stop_background_services()

    # 5. Disconnect external services
    try:
        from app.services.exchange.binance_client import binance_client
        await binance_client.disconnect()
    except Exception:
        pass

    try:
        await event_bus.disconnect()
    except Exception:
        pass

    logger.info("graceful_shutdown_complete")


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


async def _start_background_services_offline() -> None:
    """Start services in offline mode: synthetic klines replace Binance WebSocket."""
    global _background_tasks

    # 0. Bootstrap historical candles (CryptoCompare -> DB)
    #    Gives the engine 100+ candles immediately instead of waiting 7.5 hours
    try:
        from app.services.market.candle_bootstrap import bootstrap_historical_candles
        bootstrap_results = await bootstrap_historical_candles()
        logger.info("candle_bootstrap_done", results=bootstrap_results)
    except Exception as e:
        logger.warning("candle_bootstrap_failed", error=str(e))

    # 1. Autonomous price fetcher (CoinGecko/CryptoCompare -> Redis)
    from app.services.market.price_fetcher import price_fetcher
    task = asyncio.create_task(price_fetcher.start(), name="price_fetcher")
    _background_tasks.append(task)
    logger.info("price_fetcher_started")

    # 2. Synthetic kline generator (Redis prices -> KLINE_UPDATE events)
    from app.services.market.synthetic_kline_generator import synthetic_kline_generator
    task = asyncio.create_task(synthetic_kline_generator.start(), name="synthetic_kline_generator")
    _background_tasks.append(task)
    logger.info("synthetic_kline_generator_started")

    # 3. Market stream processor (KLINE_UPDATE events -> Database)
    from app.services.market.stream_processor import market_stream_processor
    task = asyncio.create_task(market_stream_processor.start(), name="market_stream_processor")
    _background_tasks.append(task)

    # 4. WebSocket broadcaster (Redis -> Dashboard clients)
    from app.services.ws_broadcaster import ws_broadcaster
    await ws_broadcaster.start()

    # 5. Trading engine
    from app.services.trading_engine import trading_engine
    if _engine_enabled:
        task = asyncio.create_task(trading_engine.start(), name="trading_engine")
        _background_tasks.append(task)
        logger.info("trading_engine_task_created_offline")

    # 6. Periodic position check (uses Redis-cached prices, no Binance needed)
    from app.services.scheduler import scheduler

    async def check_positions():
        from app.services.trading_engine import trading_engine
        await trading_engine.check_positions()

    scheduler.add_task("position_check", check_positions, interval_seconds=5, run_immediately=True)
    scheduler.start_all()

    logger.info("all_background_services_started_offline")


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

    try:
        from app.services.market.synthetic_kline_generator import synthetic_kline_generator
        await synthetic_kline_generator.stop()
    except Exception:
        pass

    try:
        from app.services.market.price_fetcher import price_fetcher
        await price_fetcher.stop()
    except Exception:
        pass

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
    app.include_router(data_quality_router.router, prefix="/api/v1", tags=["data-quality"])

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

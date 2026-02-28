"""TradeMaster FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.events import event_bus
from app.core.logging import get_logger, setup_logging
from app.core.metrics import metrics
from app.api.v1.router import api_router
from app.api.websocket.streams import router as ws_router

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

    # --- Phase 0: Ensure database tables exist ---
    try:
        from app.models.base import engine, Base
        import app.models.market  # noqa: F401
        import app.models.trade  # noqa: F401
        import app.models.portfolio  # noqa: F401
        import app.models.signal  # noqa: F401
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_created")
    except Exception as e:
        logger.warning("database_tables_failed", error=str(e))

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

    # --- Phase 3: Start background services ---
    if redis_ok and binance_ok:
        await _start_background_services()
    elif redis_ok:
        # Binance unavailable (geo-restricted) - start what we can
        logger.info("starting_partial_services", reason="binance_unavailable")
        from app.services.ws_broadcaster import ws_broadcaster
        await ws_broadcaster.start()
    else:
        logger.warning("background_services_skipped", redis=redis_ok, binance=binance_ok)

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

    scheduler.add_task("position_check", check_positions, interval_seconds=5, run_immediately=True)
    scheduler.add_task("metrics_update", update_metrics, interval_seconds=15, run_immediately=True)
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

    # CORS - restricted to frontend origin(s)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(api_router, prefix="/api/v1")

    # WebSocket routes
    app.include_router(ws_router)

    return app


app = create_app()

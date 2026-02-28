"""Run the TradeMaster trading engine locally.

This script starts the full trading loop from your local machine,
connecting to Binance Testnet for market data and execution,
and Railway PostgreSQL + Redis for data storage and events.

Usage:
    python scripts/run_trading.py
"""

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.core.events import event_bus
from app.core.logging import get_logger, setup_logging
from app.core.metrics import metrics
from app.models.base import engine, Base, async_session_factory
from app.services.exchange.binance_client import binance_client
from app.services.exchange.binance_ws import binance_ws_manager
from app.services.market.stream_processor import market_stream_processor
from app.services.ml.pipeline import ml_pipeline
from app.services.trading_engine import trading_engine

# Import models to register them
import app.models.market  # noqa: F401
import app.models.trade  # noqa: F401
import app.models.portfolio  # noqa: F401
import app.models.signal  # noqa: F401

logger = get_logger(__name__)

_shutdown_event = asyncio.Event()


async def position_check_loop():
    """Periodically check stop losses and update positions."""
    while not _shutdown_event.is_set():
        try:
            await trading_engine.check_positions()
        except Exception as e:
            logger.error("position_check_error", error=str(e))
        await asyncio.sleep(60)


async def metrics_update_loop():
    """Periodically update metrics."""
    while not _shutdown_event.is_set():
        try:
            balance = await binance_client.get_balance("USDT")
            metrics.total_equity.set(float(balance))
        except Exception:
            pass
        await asyncio.sleep(30)


async def main():
    setup_logging()
    logger.info(
        "trading_engine_local_start",
        env=settings.app_env,
        testnet=settings.binance_testnet,
        symbols=settings.symbols_list,
    )

    # 1. Ensure database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_ok")

    # 2. Connect Redis
    try:
        await event_bus.connect()
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_failed", error=str(e))
        return

    # 3. Connect Binance
    try:
        await binance_client.connect()
        logger.info("binance_connected")
    except Exception as e:
        logger.error("binance_failed", error=str(e))
        return

    # 4. Load ML models
    for symbol in settings.symbols_list:
        await ml_pipeline.load_models(symbol)
    logger.info("ml_models_loaded")

    # 5. Show account balance
    try:
        balance = await binance_client.get_balance("USDT")
        logger.info("account_balance", usdt=float(balance))
    except Exception as e:
        logger.warning("balance_check_failed", error=str(e))

    # 6. Start all background tasks
    tasks = []

    # WebSocket streams (Binance -> Redis)
    await binance_ws_manager.start(binance_client._client)
    logger.info("websocket_streams_started")

    # Market stream processor (Redis -> Database)
    tasks.append(asyncio.create_task(market_stream_processor.start(), name="stream_processor"))

    # Trading engine (main loop)
    tasks.append(asyncio.create_task(trading_engine.start(), name="trading_engine"))

    # Position monitoring
    tasks.append(asyncio.create_task(position_check_loop(), name="position_check"))

    # Metrics
    tasks.append(asyncio.create_task(metrics_update_loop(), name="metrics_update"))

    logger.info(
        "all_services_running",
        tasks=len(tasks),
        message="Paper trading active on Binance Testnet. Press Ctrl+C to stop.",
    )

    # Wait for shutdown signal
    try:
        await _shutdown_event.wait()
    except asyncio.CancelledError:
        pass

    # Graceful shutdown
    logger.info("shutting_down")
    await trading_engine.stop()
    await market_stream_processor.stop()
    await binance_ws_manager.stop()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await binance_client.disconnect()
    await event_bus.disconnect()
    logger.info("trading_engine_stopped")


def handle_signal(sig, frame):
    logger.info("signal_received", signal=sig)
    _shutdown_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    asyncio.run(main())

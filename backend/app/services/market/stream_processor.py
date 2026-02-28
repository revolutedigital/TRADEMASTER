"""Redis Streams consumer for processing market data events."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import EventType, event_bus
from app.core.logging import get_logger
from app.models.base import async_session_factory
from app.services.market.data_collector import market_data_collector

logger = get_logger(__name__)


class MarketStreamProcessor:
    """Consumes kline events from Redis Streams and persists closed candles."""

    def __init__(self) -> None:
        self._running: bool = False

    async def start(self) -> None:
        """Start consuming kline events."""
        self._running = True
        logger.info("market_stream_processor_started")

        while self._running:
            try:
                events = await event_bus.subscribe(
                    event_types=[EventType.KLINE_UPDATE],
                    group="market_data_store",
                    consumer="processor_1",
                    count=50,
                    block_ms=5000,
                )

                if not events:
                    continue

                async with async_session_factory() as db:
                    for event in events:
                        data = event.data
                        if data.get("is_closed"):
                            await market_data_collector.store_kline(
                                db=db,
                                symbol=data["symbol"],
                                interval=data["interval"],
                                data=data,
                            )
                    await db.commit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("stream_processor_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("market_stream_processor_stopped")

    async def stop(self) -> None:
        self._running = False


market_stream_processor = MarketStreamProcessor()

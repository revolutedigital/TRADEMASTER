"""Redis Streams consumer for processing market data events."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import Event, EventType, event_bus
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
                    acknowledge=False,
                    retry_pending=True,
                )

                if not events:
                    continue
                await self._persist_closed_events(events)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("stream_processor_error", error=str(e))
                await asyncio.sleep(1)

        logger.info("market_stream_processor_stopped")

    async def stop(self) -> None:
        self._running = False

    async def _persist_closed_events(self, events: list[Event]) -> None:
        """Commit closed candles before making them eligible for trading."""
        closed_events = [event for event in events if event.data.get("is_closed")]
        if closed_events:
            async with async_session_factory() as db:
                for event in closed_events:
                    data = event.data
                    await market_data_collector.store_kline(
                        db=db,
                        symbol=data["symbol"],
                        interval=data["interval"],
                        data=data,
                    )
                await db.commit()

        # An acknowledgement after the commit keeps failed writes pending for
        # retry. It also comes before the derived event, preventing duplicate
        # strategy inputs when Redis redelivers the raw candle.
        await event_bus.acknowledge(events)

        # Consumer groups do not provide cross-group ordering. Publishing this
        # distinct event only after the transaction commits guarantees the
        # engine reads the exact candle that triggered its signal.
        for event in closed_events:
            await event_bus.publish(
                Event(
                    type=EventType.KLINE_CLOSED_PERSISTED,
                    data=event.data,
                    source="market_stream_processor",
                )
            )


market_stream_processor = MarketStreamProcessor()

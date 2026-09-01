"""Tests for the post-commit candle event emitted to the trading engine."""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.events import Event, EventType
from app.services.market.stream_processor import MarketStreamProcessor


@pytest.mark.asyncio
async def test_closed_candle_is_emitted_to_the_engine_only_after_database_commit() -> None:
    processor = MarketStreamProcessor()
    database = AsyncMock()
    database.commit = AsyncMock()
    closed_event = Event(
        type=EventType.KLINE_UPDATE,
        data={"symbol": "BTCUSDT", "interval": "1h", "is_closed": True},
    )
    open_event = Event(
        type=EventType.KLINE_UPDATE,
        data={"symbol": "BTCUSDT", "interval": "1h", "is_closed": False},
    )

    class SessionContext:
        async def __aenter__(self):
            return database

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    delivery_order: list[str] = []

    async def acknowledge(events: list[Event]) -> None:
        database.commit.assert_awaited_once()
        assert events == [open_event, closed_event]
        delivery_order.append("ack")

    async def publish(event: Event) -> None:
        assert delivery_order == ["ack"]
        assert event.type == EventType.KLINE_CLOSED_PERSISTED
        assert event.data == closed_event.data

    with (
        patch(
            "app.services.market.stream_processor.async_session_factory",
            return_value=SessionContext(),
        ),
        patch(
            "app.services.market.stream_processor.market_data_collector.store_kline",
            new=AsyncMock(),
        ) as store_kline,
        patch(
            "app.services.market.stream_processor.event_bus.acknowledge",
            new=AsyncMock(side_effect=acknowledge),
        ) as acknowledge_events,
        patch(
            "app.services.market.stream_processor.event_bus.publish",
            new=AsyncMock(side_effect=publish),
        ) as publish_event,
    ):
        await processor._persist_closed_events([open_event, closed_event])

    store_kline.assert_awaited_once_with(
        db=database,
        symbol="BTCUSDT",
        interval="1h",
        data=closed_event.data,
    )
    publish_event.assert_awaited_once()
    acknowledge_events.assert_awaited_once_with([open_event, closed_event])

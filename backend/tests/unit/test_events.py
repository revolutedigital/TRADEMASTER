"""Tests for event system."""

from unittest.mock import AsyncMock

import pytest

from app.core.events import Event, EventBus, EventType


def test_event_creation():
    event = Event(
        type=EventType.KLINE_UPDATE,
        data={"symbol": "BTCUSDT", "close": 85000.0},
    )
    assert event.type == EventType.KLINE_UPDATE
    assert event.data["symbol"] == "BTCUSDT"
    assert event.source == "trademaster"
    assert event.timestamp is not None


def test_event_types():
    assert EventType.KLINE_UPDATE == "kline.update"
    assert EventType.KLINE_CLOSED_PERSISTED == "kline.closed.persisted"
    assert EventType.ORDER_FILLED == "order.filled"
    assert EventType.CIRCUIT_BREAKER_TRIGGERED == "risk.circuit_breaker"


def test_event_type_is_string():
    assert isinstance(EventType.SIGNAL_GENERATED, str)
    assert EventType.SIGNAL_GENERATED == "signal.generated"


@pytest.mark.asyncio
async def test_manual_acknowledgement_uses_event_delivery_metadata() -> None:
    bus = EventBus()
    bus._redis = AsyncMock()
    event = Event(
        type=EventType.KLINE_UPDATE,
        data={},
        _stream_key="stream:kline.update",
        _message_id="1720000000000-0",
        _consumer_group="market_data_store",
    )

    await bus.acknowledge([event])

    bus._redis.xack.assert_awaited_once_with(
        "stream:kline.update",
        "market_data_store",
        "1720000000000-0",
    )

"""Tests for event system."""

from app.core.events import Event, EventType


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
    assert EventType.ORDER_FILLED == "order.filled"
    assert EventType.CIRCUIT_BREAKER_TRIGGERED == "risk.circuit_breaker"


def test_event_type_is_string():
    assert isinstance(EventType.SIGNAL_GENERATED, str)
    assert EventType.SIGNAL_GENERATED == "signal.generated"

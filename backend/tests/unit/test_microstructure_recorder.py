"""Market stream parsing and WAL persistence preserve causal event facts."""

import asyncio
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.market.microstructure_recorder import (
    JsonlEventSink,
    MicrostructureRecorder,
    SequenceTracker,
    StreamSequenceGap,
    parse_market_message,
)


class InMemorySink:
    def __init__(self) -> None:
        self.events: list[MicrostructureEvent] = []

    async def append_batch(self, events: list[MicrostructureEvent]) -> None:
        self.events = events


class FakeWebSocketConnection:
    def __init__(self, messages: list[dict], stop_event: asyncio.Event) -> None:
        self._messages = list(messages)
        self._stop_event = stop_event

    async def __aenter__(self) -> "FakeWebSocketConnection":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def recv(self) -> str:
        if not self._messages:
            self._stop_event.set()
            raise TimeoutError
        message = self._messages.pop(0)
        if not self._messages:
            self._stop_event.set()
        return json.dumps(message)


def test_parse_aggregate_trade_keeps_aggressor_and_sequence() -> None:
    event = parse_market_message(
        stream="btcusdt@aggTrade",
        payload={
            "e": "aggTrade",
            "T": 1_767_225_600_000,
            "a": 10,
            "f": 20,
            "l": 22,
            "p": "100",
            "q": "2",
            "m": True,
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )

    assert event is not None
    assert event.event_type == MarketEventType.AGG_TRADE
    assert event.sequence_id == 10
    assert event.side == "SELL"
    assert event.quote_quantity == 200


def test_parse_depth_keeps_updates_for_reconstruction() -> None:
    event = parse_market_message(
        stream="btcusdt@depth@100ms",
        payload={
            "e": "depthUpdate",
            "E": 1_767_225_600_000,
            "T": 1_767_225_600_001,
            "U": 100,
            "u": 101,
            "pu": 99,
            "b": [["99", "2"]],
            "a": [["101", "3"]],
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )

    assert event is not None and event.event_type == MarketEventType.DEPTH
    assert event.payload == {
        "previous_final_update_id": 99,
        "bids": [["99", "2"]],
        "asks": [["101", "3"]],
    }


def test_parse_raw_trade_supports_non_contiguous_trade_ids() -> None:
    event = parse_market_message(
        stream="btcusdt@trade",
        payload={
            "e": "trade",
            "T": 1_767_225_600_000,
            "t": 12,
            "p": "100",
            "q": "0.5",
            "m": False,
            "X": "MARKET",
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )

    tracker = SequenceTracker()
    assert tracker.observe("trade", 10, require_contiguous=False) is True
    assert tracker.observe("trade", 12, require_contiguous=False) is True
    assert event is not None and event.event_type == MarketEventType.TRADE
    assert event.side == "BUY"
    assert event.payload == {"order_type": "MARKET"}


def test_parse_mark_price_stream_keeps_index_and_funding_payload() -> None:
    event = parse_market_message(
        stream="btcusdt@markPrice@1s",
        payload={
            "e": "markPriceUpdate",
            "E": 1_767_225_600_000,
            "s": "BTCUSDT",
            "p": "100.5",
            "i": "100.1",
            "P": "100.2",
            "r": "0.0001",
            "T": 1_767_254_400_000,
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )

    assert event is not None
    assert event.event_type == MarketEventType.MARK_PRICE
    assert event.price == 100.5
    assert event.payload == {
        "index_price": 100.1,
        "estimated_settle_price": 100.2,
        "funding_rate": 0.0001,
        "next_funding_time": 1_767_254_400_000,
    }


def test_parse_spot_trade_marks_product_and_omits_futures_order_payload() -> None:
    event = parse_market_message(
        stream="btcusdt@trade",
        payload={
            "e": "trade",
            "E": 1_767_225_600_000,
            "T": 1_767_225_600_000,
            "t": 12,
            "p": "100",
            "q": "0.5",
            "m": False,
            "M": True,
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        product="spot",
    )

    assert event is not None
    assert event.product == "spot"
    assert event.event_type == MarketEventType.TRADE
    assert event.side == "BUY"
    assert event.payload is None


def test_zero_quantity_trade_heartbeat_is_ignored() -> None:
    event = parse_market_message(
        stream="btcusdt@trade",
        payload={
            "e": "trade",
            "T": 1_767_225_600_000,
            "t": 12,
            "p": "0",
            "q": "0",
            "m": False,
            "X": "NA",
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )

    assert event is None


def test_liquidation_uses_order_price_when_average_price_is_zero() -> None:
    event = parse_market_message(
        stream="btcusdt@forceOrder",
        payload={
            "e": "forceOrder",
            "E": 1_767_225_600_000,
            "o": {
                "T": 1_767_225_600_001,
                "S": "SELL",
                "p": "100",
                "ap": "0.00000",
                "q": "2",
                "z": "0",
                "X": "NEW",
                "o": "LIMIT",
            },
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )

    assert event is not None and event.event_type == MarketEventType.LIQUIDATION
    assert event.price == 100
    assert event.quantity == 2


def test_unpriced_liquidation_order_is_retained_without_fake_price() -> None:
    event = parse_market_message(
        stream="btcusdt@forceOrder",
        payload={
            "e": "forceOrder",
            "E": 1_767_225_600_000,
            "o": {
                "T": 1_767_225_600_001,
                "S": "BUY",
                "p": "0",
                "ap": "0",
                "sp": "0",
                "q": "1",
                "z": "0",
                "X": "NEW",
                "o": "MARKET",
            },
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )

    assert event is not None
    assert event.price is None


def test_sequence_tracker_rejects_a_gap_and_ignores_duplicates() -> None:
    tracker = SequenceTracker()
    assert tracker.observe("agg", 10) is True
    assert tracker.observe("agg", 10) is False
    with pytest.raises(StreamSequenceGap):
        tracker.observe("agg", 12)


@pytest.mark.asyncio
async def test_jsonl_sink_partitions_and_persists_events(tmp_path: Path) -> None:
    event = parse_market_message(
        stream="btcusdt@bookTicker",
        payload={
            "e": "bookTicker",
            "E": 1_767_225_600_000,
            "u": 10,
            "b": "99",
            "B": "2",
            "a": "101",
            "A": "3",
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )
    assert event is not None

    await JsonlEventSink(tmp_path).append_batch([event])

    paths = list(tmp_path.rglob("events.jsonl.gz"))
    assert len(paths) == 1
    with gzip.open(paths[0], "rt", encoding="utf-8") as source:
        persisted = source.read()
    assert '"order_submission_allowed"' not in persisted
    assert '"BOOK_TICKER"' in persisted


@pytest.mark.asyncio
async def test_jsonl_sink_routes_spot_trades_to_separate_directory(tmp_path: Path) -> None:
    futures_event = parse_market_message(
        stream="btcusdt@trade",
        payload={
            "e": "trade",
            "T": 1_767_225_600_000,
            "t": 12,
            "p": "100",
            "q": "0.5",
            "m": False,
            "X": "MARKET",
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
    )
    spot_event = parse_market_message(
        stream="btcusdt@trade",
        payload={
            "e": "trade",
            "T": 1_767_225_600_000,
            "t": 13,
            "p": "101",
            "q": "0.25",
            "m": True,
        },
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        product="spot",
    )
    assert futures_event is not None
    assert spot_event is not None

    await JsonlEventSink(tmp_path).append_batch([futures_event, spot_event])

    paths = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("events.jsonl.gz")}
    assert paths == {
        "trade/date=2026-01-01/events.jsonl.gz",
        "spot_trade/date=2026-01-01/events.jsonl.gz",
    }


@pytest.mark.asyncio
async def test_jsonl_sink_rejects_unsafe_product_directory(tmp_path: Path) -> None:
    event = MicrostructureEvent(
        product="../spot",
        symbol="BTCUSDT",
        event_type=MarketEventType.TRADE,
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
        price=100,
        quantity=1,
    )

    with pytest.raises(ValueError, match="Unsafe event product"):
        await JsonlEventSink(tmp_path).append_batch([event])


def test_recorder_spot_stream_url_is_opt_in() -> None:
    recorder = MicrostructureRecorder(
        sink=InMemorySink(),
        stream_base_url="wss://futures.test/stream",
        spot_stream_base_url="wss://spot.test/stream",
        include_spot_trades=True,
    )

    assert (
        recorder.stream_url
        == "wss://futures.test/stream?streams=btcusdt@trade/"
        "btcusdt@depth@100ms/btcusdt@markPrice@1s/btcusdt@forceOrder"
    )
    assert recorder.spot_stream_url == "wss://spot.test/stream?streams=btcusdt@trade"


@pytest.mark.asyncio
async def test_recorder_persists_depth_snapshot_boundary_before_depth_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = asyncio.Event()
    depth_update = {
        "stream": "btcusdt@depth@100ms",
        "data": {
            "e": "depthUpdate",
            "E": 1_767_225_600_100,
            "T": 1_767_225_600_100,
            "U": 100,
            "u": 101,
            "pu": 99,
            "b": [["99", "1.5"]],
            "a": [["101", "2.5"]],
        },
    }

    def fake_connect(*_args: object, **_kwargs: object) -> FakeWebSocketConnection:
        return FakeWebSocketConnection([depth_update], stop_event)

    async def fake_snapshot() -> dict:
        return {
            "lastUpdateId": 100,
            "bids": [["99", "2"]],
            "asks": [["101", "3"]],
        }

    monkeypatch.setattr(
        "app.services.market.microstructure_recorder.websockets.connect",
        fake_connect,
    )
    recorder = MicrostructureRecorder(
        sink=InMemorySink(),
        snapshot_fetcher=fake_snapshot,
    )

    await recorder._run_connection(stop_event)

    queued_events = [recorder._queue.get_nowait(), recorder._queue.get_nowait()]
    assert queued_events[0].event_type == MarketEventType.DEPTH
    assert queued_events[0].sequence_id == 100
    assert queued_events[0].payload == {
        "kind": "depth_snapshot",
        "source": "rest_depth",
        "last_update_id": 100,
        "bids": [["99", "2"]],
        "asks": [["101", "3"]],
    }
    assert queued_events[1].event_type == MarketEventType.DEPTH
    assert queued_events[1].sequence_id == 101
    assert recorder._queue.empty()

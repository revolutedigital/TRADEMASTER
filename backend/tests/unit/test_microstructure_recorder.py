"""Market stream parsing and WAL persistence preserve causal event facts."""

import gzip
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.schemas.microstructure import MarketEventType
from app.services.market.microstructure_recorder import (
    JsonlEventSink,
    SequenceTracker,
    StreamSequenceGap,
    parse_market_message,
)


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

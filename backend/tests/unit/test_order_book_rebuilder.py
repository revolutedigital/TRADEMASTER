"""Futures order-book reconstruction fails closed on every sequence gap."""

import pytest

from app.services.market.order_book_rebuilder import OrderBookGapError, OrderBookRebuilder


def _snapshot():
    return {
        "lastUpdateId": 100,
        "bids": [["99", "2"], ["98", "3"]],
        "asks": [["101", "4"], ["102", "5"]],
    }


def test_snapshot_bridge_and_incremental_updates() -> None:
    book = OrderBookRebuilder()
    book.initialize(_snapshot())

    first = book.apply({"U": 99, "u": 101, "pu": 98, "b": [["99", "0"], ["100", "1"]], "a": []})
    second = book.apply({"U": 102, "u": 102, "pu": 101, "b": [], "a": [["101", "2"]]})

    assert first is not None and float(first.bid_price) == 100
    assert second is not None and float(second.ask_quantity) == 2
    assert second.update_id == 102


def test_sequence_gap_requires_a_new_snapshot() -> None:
    book = OrderBookRebuilder()
    book.initialize(_snapshot())
    book.apply({"U": 100, "u": 101, "pu": 99, "b": [], "a": []})

    with pytest.raises(OrderBookGapError, match="sequence gap"):
        book.apply({"U": 103, "u": 103, "pu": 102, "b": [], "a": []})


def test_crossed_book_is_rejected() -> None:
    book = OrderBookRebuilder()
    book.initialize(_snapshot())

    with pytest.raises(OrderBookGapError, match="crossed"):
        book.apply({"U": 100, "u": 101, "pu": 99, "b": [["103", "1"]], "a": []})


def test_stale_update_is_ignored() -> None:
    book = OrderBookRebuilder()
    book.initialize(_snapshot())
    assert book.apply({"U": 90, "u": 99, "pu": 89, "b": [], "a": []}) is None

"""Online shadow CLI helpers convert normalized rows into causal events safely."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.schemas.microstructure import MarketEventType
from scripts.research.run_online_shadow_policy import (
    _events_from_frames,
    decision_times_between,
    required_event_sources,
)


def test_online_shadow_cli_converts_frames_to_chronological_events() -> None:
    events = _events_from_frames(
        symbol="BTCUSDT",
        trade_frame=pd.DataFrame(
            [
                {
                    "event_time_ms": 1_000,
                    "sequence_id": 2,
                    "price": 100.0,
                    "quantity": 2.0,
                    "is_buyer_maker": False,
                }
            ]
        ),
        spot_frame=pd.DataFrame(
            [
                {
                    "event_time_ms": 1_000,
                    "sequence_id": 1,
                    "price": 99.0,
                    "quantity": 1.0,
                    "is_buyer_maker": True,
                }
            ]
        ),
        book_frame=pd.DataFrame(
            [
                {
                    "event_time_ms": 900,
                    "sequence_id": 10,
                    "bid_price": 99.5,
                    "bid_quantity": 3.0,
                    "ask_price": 100.5,
                    "ask_quantity": 4.0,
                }
            ]
        ),
        mark_frame=pd.DataFrame(
            [
                {
                    "event_time_ms": 800,
                    "mark_price": 100.1,
                    "index_price": 100.0,
                    "funding_rate": 0.0001,
                }
            ]
        ),
        liquidation_frame=pd.DataFrame(
            [
                {
                    "event_time_ms": 1_100,
                    "sequence_id": 3,
                    "price": 0.0,
                    "quantity": 5.0,
                    "side": "SELL",
                }
            ]
        ),
    )

    assert [event.event_type for event in events] == [
        MarketEventType.MARK_PRICE,
        MarketEventType.DEPTH,
        MarketEventType.TRADE,
        MarketEventType.TRADE,
        MarketEventType.LIQUIDATION,
    ]
    assert [event.product for event in events] == [
        "usdm_perpetual",
        "usdm_perpetual",
        "usdm_perpetual",
        "spot",
        "usdm_perpetual",
    ]
    assert events[2].quote_quantity == 200
    assert events[3].side == "SELL"
    assert events[4].price is None
    assert events[4].quantity == 5


def test_online_shadow_cli_decision_times_are_start_inclusive_end_exclusive() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(seconds=11)

    decision_times = decision_times_between(start, end, stride_seconds=5)

    assert decision_times == (
        int(start.timestamp() * 1000),
        int((start + timedelta(seconds=5)).timestamp() * 1000),
        int((start + timedelta(seconds=10)).timestamp() * 1000),
    )


def test_online_shadow_cli_detects_artifact_required_event_sources() -> None:
    artifact = {
        "feature_columns": [
            "book_available",
            "directed_book_pressure_imbalance_1s",
            "funding_rate",
            "directed_liquidation_net_qty_1s",
            "spot_perp_basis_bps",
        ]
    }

    assert required_event_sources(artifact) == {
        "book": True,
        "mark": True,
        "liquidation": True,
        "spot": True,
    }

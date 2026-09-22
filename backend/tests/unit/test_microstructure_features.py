"""Offline vectorization must match the causal online feature engine."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.research.microstructure_features import (
    CausalMicrostructureFeatureEngine,
    materialize_trade_flow_features,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_trade(offset_ms: int, price: float, quantity: float, maker: bool):
    return MicrostructureEvent(
        product="usdm_perpetual",
        symbol="BTCUSDT",
        event_type=MarketEventType.TRADE,
        event_time=NOW + timedelta(milliseconds=offset_ms),
        price=price,
        quantity=quantity,
        quote_quantity=price * quantity,
        is_buyer_maker=maker,
    )


def test_offline_trade_features_equal_online_snapshot() -> None:
    offsets = np.array([0, 400, 900, 1400, 2600], dtype=np.int64)
    prices = np.array([100, 101, 100, 102, 103], dtype=float)
    quantities = np.array([1, 2, 1, 3, 1], dtype=float)
    makers = np.array([False, False, True, True, False])
    engine = CausalMicrostructureFeatureEngine(windows_seconds=(1, 5))
    for offset, price, quantity, maker in zip(offsets, prices, quantities, makers, strict=True):
        engine.consume(make_trade(int(offset), price, quantity, bool(maker)))
    decision = int(NOW.timestamp() * 1000) + 2600
    online = engine.snapshot(decision).values

    offline = materialize_trade_flow_features(
        pd.DataFrame(
            {
                "event_time_ms": offsets + int(NOW.timestamp() * 1000),
                "price": prices,
                "quantity": quantities,
                "is_buyer_maker": makers,
            }
        ),
        np.array([decision]),
        windows_seconds=(1, 5),
    ).iloc[0]

    for window in (1, 5):
        for name in (
            "trade_count",
            "quote_volume",
            "flow_imbalance",
            "return",
            "realized_vol",
            "mean_interarrival_ms",
        ):
            suffix = "_bps" if name in {"return", "realized_vol"} else ""
            key = f"{name}_{window}s{suffix}"
            assert offline[key] == pytest.approx(online[key])
    assert offline["hour_sin"] == pytest.approx(online["hour_sin"])
    assert offline["hour_cos"] == pytest.approx(online["hour_cos"])


def test_book_microprice_mark_basis_and_liquidation_are_causal() -> None:
    engine = CausalMicrostructureFeatureEngine(windows_seconds=(5,))
    engine.consume(
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.BOOK_TICKER,
            event_time=NOW,
            bid_price=99,
            bid_quantity=3,
            ask_price=101,
            ask_quantity=1,
        )
    )
    engine.consume(
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.MARK_PRICE,
            event_time=NOW,
            price=101,
            payload={"index_price": 100, "funding_rate": 0.0001},
        )
    )
    engine.consume(
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.LIQUIDATION,
            event_time=NOW,
            quantity=2,
            side="SELL",
        )
    )

    values = engine.snapshot(int(NOW.timestamp() * 1000)).values

    assert values["book_available"] == 1
    assert values["depth_imbalance"] == 0.5
    assert values["microprice_displacement_bps"] > 0
    assert values["mark_index_basis_bps"] == pytest.approx(100)
    assert values["liquidation_net_qty_5s"] == -2


def test_snapshot_cannot_move_backwards() -> None:
    engine = CausalMicrostructureFeatureEngine(windows_seconds=(1,))
    engine.consume(make_trade(1000, 100, 1, False))
    with pytest.raises(ValueError, match="before"):
        engine.snapshot(int(NOW.timestamp() * 1000))

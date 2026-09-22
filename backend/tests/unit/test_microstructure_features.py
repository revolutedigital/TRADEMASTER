"""Offline vectorization must match the causal online feature engine."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.research.microstructure_features import (
    CausalMicrostructureFeatureEngine,
    feature_vector_for_side,
    materialize_book_features,
    materialize_liquidation_features,
    materialize_mark_features,
    materialize_spot_perp_features,
    materialize_trade_flow_features,
    side_aware_feature_values,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_trade(
    offset_ms: int,
    price: float,
    quantity: float,
    maker: bool,
    *,
    product: str = "usdm_perpetual",
):
    return MicrostructureEvent(
        product=product,
        symbol="BTCUSDT",
        event_type=MarketEventType.TRADE,
        event_time=NOW + timedelta(milliseconds=offset_ms),
        price=price,
        quantity=quantity,
        quote_quantity=price * quantity,
        is_buyer_maker=maker,
    )


def make_book(
    offset_ms: int,
    bid_quantity: float,
    ask_quantity: float,
    bid_price: float = 99.0,
    ask_price: float = 101.0,
):
    return MicrostructureEvent(
        product="usdm_perpetual",
        symbol="BTCUSDT",
        event_type=MarketEventType.DEPTH,
        event_time=NOW + timedelta(milliseconds=offset_ms),
        bid_price=bid_price,
        bid_quantity=bid_quantity,
        ask_price=ask_price,
        ask_quantity=ask_quantity,
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


def test_spot_perp_features_are_causal_and_compute_gaps() -> None:
    base_ms = int(NOW.timestamp() * 1000)
    spot_trades = pd.DataFrame(
        {
            "event_time_ms": [base_ms, base_ms + 1_000],
            "sequence_id": [1, 2],
            "price": [100.0, 102.0],
            "quantity": [1.0, 1.0],
            "is_buyer_maker": [False, False],
        }
    )
    perp_trades = pd.DataFrame(
        {
            "event_time_ms": [base_ms, base_ms + 1_000],
            "sequence_id": [10, 11],
            "price": [100.0, 101.0],
            "quantity": [1.0, 1.0],
            "is_buyer_maker": [False, True],
        }
    )

    features = materialize_spot_perp_features(
        spot_trades,
        perp_trades,
        np.array([base_ms + 500, base_ms + 1_000], dtype=np.int64),
        windows_seconds=(2,),
    )

    first = features.iloc[0]
    second = features.iloc[1]
    assert first["spot_trade_count_2s"] == 1
    assert first["spot_return_2s_bps"] == 0
    assert first["spot_perp_basis_bps"] == 0
    assert second["spot_available"] == 1
    assert second["spot_update_age_ms"] == 0
    assert second["spot_trade_count_2s"] == 2
    assert second["spot_return_2s_bps"] == pytest.approx(np.log(102 / 100) * 10_000)
    assert second["spot_perp_return_gap_2s_bps"] == pytest.approx(
        (np.log(102 / 100) - np.log(101 / 100)) * 10_000
    )
    assert second["spot_perp_flow_gap_2s"] > 1
    assert second["spot_perp_basis_bps"] == pytest.approx((101 / 102 - 1) * 10_000)

    engine = CausalMicrostructureFeatureEngine(windows_seconds=(2,))
    engine.consume(make_trade(0, 100, 1, False, product="spot"))
    engine.consume(make_trade(0, 100, 1, False))
    first_online = engine.snapshot(base_ms + 500).values
    engine.consume(make_trade(1_000, 102, 1, False, product="spot"))
    engine.consume(make_trade(1_000, 101, 1, True))
    second_online = engine.snapshot(base_ms + 1_000).values

    for index, online in enumerate((first_online, second_online)):
        offline_row = features.iloc[index]
        for key in (
            "spot_available",
            "spot_update_age_ms",
            "spot_trade_count_2s",
            "spot_quote_volume_2s",
            "spot_flow_imbalance_2s",
            "spot_return_2s_bps",
            "spot_perp_return_gap_2s_bps",
            "spot_perp_flow_gap_2s",
            "spot_perp_quote_volume_ratio_2s",
            "spot_perp_basis_bps",
        ):
            assert online[key] == pytest.approx(offline_row[key])


def test_offline_book_features_are_causal_and_match_online_snapshot() -> None:
    base_ms = int(NOW.timestamp() * 1000)
    decision_times = np.array([base_ms + 1000, base_ms + 1999, base_ms + 2000], dtype=np.int64)
    offline = materialize_book_features(
        pd.DataFrame(
            {
                "event_time_ms": np.array([base_ms, base_ms + 2000], dtype=np.int64),
                "sequence_id": [1, 2],
                "bid_price": [99.0, 100.0],
                "bid_quantity": [3.0, 1.0],
                "ask_price": [101.0, 102.0],
                "ask_quantity": [1.0, 3.0],
            }
        ),
        decision_times,
    )
    engine = CausalMicrostructureFeatureEngine(windows_seconds=(1,))
    engine.consume(make_book(0, bid_quantity=3, ask_quantity=1))
    online = engine.snapshot(base_ms + 1000).values

    assert offline.iloc[0]["book_available"] == 1
    assert offline.iloc[0]["book_update_age_ms"] == 1000
    assert offline.iloc[0]["spread_bps"] == pytest.approx(online["spread_bps"])
    assert offline.iloc[0]["depth_imbalance"] == pytest.approx(online["depth_imbalance"])
    assert offline.iloc[0]["microprice_displacement_bps"] == pytest.approx(
        online["microprice_displacement_bps"]
    )
    assert offline.iloc[1]["depth_imbalance"] == pytest.approx(0.5)
    assert offline.iloc[2]["depth_imbalance"] == pytest.approx(-0.5)


def test_book_window_dynamics_are_causal_and_match_online_snapshot() -> None:
    base_ms = int(NOW.timestamp() * 1000)
    book_events = pd.DataFrame(
        {
            "event_time_ms": np.array([base_ms, base_ms + 500, base_ms + 900], dtype=np.int64),
            "sequence_id": [1, 2, 3],
            "bid_price": [99.0, 99.0, 99.0],
            "bid_quantity": [5.0, 8.0, 6.0],
            "ask_price": [101.0, 101.0, 101.0],
            "ask_quantity": [5.0, 3.0, 7.0],
        }
    )
    decision_times = np.array([base_ms + 750, base_ms + 1000], dtype=np.int64)
    offline = materialize_book_features(book_events, decision_times, windows_seconds=(1,))

    engine = CausalMicrostructureFeatureEngine(windows_seconds=(1,))
    engine.consume(make_book(0, bid_quantity=5, ask_quantity=5))
    engine.consume(make_book(500, bid_quantity=8, ask_quantity=3))
    first_online = engine.snapshot(base_ms + 750).values
    engine.consume(make_book(900, bid_quantity=6, ask_quantity=7))
    second_online = engine.snapshot(base_ms + 1000).values

    for key in (
        "book_event_count_1s",
        "book_bid_replenishment_qty_1s",
        "book_ask_replenishment_qty_1s",
        "book_bid_liquidity_removed_qty_1s",
        "book_ask_liquidity_removed_qty_1s",
        "book_pressure_imbalance_1s",
        "book_depth_imbalance_change_1s",
        "book_microprice_displacement_change_bps_1s",
    ):
        assert offline.iloc[0][key] == pytest.approx(first_online[key])
        assert offline.iloc[1][key] == pytest.approx(second_online[key])

    assert offline.iloc[0]["book_event_count_1s"] == 2
    assert offline.iloc[0]["book_pressure_imbalance_1s"] == pytest.approx(1.0)
    assert offline.iloc[1]["book_event_count_1s"] == 3
    assert offline.iloc[1]["book_pressure_imbalance_1s"] == pytest.approx(-1 / 11)


def test_book_window_spread_recovery_uses_only_quotes_inside_window() -> None:
    base_ms = int(NOW.timestamp() * 1000)
    offline = materialize_book_features(
        pd.DataFrame(
            {
                "event_time_ms": np.array(
                    [base_ms - 1_500, base_ms - 800, base_ms - 100],
                    dtype=np.int64,
                ),
                "sequence_id": [1, 2, 3],
                "bid_price": [99.0, 98.0, 99.5],
                "bid_quantity": [5.0, 5.0, 5.0],
                "ask_price": [101.0, 102.0, 100.5],
                "ask_quantity": [5.0, 5.0, 5.0],
            }
        ),
        np.array([base_ms], dtype=np.int64),
        windows_seconds=(1,),
    ).iloc[0]

    assert offline["book_event_count_1s"] == 2
    assert offline["book_spread_recovery_bps_1s"] > 0
    assert offline["book_spread_widening_bps_1s"] == 0


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
    assert values["mark_available"] == 1
    assert values["mark_update_age_ms"] == 0
    assert values["mark_index_basis_bps"] == pytest.approx(100)
    assert values["liquidation_count_5s"] == 1
    assert values["liquidation_net_qty_5s"] == -2
    assert values["liquidation_abs_qty_5s"] == 2


def test_mark_and_liquidation_features_are_causal_and_match_online_snapshot() -> None:
    base_ms = int(NOW.timestamp() * 1000)
    decision_times = np.array([base_ms + 500, base_ms + 1_500], dtype=np.int64)
    offline_mark = materialize_mark_features(
        pd.DataFrame(
            {
                "event_time_ms": np.array([base_ms, base_ms + 1_000], dtype=np.int64),
                "mark_price": [101.0, 99.0],
                "index_price": [100.0, 100.0],
                "funding_rate": [0.0001, -0.0002],
            }
        ),
        decision_times,
    )
    offline_liquidations = materialize_liquidation_features(
        pd.DataFrame(
            {
                "event_time_ms": np.array([base_ms + 400, base_ms + 1_200], dtype=np.int64),
                "sequence_id": [1, 2],
                "price": [100.0, 110.0],
                "quantity": [2.0, 3.0],
                "side": ["SELL", "BUY"],
            }
        ),
        decision_times,
        windows_seconds=(1,),
    )

    engine = CausalMicrostructureFeatureEngine(windows_seconds=(1,))
    engine.consume(
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.MARK_PRICE,
            event_time=NOW,
            price=101.0,
            payload={"index_price": 100.0, "funding_rate": 0.0001},
        )
    )
    engine.consume(
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.LIQUIDATION,
            event_time=NOW + timedelta(milliseconds=400),
            price=100.0,
            quantity=2.0,
            side="SELL",
        )
    )
    first_online = engine.snapshot(base_ms + 500).values
    engine.consume(
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.MARK_PRICE,
            event_time=NOW + timedelta(milliseconds=1_000),
            price=99.0,
            payload={"index_price": 100.0, "funding_rate": -0.0002},
        )
    )
    engine.consume(
        MicrostructureEvent(
            product="usdm_perpetual",
            symbol="BTCUSDT",
            event_type=MarketEventType.LIQUIDATION,
            event_time=NOW + timedelta(milliseconds=1_200),
            price=110.0,
            quantity=3.0,
            side="BUY",
        )
    )
    second_online = engine.snapshot(base_ms + 1_500).values

    for index, online in enumerate((first_online, second_online)):
        assert offline_mark.iloc[index]["mark_available"] == pytest.approx(
            online["mark_available"]
        )
        assert offline_mark.iloc[index]["mark_update_age_ms"] == pytest.approx(
            online["mark_update_age_ms"]
        )
        assert offline_mark.iloc[index]["mark_index_basis_bps"] == pytest.approx(
            online["mark_index_basis_bps"]
        )
        assert offline_mark.iloc[index]["funding_rate"] == pytest.approx(online["funding_rate"])
        assert offline_liquidations.iloc[index]["liquidation_count_1s"] == pytest.approx(
            online["liquidation_count_1s"]
        )
        assert offline_liquidations.iloc[index]["liquidation_net_qty_1s"] == pytest.approx(
            online["liquidation_net_qty_1s"]
        )
        assert offline_liquidations.iloc[index]["liquidation_net_notional_1s"] == pytest.approx(
            online["liquidation_net_notional_1s"]
        )


def test_side_aware_feature_values_match_offline_directional_convention() -> None:
    base_values = {
        "flow_imbalance_1s": 0.25,
        "return_1s_bps": 3.0,
        "depth_imbalance": 0.4,
        "microprice_displacement_bps": 1.5,
        "mark_index_basis_bps": -2.0,
        "funding_rate": 0.0002,
        "book_pressure_imbalance_1s": 0.7,
        "liquidation_net_qty_1s": -4.0,
        "spot_perp_basis_bps": 5.0,
        "spot_return_1s_bps": -1.0,
        "spot_perp_flow_gap_1s": 0.2,
    }

    buy = side_aware_feature_values(base_values, "BUY")
    sell = side_aware_feature_values(base_values, "SELL")
    selected = feature_vector_for_side(
        base_values,
        "SELL",
        (
            "side_sign",
            "directed_flow_imbalance_1s",
            "directed_return_1s_bps",
            "directed_depth_imbalance",
            "directed_funding_rate",
            "directed_book_pressure_imbalance_1s",
            "directed_liquidation_net_qty_1s",
            "directed_spot_perp_basis_bps",
            "directed_spot_return_1s_bps",
            "directed_spot_perp_flow_gap_1s",
        ),
    )

    assert buy["side_sign"] == 1.0
    assert sell["side_sign"] == -1.0
    assert buy["directed_flow_imbalance_1s"] == pytest.approx(0.25)
    assert sell["directed_flow_imbalance_1s"] == pytest.approx(-0.25)
    assert buy["directed_funding_rate"] == pytest.approx(-0.0002)
    assert sell["directed_funding_rate"] == pytest.approx(0.0002)
    assert selected["directed_return_1s_bps"] == pytest.approx(-3.0)
    assert selected["directed_depth_imbalance"] == pytest.approx(-0.4)
    assert selected["directed_book_pressure_imbalance_1s"] == pytest.approx(-0.7)
    assert selected["directed_liquidation_net_qty_1s"] == pytest.approx(4.0)
    assert selected["directed_spot_perp_basis_bps"] == pytest.approx(-5.0)
    assert selected["directed_spot_return_1s_bps"] == pytest.approx(1.0)
    assert selected["directed_spot_perp_flow_gap_1s"] == pytest.approx(-0.2)


def test_feature_vector_for_side_rejects_missing_model_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        feature_vector_for_side({"flow_imbalance_1s": 0.1}, "BUY", ("quote_volume_1s",))


def test_snapshot_cannot_move_backwards() -> None:
    engine = CausalMicrostructureFeatureEngine(windows_seconds=(1,))
    engine.consume(make_trade(1000, 100, 1, False))
    with pytest.raises(ValueError, match="before"):
        engine.snapshot(int(NOW.timestamp() * 1000))

"""Online event-driven shadow scoring stays research-only and causal."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.research.online_shadow_policy import score_online_frozen_top_p_shadow_events
from app.services.research.top_p_model import freeze_top_p_policy


BASE_TIME = datetime(2026, 1, 8, tzinfo=UTC)


def test_online_frozen_top_p_shadow_scoring_derives_side_aware_features() -> None:
    decision_time_ms = int((BASE_TIME + timedelta(seconds=1)).timestamp() * 1000)

    selection = score_online_frozen_top_p_shadow_events(
        (
            _trade(0, 100.0, 1.0, False, product="spot"),
            _trade(0, 100.0, 1.0, False),
            _trade(900, 101.0, 2.0, False),
            _trade(950, 90.0, 10.0, True, product="spot"),
        ),
        artifact=_artifact(),
        decision_times_ms=(decision_time_ms,),
        include_non_entries=True,
        windows_seconds=(1,),
    )

    decisions = {decision.side: decision for decision in selection.decisions}
    buy = decisions["BUY"]
    sell = decisions["SELL"]
    assert selection.scored_rows == 2
    assert selection.selected_count == 2
    assert 0 <= buy.probability <= 1
    assert 0 <= sell.probability <= 1
    assert buy.feature_vector["directed_flow_imbalance_1s"] > 0
    assert sell.feature_vector["directed_flow_imbalance_1s"] < 0
    assert buy.feature_vector["quote_volume_1s"] == 302
    assert buy.feature_vector["spot_quote_volume_1s"] == 1_000
    assert buy.model_sha256 == selection.model_sha256
    assert selection.order_submission_allowed is False
    assert selection.execution_authorization == "none"


def _trade(
    offset_ms: int,
    price: float,
    quantity: float,
    maker: bool,
    *,
    product: str = "usdm_perpetual",
) -> MicrostructureEvent:
    return MicrostructureEvent(
        product=product,
        symbol="BTCUSDT",
        event_type=MarketEventType.TRADE,
        event_time=BASE_TIME + timedelta(milliseconds=offset_ms),
        price=price,
        quantity=quantity,
        quote_quantity=price * quantity,
        is_buyer_maker=maker,
    )


def _artifact() -> dict[str, object]:
    return freeze_top_p_policy(
        _model_frame(),
        horizon_seconds=120,
        feature_set="flow_price_aux_session",
        tail_fraction=0.10,
        calibration_date="2026-01-06",
        dataset_manifest_sha256="d" * 64,
        embargo_seconds=300,
    ).to_dict()


def _model_frame() -> pd.DataFrame:
    random = np.random.default_rng(456)
    rows = []
    for day in range(7):
        day_start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
        for sample in range(60):
            signal = random.normal()
            rows.append(
                {
                    "decision_time_ms": int(
                        (day_start + pd.Timedelta(minutes=sample * 10)).timestamp() * 1000
                    ),
                    "horizon_seconds": 120,
                    "target": int(signal + random.normal(scale=0.3) > 0),
                    "side": "BUY",
                    "flow_imbalance_1s": signal,
                    "directed_flow_imbalance_1s": signal,
                    "trade_count_1s": 10 + abs(signal),
                    "quote_volume_1s": 100 + abs(signal) * 20,
                    "return_1s_bps": signal,
                    "directed_return_1s_bps": signal,
                    "realized_vol_1s_bps": abs(signal),
                    "mean_interarrival_ms_1s": 20,
                    "liquidation_count_1s": 0,
                    "liquidation_abs_qty_1s": 0,
                    "liquidation_abs_notional_1s": 0,
                    "mark_available": 0,
                    "mark_update_age_ms": 0,
                    "mark_index_basis_bps": 0,
                    "directed_mark_index_basis_bps": 0,
                    "funding_rate": 0,
                    "directed_funding_rate": 0,
                    "spot_available": 1,
                    "spot_update_age_ms": 0,
                    "spot_trade_count_1s": 10 + abs(signal),
                    "spot_quote_volume_1s": 80 + abs(signal) * 10,
                    "spot_flow_imbalance_1s": signal,
                    "directed_spot_flow_imbalance_1s": signal,
                    "spot_return_1s_bps": signal,
                    "directed_spot_return_1s_bps": signal,
                    "spot_realized_vol_1s_bps": abs(signal),
                    "spot_mean_interarrival_ms_1s": 20,
                    "spot_perp_basis_bps": signal,
                    "directed_spot_perp_basis_bps": signal,
                    "spot_perp_return_gap_1s_bps": signal,
                    "directed_spot_perp_return_gap_1s_bps": signal,
                    "spot_perp_flow_gap_1s": signal,
                    "directed_spot_perp_flow_gap_1s": signal,
                    "spot_perp_quote_volume_ratio_1s": 1,
                    "side_sign": 1,
                    "hour_sin": 0,
                    "hour_cos": 1,
                }
            )
    return pd.DataFrame(rows)

"""Causality and cost-hurdle tests for the executed-flow pilot."""

import numpy as np
import pandas as pd

from scripts.research.fx_fast11_flow import (
    build_features,
    build_horizon_data,
    utc_clustered_lower_bound,
)


def _market(rows: int = 400) -> pd.DataFrame:
    index = pd.date_range("2026-09-13", periods=rows, freq="s", tz="UTC")
    close = 100 + np.arange(rows) * 0.001
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": np.full(rows, 2.0),
            "quote_volume": np.full(rows, 200.0),
            "trades": np.full(rows, 4.0),
            "taker_buy_volume": np.full(rows, 1.2),
            "taker_buy_quote_volume": np.full(rows, 120.0),
        },
        index=index,
    )


def test_features_do_not_change_when_future_market_changes() -> None:
    market = _market()
    original = build_features(market)
    changed = market.copy()
    changed.loc[changed.index[350]:, "close"] *= 2
    altered = build_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:350], altered.iloc[:350])


def test_horizon_targets_apply_declared_cost_hurdles() -> None:
    market = _market()
    features = build_features(market)
    data = build_horizon_data(market, features, 5)
    assert len(data.base_r) > 0
    assert np.allclose(data.base_r - data.stress_r, 15.0)


def test_horizon_rejects_paths_with_missing_seconds() -> None:
    market = _market()
    market.loc[market.index[390], :] = np.nan
    features = build_features(market)
    data = build_horizon_data(market, features, 5)
    affected = (data.index >= market.index[385]) & (data.index < market.index[390])
    assert not affected.any()


def test_utc_clustered_bound_requires_both_days() -> None:
    index = pd.DatetimeIndex(["2026-09-18T12:00:00Z", "2026-09-19T12:00:00Z"])
    lower, days = utc_clustered_lower_bound(np.asarray([1.0, 1.0]), index, 0.05)
    assert days == 2
    assert lower == 1.0

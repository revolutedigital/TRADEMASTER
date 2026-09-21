"""Path, cost, and conservative intrabar tests for managed crypto outcomes."""

import numpy as np
import pandas as pd

from scripts.research.fx_fast12_managed import Management, simulate_managed_batch


def _market(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2026-09-20", periods=len(prices), freq="s", tz="UTC")
    return pd.DataFrame(prices, columns=["open", "high", "low", "close"], index=index)


def test_initial_stop_is_charged_roundtrip_cost() -> None:
    market = _market([(100, 100, 100, 100), (100, 100.01, 99.8, 99.9), (100, 100, 100, 100)])
    result = simulate_managed_batch(
        market,
        market.index[:1],
        np.asarray([1]),
        max_hold_seconds=2,
        management=Management(10, 5, 3),
    )
    assert np.allclose(result[0], [-15, -30])


def test_breakeven_is_net_of_each_cost_scenario() -> None:
    market = _market(
        [
            (100, 100, 100, 100),
            (100, 100.25, 100, 100.2),
            (100.2, 100.21, 99.9, 100),
        ]
    )
    result = simulate_managed_batch(
        market,
        market.index[:1],
        np.asarray([1]),
        max_hold_seconds=2,
        management=Management(40, 5, 50),
    )
    assert np.allclose(result[0], [0, 0], atol=1e-9)


def test_trailing_captures_runner_for_long_and_short() -> None:
    long_market = _market(
        [(100, 100, 100, 100), (100, 100.3, 100, 100.25), (100.25, 100.26, 100.1, 100.15)]
    )
    short_market = _market(
        [(100, 100, 100, 100), (100, 100, 99.7, 99.75), (99.75, 99.9, 99.7, 99.85)]
    )
    management = Management(40, 5, 10)
    long = simulate_managed_batch(
        long_market, long_market.index[:1], np.asarray([1]), max_hold_seconds=2, management=management
    )
    short = simulate_managed_batch(
        short_market,
        short_market.index[:1],
        np.asarray([-1]),
        max_hold_seconds=2,
        management=management,
    )
    assert long[0, 0] > 0
    assert short[0, 0] > 0
    assert long[0, 1] >= 0
    assert short[0, 1] >= 0


def test_same_bar_uses_pessimistic_initial_stop_before_activation() -> None:
    market = _market([(100, 100, 100, 100), (100, 100.3, 99.8, 100.2), (100.2, 100.2, 100.2, 100.2)])
    result = simulate_managed_batch(
        market,
        market.index[:1],
        np.asarray([1]),
        max_hold_seconds=2,
        management=Management(10, 5, 3),
    )
    assert np.allclose(result[0], [-15, -30])

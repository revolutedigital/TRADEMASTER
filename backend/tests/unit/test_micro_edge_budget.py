"""Economic edge budget is explicit about costs and oracle limitations."""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.cost_model import RoundTripCostModel, stressed_cost_model
from app.services.backtest.micro_edge_budget import compute_edge_budget


def _market(prices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": prices,
            "high": prices * 1.0001,
            "low": prices * 0.9999,
        },
        index=pd.date_range("2026-01-01", periods=len(prices), freq="s", tz="UTC"),
    )


def test_round_trip_cost_and_stress_are_explicit() -> None:
    expected = RoundTripCostModel(
        fee_bps_per_side=4,
        slippage_bps_per_side=1,
        spread_bps=2,
        funding_bps=0.5,
    )
    stress = stressed_cost_model(expected)

    assert expected.round_trip_bps == 12.5
    assert expected.net_bps(20) == 7.5
    assert stress.round_trip_bps == 25


def test_flat_market_fails_the_feasibility_gate_after_costs() -> None:
    market = _market(np.full(120, 100.0))
    expected = RoundTripCostModel(2, 1)
    stress = stressed_cost_model(expected)

    report = compute_edge_budget(
        market,
        horizons_seconds=(5, 30),
        expected_cost=expected,
        stress_cost=stress,
    )

    assert all(not horizon.feasibility_gate for horizon in report.horizons)
    assert all(horizon.stress_oracle_p99_net_bps < 0 for horizon in report.horizons)
    assert "cannot approve" in report.oracle_warning


def test_large_paths_pass_only_the_upper_bound_gate() -> None:
    prices = 100 * np.exp(np.arange(400) * 0.0002)
    expected = RoundTripCostModel(2, 1)
    stress = RoundTripCostModel(0, 10)

    report = compute_edge_budget(
        _market(prices),
        horizons_seconds=(30,),
        expected_cost=expected,
        stress_cost=stress,
    )
    budget = report.horizons[0]

    assert budget.observations > 50
    assert budget.oracle_mfe_p99_bps > budget.stress_round_trip_bps
    assert budget.feasibility_gate is True


def test_stress_cost_cannot_be_below_expected_cost() -> None:
    market = _market(np.linspace(100, 101, 100))
    with pytest.raises(ValueError, match="Stress cost"):
        compute_edge_budget(
            market,
            horizons_seconds=(5,),
            expected_cost=RoundTripCostModel(0, 10),
            stress_cost=RoundTripCostModel(0, 1),
        )

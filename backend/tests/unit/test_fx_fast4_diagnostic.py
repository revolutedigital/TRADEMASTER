"""Policy mechanics and gates for the quote-event diagnostic."""

import numpy as np
import pandas as pd

from scripts.research import fx_fast4_diagnostic as diagnostic


def predictions() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-07-01 00:00:00", tz="UTC"),
            pd.Timestamp("2021-07-01 00:00:00", tz="UTC"),
            pd.Timestamp("2021-07-01 00:00:20", tz="UTC"),
            pd.Timestamp("2021-07-01 00:02:01", tz="UTC"),
        ]
    )
    return pd.DataFrame(
        {
            "pair": ["EURUSD"] * 4,
            "side": [1, -1, 1, 1],
            "expected_r": [0.04, 0.09, 0.12, 0.10],
            "base_r": [0.1, 0.2, 0.3, 0.4],
            "stress_r": [0.0, 0.1, 0.2, 0.3],
        },
        index=index,
    )


def test_policy_chooses_stronger_side_and_enforces_holding_time() -> None:
    chosen = diagnostic.apply_policy(predictions(), horizon=120, ev_threshold=0.03)
    assert list(chosen["side"]) == [-1, 1]
    assert list(chosen.index) == [
        pd.Timestamp("2021-07-01 00:00:00", tz="UTC"),
        pd.Timestamp("2021-07-01 00:02:01", tz="UTC"),
    ]


def test_model_matrix_flips_directional_features_only() -> None:
    frame = pd.DataFrame({"mid_return_pips_16": [0.5], "spread_pips": [0.8]})
    matrix = diagnostic.model_matrix(
        frame, ["mid_return_pips_16", "spread_pips"], "EURUSD", -1
    )
    assert matrix[0, 0] == -0.5
    assert matrix[0, 1] == 0.8
    assert matrix[0, -1] == -1


def test_core_gate_requires_positive_stress_and_diversification() -> None:
    passing = diagnostic.PolicyMetrics(1_500, 0.1, 0.05, 0.01, 7, 2 / 3, 0.3)
    assert diagnostic.core_selection_gate(passing)
    assert not diagnostic.core_selection_gate(
        diagnostic.PolicyMetrics(1_500, 0.1, -0.01, 0.01, 7, 2 / 3, 0.3)
    )
    assert not diagnostic.core_selection_gate(
        diagnostic.PolicyMetrics(1_500, 0.1, 0.05, 0.01, 6, 2 / 3, 0.3)
    )


def test_stationary_bootstrap_is_deterministic() -> None:
    index = pd.date_range("2021-07-01", periods=400, freq="h", tz="UTC")
    trades = pd.DataFrame(
        {"base_r": np.linspace(-0.1, 0.2, len(index)), "pair": "EURUSD"}, index=index
    )
    first = diagnostic.stationary_bootstrap_lower_bound(trades, samples=100)
    second = diagnostic.stationary_bootstrap_lower_bound(trades, samples=100)
    assert first == second

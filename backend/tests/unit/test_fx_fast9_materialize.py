"""Deterministic sampling and protected-boundary tests for round-9 payoffs."""

import pandas as pd
import pytest

from scripts.research.fx_fast9_materialize import Q1_END, sampled_positions


def test_training_sampling_uses_fixed_pair_offset_and_stride() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    assert sampled_positions(index, "EURUSD", 2020).tolist() == [0, 4, 8]
    assert sampled_positions(index, "GBPUSD", 2020).tolist() == [1, 5, 9]


def test_2021_materialization_stops_before_q2() -> None:
    index = pd.DatetimeIndex(
        [Q1_END - pd.Timedelta(seconds=1), Q1_END, Q1_END + pd.Timedelta(seconds=1)]
    )
    assert sampled_positions(index, "EURUSD", 2021).tolist() == [0]


def test_protected_year_is_rejected() -> None:
    index = pd.date_range("2022-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="restricted"):
        sampled_positions(index, "EURUSD", 2022)

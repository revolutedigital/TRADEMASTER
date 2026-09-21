"""Causal global Top-P selection and portfolio scheduling."""

import pandas as pd
import pytest

from scripts.research.fx_fast6_policy import (
    as_candidate_multiindex,
    causal_daily_cutoffs,
    enforce_global_portfolio,
    global_policy_metrics,
    qualifying_candidates,
)


def test_candidate_multiindex_keeps_pair_only_as_index_level() -> None:
    timestamp = pd.Timestamp("2021-04-01T10:00:00Z")
    frame = pd.DataFrame(
        {"pair": ["EURUSD"], "trusted_score": [0.4]},
        index=pd.DatetimeIndex([timestamp], name="decision_time"),
    )
    indexed = as_candidate_multiindex(frame)
    assert indexed.index.tolist() == [(timestamp, "EURUSD")]
    assert "pair" not in indexed.columns


def test_daily_cutoff_uses_only_prior_fx_days() -> None:
    index = pd.DatetimeIndex(
        ["2021-03-30T12:00:00Z", "2021-03-31T12:00:00Z", "2021-04-01T12:00:00Z"]
    )
    reference = pd.DataFrame(
        {"trusted_score": [0.2, 0.4, 0.99], "pair": ["EURUSD"] * 3}, index=index
    )
    start = pd.Timestamp("2021-04-01T06:00:00Z")
    end = pd.Timestamp("2021-04-01T18:00:00Z")
    cutoffs = causal_daily_cutoffs(reference, 0.5, 0.0, start=start, end=end)
    day = next(iter(cutoffs))
    assert cutoffs[day] == pytest.approx(0.3)
    qualified = qualifying_candidates(reference, cutoffs, start=start, end=end)
    assert qualified["trusted_score"].tolist() == [0.99]


def test_absolute_floor_can_block_daily_top_candidate() -> None:
    index = pd.DatetimeIndex(["2021-03-31T12:00:00Z", "2021-04-01T12:00:00Z"])
    reference = pd.DataFrame({"trusted_score": [0.2, 0.3], "pair": ["EURUSD"] * 2}, index=index)
    start = pd.Timestamp("2021-04-01T06:00:00Z")
    end = pd.Timestamp("2021-04-01T18:00:00Z")
    cutoffs = causal_daily_cutoffs(reference, 0.5, 0.5, start=start, end=end)
    assert qualifying_candidates(reference, cutoffs, start=start, end=end).empty


def test_global_scheduler_limits_slots_and_one_position_per_pair() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2021-04-01T10:00:00Z"), "EURUSD"),
            (pd.Timestamp("2021-04-01T10:00:00Z"), "GBPUSD"),
            (pd.Timestamp("2021-04-01T10:00:00Z"), "USDJPY"),
            (pd.Timestamp("2021-04-01T10:00:00Z"), "AUDUSD"),
            (pd.Timestamp("2021-04-01T10:00:30Z"), "EURUSD"),
            (pd.Timestamp("2021-04-01T10:01:01Z"), "AUDUSD"),
        ],
        names=("decision_time", "pair"),
    )
    frame = pd.DataFrame(
        {
            "base_r": [0.1] * 6,
            "stress_r": [0.05] * 6,
            "holding_seconds_base": [60.0] * 6,
            "holding_seconds_stress": [60.0] * 6,
            "trusted_score": [0.9, 0.8, 0.7, 0.6, 0.95, 0.5],
        },
        index=index,
    )
    selected = enforce_global_portfolio(frame, max_positions=3)
    assert selected.reset_index()["pair"].tolist() == ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


def test_empty_global_metrics_are_well_formed() -> None:
    metrics = global_policy_metrics(pd.DataFrame())
    assert metrics.trades == 0
    assert metrics.participating_pairs == 0

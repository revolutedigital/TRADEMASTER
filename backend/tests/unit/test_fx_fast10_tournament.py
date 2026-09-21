"""Tests for hypothesis generation and multiple-testing guards."""

import numpy as np
import pandas as pd

from scripts.research.fx_fast10_tournament import (
    TournamentData,
    audit_finalists,
    clustered_lower_bound,
    enumerate_tail_candidates,
)


def test_tail_generation_is_fixed_per_feature() -> None:
    index = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
    data = TournamentData(
        matrix=np.column_stack((np.arange(100), np.arange(100)[::-1])).astype(np.float32),
        names=("first", "second"),
        index=index,
        base_r=np.zeros(100),
        stress_r=np.zeros(100),
    )
    candidates = enumerate_tail_candidates(data, np.ones(100, dtype=bool))
    assert len(candidates) == 8
    assert {candidate.family for candidate in candidates} == {"tail"}


def test_clustered_lower_bound_respects_day_clusters() -> None:
    index = pd.date_range("2021-03-15", periods=240, freq="h", tz="UTC")
    lower, days = clustered_lower_bound(np.full(240, 0.2), index, 0.05)
    assert days >= 8
    assert lower == 0.2


def test_empty_finalist_set_does_not_open_a_gate() -> None:
    index = pd.date_range("2021-03-15", periods=100, freq="h", tz="UTC")
    data = TournamentData(
        matrix=np.zeros((100, 1), dtype=np.float32),
        names=("only",),
        index=index,
        base_r=np.full(100, -0.1),
        stress_r=np.full(100, -0.2),
    )
    masks = {"audit": np.ones(100, dtype=bool)}
    assert audit_finalists([], masks, data) == []

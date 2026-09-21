"""Round-5 Q2 policy selection and non-overlap."""

import pandas as pd

from scripts.research.fx_fast5_policy import (
    PolicyMetrics,
    enforce_common_non_overlap,
    select_candidate_side,
    selection_gate,
)


def test_select_candidate_side_keeps_stronger_direction() -> None:
    timestamp = pd.Timestamp("2021-04-05T10:00:00Z")
    frame = pd.DataFrame(
        {
            "score": [0.60, 0.70],
            "probability_base": [0.65, 0.71],
            "side": [1, -1],
        },
        index=pd.DatetimeIndex([timestamp, timestamp]),
    )
    selected = select_candidate_side(frame, 0.55)
    assert selected["side"].tolist() == [-1]


def test_common_schedule_uses_later_scenario_exit() -> None:
    index = pd.DatetimeIndex(
        ["2021-04-05T10:00:00Z", "2021-04-05T10:01:30Z", "2021-04-05T10:02:01Z"]
    )
    frame = pd.DataFrame(
        {
            "base_r": [0.1, 0.2, 0.3],
            "stress_r": [0.0, 0.1, 0.2],
            "holding_seconds_base": [60.0, 60.0, 60.0],
            "holding_seconds_stress": [120.0, 60.0, 60.0],
        },
        index=index,
    )
    selected = enforce_common_non_overlap(frame)
    assert selected.index.tolist() == [index[0], index[2]]


def test_selection_gate_requires_every_declared_condition() -> None:
    passing = PolicyMetrics(30, 0.1, 0.05, -0.01, 1.06, 2 / 3)
    assert selection_gate(passing)
    assert not selection_gate(PolicyMetrics(29, 0.1, 0.05, 0.1, 1.2, 1.0))
    assert not selection_gate(PolicyMetrics(30, 0.1, -0.01, 0.1, 1.2, 1.0))

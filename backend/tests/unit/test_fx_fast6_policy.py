"""Ranked Q2 policy selection for round 6."""

import pandas as pd
import pytest

from scripts.research.fx_fast6_policy import top_ranked_candidates


def test_top_ranked_uses_exact_ceiling_count_and_stronger_score() -> None:
    index = pd.date_range("2021-04-01", periods=10, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "score": [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0],
            "probability_base": [0.5] * 10,
        },
        index=index,
    )
    selected = top_ranked_candidates(frame, 0.25)
    assert len(selected) == 3
    assert set(selected["score"]) == {0.9, 0.8, 0.7}


def test_top_ranked_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        top_ranked_candidates(pd.DataFrame(), 0.0)

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from scripts.research.run_fast109_fx_fast2_pair_specific_audit import (
    PairConfigAccumulator,
    ScenarioAccumulator,
    fast109_status,
    strict_pair_config_candidates,
    summarize_pair_config_accumulator,
    weak_pair_config_candidates,
)


def _seconds(*stamps: datetime) -> np.ndarray:
    return np.array([stamp.timestamp() for stamp in stamps], dtype=float)


def test_summarize_pair_config_accumulator_tracks_stress_stability() -> None:
    accumulator = PairConfigAccumulator(config="F1a.test", pair="EURUSD", family="F1a")
    accumulator.base.add(
        r_values=np.array([1.0, 0.5, -0.25]),
        entry_seconds=_seconds(
            datetime(2015, 1, 1, tzinfo=UTC),
            datetime(2015, 2, 1, tzinfo=UTC),
            datetime(2016, 1, 1, tzinfo=UTC),
        ),
    )
    accumulator.stress.add(
        r_values=np.array([0.75, -0.25, 1.0]),
        entry_seconds=_seconds(
            datetime(2015, 1, 1, tzinfo=UTC),
            datetime(2015, 2, 1, tzinfo=UTC),
            datetime(2016, 1, 1, tzinfo=UTC),
        ),
    )
    accumulator.add_stop_fit(stop_pips=np.array([8.0, 16.0, 10.0]), limit_pips=12.5)

    row = summarize_pair_config_accumulator(accumulator)

    assert row["base_trade_count"] == 3
    assert row["stress_trade_count"] == 3
    assert row["base_mean_r"] == (1.0 + 0.5 - 0.25) / 3
    assert row["stress_mean_r"] == (0.75 - 0.25 + 1.0) / 3
    assert row["month_count"] == 3
    assert row["year_count"] == 2
    assert row["positive_stress_month_count"] == 2
    assert row["positive_stress_year_count"] == 2
    assert row["stop_fits_min_lot"] == 2 / 3


def test_candidate_filters_separate_weak_from_strict_year_stability() -> None:
    stable = {
        "base_trade_count": 120,
        "stress_trade_count": 120,
        "month_count": 12,
        "year_count": 3,
        "base_mean_r": 0.10,
        "stress_mean_r": 0.04,
        "positive_base_year_count": 3,
        "positive_stress_year_count": 3,
    }
    unstable = {
        **stable,
        "positive_stress_year_count": 2,
    }

    strict = strict_pair_config_candidates(
        [stable, unstable],
        min_trades=100,
        min_months=12,
        min_years=3,
    )
    weak = weak_pair_config_candidates(
        [stable, unstable],
        min_trades=100,
        min_months=12,
        min_years=3,
    )

    assert strict == [stable]
    assert weak == [stable, unstable]


def test_fast109_status_never_approves_strategy() -> None:
    status, reasons = fast109_status(strict_candidate_count=0, weak_candidate_count=0)

    assert status == "FAST109_FX_FAST2_PAIR_SPECIFIC_FAILED_SCREEN"
    assert "not_approved_strategy" in reasons
    assert "no_pair_config_positive_base_and_stress_after_minimum_sample" in reasons

    status, reasons = fast109_status(strict_candidate_count=1, weak_candidate_count=1)

    assert status == "FAST109_FX_FAST2_PAIR_SPECIFIC_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "post_hoc_pair_specific_diagnostic_requires_fresh_preregistration" in reasons

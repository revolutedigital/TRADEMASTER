"""C9 context features: sessions, runs, streaks, dollar exposure, ADX and initial stop/target levels."""

import numpy as np
import pytest

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c9_lab as c9


def test_sessions_follow_new_york_hours() -> None:
    np.testing.assert_array_equal(c9.session(np.array([18, 2, 3, 8, 12, 16])), [0, 0, 1, 2, 3, 3])


def test_run_length_and_streaks_only_look_back() -> None:
    np.testing.assert_array_equal(c9.run_length(np.array([True, True, False, True])), [1, 2, 0, 1])
    # the streak shown for a trade is the run of results before it
    np.testing.assert_array_equal(c9.streaks(np.array([1.0, 2.0, -1.0, -1.0, 3.0])), [0, 1, 2, -1, -2])


def test_dollar_exposure_of_a_long_position() -> None:
    assert (c9.usd_exposure("USDJPY"), c9.usd_exposure("EURUSD"), c9.usd_exposure("EURGBP"),
            c9.usd_exposure("SPXUSD"), c9.usd_exposure("UDXUSD")) == (1, -1, 0, 0, 1)


def test_adx_is_bounded_and_high_in_a_steady_trend() -> None:
    c = np.linspace(100, 160, 120)
    o, h, l = c - 0.2, c + 0.5, c - 0.5
    value = c9.adx(o, h, l, c)
    finite = value[~np.isnan(value)]
    assert finite.size and (finite >= 0).all() and (finite <= 100).all() and finite[-1] > 50


def test_initial_levels_use_the_level_or_the_distance_and_the_target_multiple() -> None:
    n = 2
    level = c3._plan(n, stop_level=np.array([1.09, 1.11]), target_r=1.5, unit=np.ones(n))  # noqa: SLF001
    stop, target = c9.initial_levels(level, np.array([0, 1]), np.array([1.0, -1.0]), np.array([1.10, 1.10]))
    np.testing.assert_allclose(stop, [1.09, 1.11])
    np.testing.assert_allclose(target, [1.115, 1.085])
    distance = c3._plan(n, stop_distance=np.array([0.02, 0.02]), unit=np.ones(n))  # noqa: SLF001
    stop, target = c9.initial_levels(distance, np.array([0, 1]), np.array([1.0, -1.0]), np.array([1.10, 1.10]))
    np.testing.assert_allclose(stop, [1.08, 1.12])
    assert np.isnan(target).all()

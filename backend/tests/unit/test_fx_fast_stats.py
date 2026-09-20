"""The pre-registered criterion: cluster t, max-t bootstrap, placebo p-value and the verdict."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_fast_stats as st


def random_trades(seed: int, days: int = 400, configs: int = 3, per_day: int = 3, effect=None):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(days), per_day * configs)
    config = np.tile(np.repeat(np.arange(configs), per_day), days)
    r = rng.normal(0.0, 1.0, day.size)
    for index, mean in (effect or {}).items():
        r[config == index] += mean
    return day, r, config


def test_daily_tables_match_a_groupby() -> None:
    day, r, config = random_trades(1, days=50)

    sums, counts = st.daily_tables(day, r, config, 50, 3)

    frame = pd.DataFrame({"day": day, "config": config, "r": r})
    expected = frame.groupby(["day", "config"])["r"].agg(["sum", "count"])
    assert np.allclose(sums[expected.index.get_level_values(0), expected.index.get_level_values(1)], expected["sum"])
    assert np.allclose(counts.sum(), len(frame))


def test_cluster_t_matches_a_brute_force_computation() -> None:
    day, r, config = random_trades(2, days=80, configs=2)
    sums, counts = st.daily_tables(day, r, config, 80, 2)

    mean, error, t = st.cluster_t(sums, counts)

    for k in range(2):
        mine = pd.DataFrame({"day": day[config == k], "r": r[config == k]})
        mu = mine["r"].mean()
        residual = mine["r"] - mu
        clustered = np.sqrt((residual.groupby(mine["day"]).sum() ** 2).sum()) / len(mine)
        assert mean[k] == pytest.approx(mu) and error[k] == pytest.approx(clustered)
        assert t[k] == pytest.approx(mu / clustered)


def test_the_resampling_weights_cover_every_draw_and_are_reproducible() -> None:
    weights = st._stationary_weights(200, 50, 0.1, 7)

    assert np.all(weights.sum(axis=1) == 200)
    assert np.all(weights >= 0)
    assert np.array_equal(weights, st._stationary_weights(200, 50, 0.1, 7))
    assert not np.array_equal(weights, st._stationary_weights(200, 50, 0.1, 8))


def test_the_bootstrap_does_not_approve_noise_more_often_than_five_percent_plus_slack() -> None:
    wins = 0
    trials = 120
    for seed in range(trials):
        day, r, config = random_trades(100 + seed, days=250, configs=10, per_day=2)
        sums, counts = st.daily_tables(day, r, config, 250, 10)
        _, p = st.bootstrap_adjusted_p_values(sums, counts, draws=300, seed=seed)
        wins += bool((p <= 0.05).any())
    assert wins / trials <= 0.11  # nominal 5%; a family-wise rate near 5% is the whole point of max-t


def test_a_planted_edge_is_found_and_the_other_configurations_are_not() -> None:
    day, r, config = random_trades(3, days=500, configs=5, per_day=3, effect={0: 0.25})
    sums, counts = st.daily_tables(day, r, config, 500, 5)

    t, p = st.bootstrap_adjusted_p_values(sums, counts, draws=500, seed=1)

    assert t[0] > 4 and p[0] <= 0.01
    assert np.all(p[1:] > 0.05)


def test_a_configuration_without_trades_can_never_pass() -> None:
    day, r, config = random_trades(4, days=200, configs=3)
    sums, counts = st.daily_tables(day, r, config, 200, 3)
    sums[:, 2] = 0.0
    counts[:, 2] = 0.0

    t, p = st.bootstrap_adjusted_p_values(sums, counts, draws=200, seed=2)

    assert not np.isfinite(t[2]) and p[2] == 1.0
    assert st.placebo_p_value(np.array([np.nan, 5.0]), np.zeros(99))[0] == 1.0


def test_the_bootstrap_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        st.bootstrap_adjusted_p_values(np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        st.bootstrap_adjusted_p_values(np.zeros((5, 2)), np.zeros((5, 2)), mean_block=0.5)


def test_the_placebo_p_value_counts_the_null_maxima_that_reach_the_real_t() -> None:
    null_best = np.arange(1, 100) / 50.0  # 0.02 ... 1.98

    p = st.placebo_p_value(np.array([1.0, 5.0, -3.0]), null_best)

    assert p[0] == pytest.approx((1 + (null_best >= 1.0).sum()) / 100)
    assert p[1] == pytest.approx(1 / 100)  # the floor: one over the number of draws plus one
    assert p[2] == 1.0


def test_the_minimum_detectable_effect_is_a_multiple_of_the_standard_error() -> None:
    assert st.minimum_detectable_effect(np.array([0.02]))[0] == pytest.approx(0.068)


GOOD = dict(
    p_bootstrap=0.01, p_placebo=0.02, mean_r=0.05, mean_r_stress=0.01,
    pair_share_positive=0.7, trades=900, mde=0.08,
)


def test_a_configuration_meeting_every_rule_is_approved() -> None:
    verdict = st.judge(**GOOD)

    assert verdict.approved and not verdict.inconclusive and verdict.failed == ()


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        (dict(p_bootstrap=0.06), "key A"),
        (dict(p_placebo=0.2), "key B"),
        (dict(mean_r=0.0), "base costs"),
        (dict(pair_share_positive=0.5), "60% of pairs"),
        (dict(trades=299), "300 trades"),
        (dict(mean_r_stress=-0.01), "stress"),
    ],
)
def test_each_rule_alone_blocks_the_approval(override, fragment) -> None:
    verdict = st.judge(**{**GOOD, **override})

    assert not verdict.approved
    assert any(fragment in reason for reason in verdict.failed)


def test_inconclusive_is_for_thin_samples_and_weak_power_only_when_not_approved() -> None:
    thin = st.judge(**{**GOOD, "trades": 120})
    weak = st.judge(**{**GOOD, "mde": 0.2, "p_bootstrap": 0.3})
    lost = st.judge(**{**GOOD, "mean_r": -0.05})
    powerful_but_approved = st.judge(**{**GOOD, "mde": 0.4})

    assert thin.inconclusive and weak.inconclusive
    assert not lost.inconclusive and not lost.approved
    assert powerful_but_approved.approved and not powerful_but_approved.inconclusive


def test_the_confirmation_uses_its_own_trade_floor_and_no_inconclusive_label() -> None:
    verdict = st.judge(**{**GOOD, "trades": 150}, min_trades=st.MIN_TRADES_CONFIRMATION, allow_inconclusive=False)

    assert verdict.approved
    assert not st.judge(**{**GOOD, "trades": 90}, min_trades=100, allow_inconclusive=False).inconclusive

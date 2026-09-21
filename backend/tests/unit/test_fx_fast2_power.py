"""The power check: the algebra of the injected edge, and that power rises with the edge."""

import numpy as np

from scripts.research import fx_fast2_power as power
from scripts.research import fx_fast_lab as lab
from scripts.research import fx_fast_stats as stats


def fake_result(days: int = 400, keys=("A", "B", "C"), seed: int = 1) -> lab.SampleResult:
    rng = np.random.default_rng(seed)
    counts = rng.integers(2, 9, size=(days, len(keys))).astype(float)
    sums = -0.1 * counts + rng.normal(0.0, 1.0, size=counts.shape) * np.sqrt(counts)  # costs, no edge
    means = {key: {"P1": -0.1, "P2": -0.1, "P3": -0.1} for key in keys}
    return lab.SampleResult(list(keys), sums, counts, means, np.full(len(keys), -0.25), np.ones(len(keys)))


def test_adding_a_constant_to_every_trade_moves_the_mean_and_leaves_the_standard_error() -> None:
    result = fake_result()
    mean, error, t = stats.cluster_t(result.sums, result.counts)
    shifted_sums = result.sums + 0.3 * result.counts

    shifted_mean, shifted_error, shifted_t = stats.cluster_t(shifted_sums, result.counts)

    np.testing.assert_allclose(shifted_mean, mean + 0.3)
    np.testing.assert_allclose(shifted_error, error)
    np.testing.assert_allclose(shifted_t, (mean + 0.3) / error)


def test_the_bootstrap_null_does_not_move_when_a_constant_is_added() -> None:
    result = fake_result()
    before = power.bootstrap_best(result.sums, result.counts, 300, 7)
    after = power.bootstrap_best(result.sums + 0.3 * result.counts, result.counts, 300, 7)

    np.testing.assert_allclose(before, after)


def test_power_rises_with_the_true_edge_and_a_pure_noise_replicate_is_rarely_approved() -> None:
    results = [fake_result(seed=seed) for seed in range(1, 21)]
    null_t = np.stack([stats.cluster_t(r.sums, r.counts)[2] for r in results])
    configs = {key: type("C", (), {"pairs": ("P1", "P2", "P3")}) for key in ("A", "B", "C")}

    table = power.power_table(results, null_t, configs, probes=("A",), net_edges=(0.02, 0.10, 0.5), draws=300)

    row = table.loc["A"]
    assert row["+0.02 R"] <= row["+0.10 R"] <= row["+0.50 R"]
    assert row["+0.50 R"] >= 0.9 and row["+0.02 R"] <= 0.2


def test_the_stress_and_trade_count_rules_still_apply() -> None:
    result = fake_result()
    mean, error, _ = stats.cluster_t(result.sums, result.counts)
    best = power.bootstrap_best(result.sums, result.counts, 300, 7)
    placebo_best = np.full(50, -5.0)  # an easy key B
    pairs = ("P1", "P2", "P3")
    result.stress_mean[0] = -5.0  # the edge would not survive the stress scenario

    assert not power.approves(result, 0, 0.5, error[0], mean[0], best, placebo_best, pairs)
    result.stress_mean[0] = -0.25
    assert power.approves(result, 0, 0.5, error[0], mean[0], best, placebo_best, pairs)

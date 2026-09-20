"""The lab pipeline end to end on small synthetic data: windows, running, costs, placebo, verdicts."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates
from scripts.research import fx_fast_lab as lab
from scripts.research import fx_fast_placebo as placebo
from scripts.research import fx_fast_stats as st
from tests.unit.fx_strategy_checks import synthetic_frame

RATES = ConversionRates({"EURUSD": 1.1, "GBPUSD": 1.3, "AUDUSD": 0.65, "NZDUSD": 0.6,
                         "USDJPY": 150.0, "USDCAD": 1.35, "USDCHF": 0.9})
PAIRS = ("EURUSD", "USDJPY", "AUDUSD", "NZDUSD")


def dataset(weeks: int = 3):
    matrices = {}
    for seed, pair in enumerate(PAIRS, start=1):
        base = 150.0 if pair == "USDJPY" else 1.10
        pip = 0.01 if pair == "USDJPY" else 0.0001
        frame = synthetic_frame(start="2024-05-13", weeks=weeks, bar_seconds=60, seed=seed, sigma_pips=0.8,
                                base=base, pip=pip)
        matrices[pair] = fx.bars_to_matrix(frame)
    start = matrices["EURUSD"][0, fx.BAR_TIME]
    end = matrices["EURUSD"][-1, fx.BAR_TIME] + 60
    windows = {pair: [(start, end)] for pair in (*PAIRS, lab.SYNTHETIC)}
    return matrices, windows, lab.day_universe(matrices, windows)


def chosen(*keys):
    everything = lab.configurations()
    return {key: everything[key] for key in keys}


def test_the_configuration_table_is_the_pre_registered_family() -> None:
    configs = lab.configurations()

    assert set(configs) == registry_keys() - {"F4"}
    assert configs["F2a"].pairs == lab.USD_PAIRS and configs["F5b"].pairs == (lab.SYNTHETIC,)
    assert {k: c.seconds for k, c in configs.items()} == {
        "F1a": 900, "F1b": 900, "F2a": 300, "F2b": 300, "F3a": 300, "F3b": 300,
        "F5a": 3600, "F5b": 3600, "C1": 60, "C2": 300,
    }


def registry_keys():
    from scripts.research import fx_fast_registry

    return set(fx_fast_registry.CONFIGURATIONS)


def test_windows_split_at_an_excluded_month_and_respect_the_sample_bounds() -> None:
    months = ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
    manifest = pd.DataFrame({"pair": "EURUSD", "month": months, "included": [True, True, False, True, True]})

    windows = lab.included_windows(manifest, "EURUSD", ("2023-02", "2023-05"))

    stamp = lambda text: pd.Timestamp(text, tz="UTC").timestamp()  # noqa: E731
    assert windows == [(stamp("2023-02-01"), stamp("2023-03-01")), (stamp("2023-04-01"), stamp("2023-06-01"))]
    assert lab.included_windows(manifest, "EURUSD", ("2024-01", "2024-12")) == []


def test_the_synthetic_cross_only_uses_months_both_legs_have() -> None:
    manifest = pd.DataFrame({
        "pair": ["AUDUSD"] * 3 + ["NZDUSD"] * 3, "month": ["2023-01", "2023-02", "2023-03"] * 2,
        "included": [True, True, True, True, False, True],
    })

    windows = lab.included_windows(manifest, lab.SYNTHETIC, ("2023-01", "2023-03"))

    assert len(windows) == 2  # February is missing on one leg


def test_running_gives_daily_tables_that_add_up_and_stress_costs_more_than_base() -> None:
    matrices, windows, universe = dataset()
    configs = chosen("C2", "F2a", "F5b")

    result = lab.run_sample(matrices, windows, universe, RATES, configs)

    k = result.keys.index("C2")
    assert result.trades[k] > 100
    assert result.counts.shape == (len(universe), 3) and result.counts.sum() == result.trades.sum()
    base_mean = result.sums[:, k].sum() / result.counts[:, k].sum()
    assert result.stress_mean[k] < base_mean < 0  # a control with no edge loses, more so under stress
    assert 0 < result.trades[result.keys.index("F2a")] <= len(universe) * len(lab.USD_PAIRS)
    assert result.trades[result.keys.index("F5b")] == 0  # three weeks cannot fill a 480-bar window
    assert set(result.pair_means["C2"]) <= set(PAIRS)
    assert 0 <= result.fits_min_lot[k] <= 1


def test_running_twice_gives_the_same_answer() -> None:
    matrices, windows, universe = dataset(weeks=2)
    configs = chosen("C2")

    first = lab.run_sample(matrices, windows, universe, RATES, configs)
    second = lab.run_sample(matrices, windows, universe, RATES, configs)

    assert np.array_equal(first.sums, second.sums) and np.array_equal(first.counts, second.counts)


def test_the_placebo_keeps_the_number_of_days_and_changes_the_trades() -> None:
    matrices, windows, universe = dataset(weeks=2)
    configs = chosen("C2")
    first = int(min(m[0, fx.BAR_TIME] for m in matrices.values()) // 60)
    last = int(max(m[-1, fx.BAR_TIME] for m in matrices.values()) // 60)

    real = lab.run_sample(matrices, windows, universe, RATES, configs)
    fake = lab.run_sample(matrices, windows, universe, RATES, configs,
                          coins=(placebo.coin_table(first, last, 3), first))

    assert real.counts.shape == fake.counts.shape
    assert not np.array_equal(real.sums, fake.sums)
    assert np.isfinite(st.cluster_t(fake.sums, fake.counts)[2]).all()


def test_the_evaluation_rejects_noise_and_reports_every_configuration() -> None:
    matrices, windows, universe = dataset()
    configs = chosen("C2", "F2a")
    real = lab.run_sample(matrices, windows, universe, RATES, configs)
    null = np.random.default_rng(1).normal(0, 1, (60, 2))

    table = lab.evaluate(real, configs, null, draws=200, min_trades=st.MIN_TRADES_DISCOVERY, allow_inconclusive=True)

    assert table["config"].tolist() == ["C2", "F2a"]
    assert not table["approved"].any()
    assert {"p_bootstrap", "p_placebo", "mde", "stop_fits_min_lot", "failed"} <= set(table.columns)


def test_the_commission_of_the_synthetic_cross_counts_two_legs() -> None:
    single = lab.commission_pips("EURUSD", 1.1, RATES, 2.25)
    synthetic = lab.commission_pips(lab.SYNTHETIC, 1.08, RATES, 2.25)

    assert single == pytest.approx(0.45, abs=0.01)
    assert synthetic > 2 * single


def test_code_commit_refuses_uncommitted_code(monkeypatch) -> None:
    monkeypatch.setattr(lab, "_git", lambda *arguments: " M app/fx/strategy.py")

    with pytest.raises(RuntimeError, match="uncommitted"):
        lab.code_commit()
    monkeypatch.setattr(lab, "_git", lambda *arguments: "" if arguments[0] == "status" else "abc123")
    assert lab.code_commit() == "abc123"

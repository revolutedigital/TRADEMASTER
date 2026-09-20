"""Regression: the compiled simulator against the pandas simulator used for the G0 study.

Both are given the very same signals, ATR, quotes and costs, so any difference in the trades is
a difference in the simulator mechanics. The synthetic case always runs; the real-data case runs
where the Dukascopy hourly bars are present (they are not committed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from app.fx.sim import core, sweep
from scripts.research import fx_spike
from tests.unit.test_engine_spike_numba import random_bars

PIP = 0.0001
REAL_DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "fx" / "EURUSD_H1.parquet"


def as_fx_spike_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["open_spread_pips"] = (frame["ask_open"] - frame["bid_open"]) / PIP
    frame["median_spread_pips"] = frame["open_spread_pips"]
    return frame


def compare(bars: pd.DataFrame, *, fast=8, slow=21, slippage_pips=0.2, warmup=120) -> tuple[int, int]:
    params = fx.ema_cross_params(
        fast_span=fast, slow_span=slow, atr_period=14, stop_atr=2.0, reward_risk=2.0, warmup=warmup
    )
    matrix = fx.bars_to_matrix(bars)
    intents, _, _ = fx.run_batch(
        fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, matrix,
        np.zeros(len(matrix), dtype=np.int64),
    )
    signals = pd.Series(intents.astype(float), index=bars.index)
    frame = as_fx_spike_bars(bars)
    mid = fx_spike.mid_ohlc(frame)
    atr = fx_spike.wilder_atr(mid)
    scenario = fx_spike.Scenario("free", fx_spike.FxCostModel(fx_spike.NoCommission(), slippage_pips=slippage_pips))

    reference = fx_spike.simulate(frame, signals, atr, symbol="EURUSD", scenario=scenario, warmup=warmup)
    new = core.run_simulation(
        fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, matrix, slippage_pips * PIP
    )

    entry_index, exit_index, side, entry_price, exit_price, _, reason = new
    assert len(reference) == len(side) > 5
    reason_names = {
        core.EXIT_SIGNAL: "signal", core.EXIT_STOP: "stop", core.EXIT_STOP_GAP: "stop_gap",
        core.EXIT_TARGET: "take_profit", core.EXIT_END: "end",
    }
    for i, trade in enumerate(reference):
        assert bars.index[int(entry_index[i])].to_pydatetime() == trade.entry_time
        assert (1 if trade.side == "LONG" else -1) == int(side[i])
        assert reason_names[int(reason[i])] == trade.exit_reason
        assert entry_price[i] == pytest.approx(trade.entry_price, abs=1e-9)
        assert exit_price[i] == pytest.approx(trade.exit_price, abs=1e-9)
        assert int(exit_index[i]) - int(entry_index[i]) == trade.bars_held
    return len(reference), int(np.sum(reason == core.EXIT_STOP_GAP))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_compiled_simulator_reproduces_the_pandas_simulator_on_synthetic_bars(seed) -> None:
    trades, _ = compare(random_bars(seed, n=6000), warmup=60)

    assert trades > 20


def test_the_compiled_simulator_reproduces_the_pandas_simulator_with_no_slippage() -> None:
    compare(random_bars(7, n=5000), slippage_pips=0.0, warmup=60)


@pytest.mark.skipif(not REAL_DATA.exists(), reason="Dukascopy hourly bars are not available")
@pytest.mark.parametrize("fast,slow", [(8, 21), (12, 26), (20, 100)])
def test_the_compiled_simulator_reproduces_the_pandas_simulator_on_ten_years_of_real_bars(fast, slow) -> None:
    hourly = pd.concat(
        [pd.read_parquet(REAL_DATA.parent.parent / directory / "EURUSD_H1.parquet") for directory in ("fx_confirm", "fx")]
    ).sort_index()
    bars = hourly[list(fx.BAR_COLUMNS)]

    trades, gaps = compare(bars, fast=fast, slow=slow, warmup=120)

    assert trades > 100
    assert gaps >= 1  # the weekend gaps in real data exercise the gapped-stop fill


def test_the_parallel_sweep_equals_running_each_configuration_alone() -> None:
    bars = random_bars(11, n=5000)
    matrix = fx.bars_to_matrix(bars)
    params = np.array(
        [
            fx.ema_cross_params(fast_span=f, slow_span=s, atr_period=14, stop_atr=1.5, reward_risk=2.0, warmup=60)
            for f, s in ((5, 21), (8, 34), (13, 55), (3, 15))
        ]
    )
    slip = np.full(len(matrix), 0.2 * PIP)
    commission = 0.45 * PIP

    counts, pips, r_total, wins = sweep.sweep(
        fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, matrix, slip, commission, PIP
    )

    for i in range(len(params)):
        _, _, side, entry_price, exit_price, stop_distance, _ = core.simulate(
            fx.ema_cross_step, fx.ema_cross_init, params[i], fx.EMA_CROSS_STATE_SIZE, matrix, slip
        )
        count, total, total_r, win = sweep.summarize(side, entry_price, exit_price, stop_distance, commission, PIP)
        assert counts[i] == count > 0 and wins[i] == win
        assert pips[i] == pytest.approx(total, abs=1e-9)
        assert r_total[i] == pytest.approx(total_r, abs=1e-9)


def test_the_strategy_driven_core_runs_far_above_the_one_million_bars_per_second_target() -> None:
    import time

    bars = random_bars(21, n=400_000)
    matrix = fx.bars_to_matrix(bars)
    params = fx.ema_cross_params(fast_span=8, slow_span=21, atr_period=14, stop_atr=1.5, reward_risk=2.0, warmup=60)
    slip = np.full(len(matrix), 0.2 * PIP)
    core.simulate(fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, matrix[:2000], slip[:2000])

    started = time.perf_counter()
    core.simulate(fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, matrix, slip)
    elapsed = time.perf_counter() - started

    assert len(matrix) / elapsed > 5_000_000  # a loose floor for shared CI machines; measured ~65M here

"""The compiled simulator core must agree, trade by trade, with an independent reference."""

import numpy as np
import pandas as pd
import pytest

from scripts.research.engine_spike import numba_core as core
from scripts.research.engine_spike.reference import simulate_reference

PIP = 0.0001
COLUMNS = (
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
)


def random_bars(seed: int, n: int = 3000, spread_pips: float = 0.6, volatility: float = 0.0004) -> pd.DataFrame:
    """A random-walk series with a realistic bid/ask spread and occasional gaps."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, volatility, n) * (1 + 6 * (rng.random(n) < 0.01))  # rare large moves
    mid_close = 1.10 * np.exp(np.cumsum(steps))
    mid_open = np.concatenate(([1.10], mid_close[:-1])) * (1 + rng.normal(0, 0.00005, n))
    high = np.maximum(mid_open, mid_close) * (1 + np.abs(rng.normal(0, 0.0002, n)))
    low = np.minimum(mid_open, mid_close) * (1 - np.abs(rng.normal(0, 0.0002, n)))
    half = spread_pips * PIP / 2
    frame = pd.DataFrame(index=pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"))
    for side, sign in (("bid", -1), ("ask", 1)):
        frame[f"{side}_open"] = mid_open + sign * half
        frame[f"{side}_high"] = high + sign * half
        frame[f"{side}_low"] = low + sign * half
        frame[f"{side}_close"] = mid_close + sign * half
    return frame


def run_core(bars: pd.DataFrame, **params):
    arrays = [bars[name].to_numpy(dtype=np.float64) for name in COLUMNS]
    return core.simulate(
        *arrays,
        params["fast_span"], params["slow_span"], params["atr_period"], params["stop_atr"],
        params["reward_risk"], params["slippage"], params.get("commission", 0.0), params["warmup"],
    )


PARAMS = dict(fast_span=8, slow_span=21, atr_period=14, stop_atr=1.5, reward_risk=2.0,
              slippage=0.2 * PIP, warmup=60)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "overrides",
    [{}, {"fast_span": 5, "slow_span": 34}, {"stop_atr": 0.8, "reward_risk": 1.0}, {"slippage": 0.0}],
)
def test_the_compiled_core_matches_the_independent_reference_trade_by_trade(seed, overrides) -> None:
    bars = random_bars(seed)
    params = PARAMS | overrides

    entry_index, exit_index, side, entry_price, exit_price, _, reason = run_core(bars, **params)
    reference = simulate_reference(bars, **params)

    assert len(reference) == len(side) > 10
    for i, trade in enumerate(reference):
        assert int(entry_index[i]) == trade["entry_index"]
        assert int(exit_index[i]) == trade["exit_index"]
        assert int(side[i]) == trade["side"]
        assert int(reason[i]) == trade["reason"]
        assert entry_price[i] == pytest.approx(trade["entry_price"], abs=1e-12)
        assert exit_price[i] == pytest.approx(trade["exit_price"], abs=1e-12)


def test_the_net_result_in_pips_matches_the_reference_within_a_hundredth_of_a_pip_per_trade() -> None:
    bars = random_bars(11, n=6000)
    params = PARAMS
    commission = 0.45 * PIP

    _, _, side, entry_price, exit_price, _, _ = run_core(bars, **params, commission=commission)
    count, total, _ = core.summarize(side, entry_price, exit_price, commission, PIP)
    reference = simulate_reference(bars, **params)
    reference_total = sum(
        ((t["exit_price"] - t["entry_price"]) * t["side"] - commission) / PIP for t in reference
    )

    assert count == len(reference)
    assert abs(total - reference_total) / count < 0.01


def test_every_trade_obeys_the_fill_semantics_without_looking_at_the_reference() -> None:
    bars = random_bars(21, n=8000)
    slip = 0.2 * PIP
    entry_index, exit_index, side, entry_price, exit_price, distance, reason = run_core(bars, **PARAMS)

    gross = (exit_price - entry_price) * side
    stop = reason == core.EXIT_STOP
    gapped = reason == core.EXIT_STOP_GAP
    target = reason == core.EXIT_TARGET

    assert stop.any() and gapped.any() and target.any()
    assert np.all(exit_index >= entry_index)
    assert np.allclose(gross[stop], -distance[stop] - slip, atol=1e-12)
    assert np.allclose(gross[target], PARAMS["reward_risk"] * distance[target], atol=1e-12)
    assert np.all(gross[gapped] <= -distance[gapped] - slip + 1e-12)


def test_entries_pay_the_spread_and_a_long_enters_above_the_bid_at_the_next_open() -> None:
    bars = random_bars(31, n=4000)
    entry_index, _, side, entry_price, _, _, _ = run_core(bars, **PARAMS)

    longs = side == core.LONG
    opens_ask = bars["ask_open"].to_numpy()[entry_index]
    opens_bid = bars["bid_open"].to_numpy()[entry_index]
    assert np.allclose(entry_price[longs], opens_ask[longs] + PARAMS["slippage"], atol=1e-12)
    assert np.allclose(entry_price[~longs], opens_bid[~longs] - PARAMS["slippage"], atol=1e-12)
    assert np.all(entry_index >= PARAMS["warmup"] + 1)  # nothing is entered during warmup or same-bar


def test_costs_only_ever_reduce_the_result() -> None:
    bars = random_bars(41, n=5000)
    free = run_core(bars, **(PARAMS | {"slippage": 0.0}))
    costly = run_core(bars, **PARAMS)

    _, free_total, _ = core.summarize(free[2], free[3], free[4], 0.0, PIP)
    _, costly_total, _ = core.summarize(costly[2], costly[3], costly[4], 0.0, PIP)

    assert costly_total < free_total


def test_the_parallel_sweep_equals_running_each_configuration_alone() -> None:
    bars = random_bars(51, n=4000)
    arrays = [bars[name].to_numpy(dtype=np.float64) for name in COLUMNS]
    fast = np.array([5, 8, 13], dtype=np.int64)
    slow = np.array([21, 34, 55], dtype=np.int64)
    commission = 0.3 * PIP

    counts, totals, wins = core.sweep(
        *arrays, fast, slow, 14, 1.5, 2.0, 0.2 * PIP, commission, PIP, 60
    )

    for i in range(3):
        _, _, side, entry_price, exit_price, _, _ = core.simulate(
            *arrays, fast[i], slow[i], 14, 1.5, 2.0, 0.2 * PIP, commission, 60
        )
        count, total, win = core.summarize(side, entry_price, exit_price, commission, PIP)
        assert counts[i] == count and wins[i] == win
        assert totals[i] == pytest.approx(total, abs=1e-9)


def test_a_series_shorter_than_its_warmup_never_trades() -> None:
    bars = random_bars(61, n=50)

    result = run_core(bars, **PARAMS)

    assert len(result[2]) == 0

"""The strategy contract: the same decisions streaming and in batch, and never any look-ahead."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx

PIP = 0.0001


def random_matrix(seed: int, n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.0004, n) * (1 + 6 * (rng.random(n) < 0.01))
    close = 1.10 * np.exp(np.cumsum(steps))
    open_ = np.concatenate(([1.10], close[:-1])) * (1 + rng.normal(0, 0.00005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0002, n)))
    half = 0.3 * PIP
    frame = pd.DataFrame(index=pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"))
    for side, sign in (("bid", -1), ("ask", 1)):
        frame[f"{side}_open"] = open_ + sign * half
        frame[f"{side}_high"] = high + sign * half
        frame[f"{side}_low"] = low + sign * half
        frame[f"{side}_close"] = close + sign * half
    return fx.bars_to_matrix(frame)


PARAMS = fx.ema_cross_params(
    fast_span=8, slow_span=21, atr_period=14, stop_atr=1.5, reward_risk=2.0, warmup=60
)


def batch(matrix: np.ndarray, params: np.ndarray = PARAMS):
    return fx.run_batch(
        fx.ema_cross_step, fx.ema_cross_init, params, fx.EMA_CROSS_STATE_SIZE, matrix,
        np.zeros(matrix.shape[0], dtype=np.int64),
    )


def test_bars_are_packed_in_the_documented_column_order() -> None:
    frame = pd.DataFrame(
        {
            "bid_open": [1.0], "bid_high": [2.0], "bid_low": [0.5], "bid_close": [1.5],
            "ask_open": [1.1], "ask_high": [2.1], "ask_low": [0.6], "ask_close": [1.6],
        },
        index=pd.DatetimeIndex(["2024-05-14 13:00"], tz="UTC"),
    )

    matrix = fx.bars_to_matrix(frame)

    assert matrix.shape == (1, fx.BAR_WIDTH)
    assert matrix[0, fx.BAR_TIME] == pd.Timestamp("2024-05-14 13:00", tz="UTC").timestamp()
    assert matrix[0, fx.BID_HIGH] == 2.0 and matrix[0, fx.ASK_CLOSE] == 1.6
    with pytest.raises(ValueError, match="timezone-aware"):
        fx.bars_to_matrix(frame.tz_localize(None))


def test_streaming_bar_by_bar_gives_exactly_the_decisions_of_the_batch_replay() -> None:
    matrix = random_matrix(1)
    intents, stops, targets = batch(matrix)
    runner = fx.StreamingRunner(fx.ema_cross_step, fx.ema_cross_init, PARAMS, fx.EMA_CROSS_STATE_SIZE)

    streamed = [runner.on_bar(bar) for bar in matrix]

    assert (intents != fx.HOLD).sum() > 10
    assert [s[0] for s in streamed] == intents.tolist()
    assert np.allclose([s[1] for s in streamed], stops, atol=0.0, rtol=0.0)
    assert np.allclose([s[2] for s in streamed], targets, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_a_decision_never_depends_on_bars_that_have_not_happened_yet(seed: int) -> None:
    matrix = random_matrix(seed, n=1500)
    full_intents, full_stops, _ = batch(matrix)

    for t in (200, 400, 777, 1100, 1499):
        prefix_intents, prefix_stops, _ = batch(matrix[: t + 1])
        assert prefix_intents[t] == full_intents[t]
        assert prefix_stops[t] == full_stops[t]


def test_changing_the_future_does_not_change_the_past() -> None:
    matrix = random_matrix(5, n=1500)
    altered = matrix.copy()
    altered[900:, fx.BID_CLOSE : fx.ASK_CLOSE + 1] *= 1.05  # a 5% jump from bar 900 on

    original = batch(matrix)
    changed = batch(altered)

    assert np.array_equal(original[0][:900], changed[0][:900])
    assert np.array_equal(original[1][:900], changed[1][:900])


def test_the_strategy_matches_an_independent_pandas_implementation_of_the_signals() -> None:
    matrix = random_matrix(6, n=3000)
    mid_close = pd.Series((matrix[:, fx.BID_CLOSE] + matrix[:, fx.ASK_CLOSE]) / 2)
    fast = mid_close.ewm(span=8, adjust=False).mean()
    slow = mid_close.ewm(span=21, adjust=False).mean()
    up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    down = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
    expected = np.zeros(len(mid_close), dtype=np.int8)
    expected[up.to_numpy()] = fx.ENTER_LONG
    expected[down.to_numpy()] = fx.ENTER_SHORT
    expected[:60] = 0

    intents, _, _ = batch(matrix)

    assert np.array_equal(intents, expected)


def test_a_new_entry_carries_an_atr_stop_and_a_reward_to_risk_target() -> None:
    matrix = random_matrix(7)
    intents, stops, targets = batch(matrix)

    entries = intents != fx.HOLD
    assert entries.any()
    assert np.all(stops[entries] > 0)
    assert np.allclose(targets[entries], 2.0 * stops[entries])
    assert np.all(stops[~entries] == 0) and np.all(targets[~entries] == 0)


def test_the_strategy_does_not_ask_to_enter_the_side_it_already_holds() -> None:
    matrix = random_matrix(8)
    flat = batch(matrix)[0]
    long_signals = np.flatnonzero(flat == fx.ENTER_LONG)
    runner = fx.StreamingRunner(fx.ema_cross_step, fx.ema_cross_init, PARAMS, fx.EMA_CROSS_STATE_SIZE)

    intents = []
    for t, bar in enumerate(matrix):
        intents.append(runner.on_bar(bar, position=fx.LONG)[0])

    assert long_signals.size > 0
    assert fx.ENTER_LONG not in intents
    assert fx.ENTER_SHORT in intents


def test_a_fresh_runner_starts_from_a_clean_state_each_time() -> None:
    matrix = random_matrix(9, n=800)

    def decisions() -> list[int]:
        runner = fx.StreamingRunner(fx.ema_cross_step, fx.ema_cross_init, PARAMS, fx.EMA_CROSS_STATE_SIZE)
        return [runner.on_bar(bar)[0] for bar in matrix]

    assert decisions() == decisions()


def test_nothing_is_decided_during_the_warmup() -> None:
    matrix = random_matrix(10, n=200)

    intents, _, _ = batch(matrix)

    assert np.all(intents[:60] == fx.HOLD)


def test_the_runner_rejects_a_malformed_bar_and_the_params_reject_nonsense() -> None:
    runner = fx.StreamingRunner(fx.ema_cross_step, fx.ema_cross_init, PARAMS, fx.EMA_CROSS_STATE_SIZE)

    with pytest.raises(ValueError, match="9 values"):
        runner.on_bar(np.zeros(5))
    for bad in (
        dict(fast_span=21, slow_span=8, atr_period=14, stop_atr=1.5, reward_risk=2.0, warmup=60),
        dict(fast_span=8, slow_span=21, atr_period=0, stop_atr=1.5, reward_risk=2.0, warmup=60),
        dict(fast_span=8, slow_span=21, atr_period=14, stop_atr=-1, reward_risk=2.0, warmup=60),
    ):
        with pytest.raises(ValueError):
            fx.ema_cross_params(**bad)


def test_the_timestamp_column_is_in_seconds_whatever_the_index_resolution() -> None:
    stamps = pd.date_range("2024-05-14 13:00", periods=3, freq="min", tz="UTC")
    frame = pd.DataFrame({name: [1.0, 1.0, 1.0] for name in fx.BAR_COLUMNS}, index=stamps)

    seconds = fx.bars_to_matrix(frame)[:, fx.BAR_TIME]
    coarse = fx.bars_to_matrix(frame.set_axis(stamps.as_unit("s")))[:, fx.BAR_TIME]
    fine = fx.bars_to_matrix(frame.set_axis(stamps.as_unit("ns")))[:, fx.BAR_TIME]

    assert seconds.tolist() == [1715691600.0, 1715691660.0, 1715691720.0]
    assert coarse.tolist() == seconds.tolist() == fine.tolist()

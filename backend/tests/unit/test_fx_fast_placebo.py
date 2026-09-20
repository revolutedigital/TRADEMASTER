"""The placebo keeps costs and volatility and removes every predictable move."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from scripts.research import fx_fast_placebo as pl
from tests.unit.fx_strategy_checks import synthetic_frame


def minute_matrix(seed: int = 1, weeks: int = 1) -> np.ndarray:
    return fx.bars_to_matrix(synthetic_frame(start="2024-05-13", weeks=weeks, bar_seconds=60, seed=seed, sigma_pips=0.7))


def mid(matrix: np.ndarray, column_bid: int, column_ask: int) -> np.ndarray:
    return 0.5 * (matrix[:, column_bid] + matrix[:, column_ask])


def test_with_every_coin_positive_the_bars_come_back_unchanged() -> None:
    matrix = minute_matrix()

    out = pl.sign_randomize(matrix, np.ones(len(matrix), dtype=np.int8))

    assert np.allclose(out, matrix, rtol=0, atol=1e-10)


def test_with_every_coin_negative_every_move_is_mirrored_and_the_spreads_stay() -> None:
    matrix = minute_matrix(2)

    out = pl.sign_randomize(matrix, -np.ones(len(matrix), dtype=np.int8))

    original = mid(matrix, fx.BID_CLOSE, fx.ASK_CLOSE)
    mirrored = mid(out, fx.BID_CLOSE, fx.ASK_CLOSE)
    assert np.allclose(np.diff(mirrored), -np.diff(original), atol=1e-12)
    for bid, ask in ((fx.BID_OPEN, fx.ASK_OPEN), (fx.BID_CLOSE, fx.ASK_CLOSE)):
        assert np.allclose(out[:, ask] - out[:, bid], matrix[:, ask] - matrix[:, bid], atol=1e-12)


def test_the_output_is_always_a_consistent_bid_ask_bar() -> None:
    matrix = minute_matrix(3, weeks=2)
    coins = pl.coins_for(matrix, pl.coin_table(0, int(matrix[-1, fx.BAR_TIME] // 60) + 1, 5), 0)

    out = pl.sign_randomize(matrix, coins)

    for prefix in (fx.BID_OPEN, fx.ASK_OPEN):
        assert np.all(out[:, prefix + 1] >= out[:, prefix]) and np.all(out[:, prefix + 2] <= out[:, prefix])
    assert np.all(out[:, fx.ASK_CLOSE] > out[:, fx.BID_CLOSE])
    assert np.all(out[:, fx.BID_HIGH] >= out[:, fx.BID_CLOSE]) and np.all(out[:, fx.ASK_LOW] <= out[:, fx.ASK_OPEN])


def test_sizes_of_the_moves_are_kept_but_their_direction_is_lost() -> None:
    matrix = minute_matrix(4, weeks=4)
    table = pl.coin_table(0, int(matrix[-1, fx.BAR_TIME] // 60) + 1, 9)
    out = pl.sign_randomize(matrix, pl.coins_for(matrix, table, 0))

    original = np.diff(mid(matrix, fx.BID_CLOSE, fx.ASK_CLOSE))
    placebo = np.diff(mid(out, fx.BID_CLOSE, fx.ASK_CLOSE))

    assert np.allclose(np.abs(placebo), np.abs(original), atol=1e-12)
    assert abs(placebo.mean()) < 3 * placebo.std() / np.sqrt(placebo.size)
    assert abs(np.corrcoef(placebo[1:], placebo[:-1])[0, 1]) < 0.05  # no serial dependence left


def test_a_drift_is_removed() -> None:
    # A market that only rises: the placebo has no trend, whatever the original does.
    matrix = minute_matrix(5, weeks=2)
    rising = matrix.copy()
    rising[:, 1:] += np.arange(len(matrix))[:, None] * 1e-5
    table = pl.coin_table(0, int(matrix[-1, fx.BAR_TIME] // 60) + 1, 11)

    out = pl.sign_randomize(rising, pl.coins_for(rising, table, 0))

    drift = np.diff(mid(out, fx.BID_CLOSE, fx.ASK_CLOSE))
    assert np.diff(mid(rising, fx.BID_CLOSE, fx.ASK_CLOSE)).mean() > 5e-6
    assert abs(drift.mean()) < 5 * drift.std() / np.sqrt(drift.size)


def test_two_pairs_share_the_coin_at_the_same_minute() -> None:
    first, second = minute_matrix(6), minute_matrix(7)
    second = second[10:]  # the second pair is missing its first ten minutes
    table = pl.coin_table(0, int(first[-1, fx.BAR_TIME] // 60) + 1, 13)

    a, b = pl.coins_for(first, table, 0), pl.coins_for(second, table, 0)

    assert np.array_equal(a[10:], b)
    assert set(np.unique(a)) == {-1, 1}


def test_the_coin_table_is_reproducible_and_bounds_are_checked() -> None:
    assert np.array_equal(pl.coin_table(0, 99, 1), pl.coin_table(0, 99, 1))
    assert not np.array_equal(pl.coin_table(0, 99, 1), pl.coin_table(0, 99, 2))
    with pytest.raises(ValueError):
        pl.coin_table(5, 4, 1)
    matrix = minute_matrix()
    with pytest.raises(ValueError, match="outside"):
        pl.coins_for(matrix, pl.coin_table(0, 10, 1), 0)

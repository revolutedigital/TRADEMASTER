"""Minute bars aggregate into coarser bid/ask bars exactly like an independent pandas reference."""

import numpy as np
import pandas as pd
import pytest

from app.fx import bars
from app.fx import strategy as fx


def minute_frame(seed: int, minutes: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0002, minutes))
    open_ = np.concatenate(([1.10], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0001, minutes))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0001, minutes))
    frame = pd.DataFrame(index=pd.date_range("2024-05-13 00:00", periods=minutes, freq="min", tz="UTC"))
    for side, sign in (("bid", -1), ("ask", 1)):
        half = 0.00005 + 0.00002 * rng.random(minutes)
        frame[f"{side}_open"] = open_ + sign * half
        frame[f"{side}_high"] = high + sign * half
        frame[f"{side}_low"] = low + sign * half
        frame[f"{side}_close"] = close + sign * half
    drop = rng.random(minutes) < 0.05
    drop[1200:1500] = True  # a five-hour hole
    return frame[~drop]


def reference(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    resampler = frame.resample(f"{minutes}min", origin="epoch", label="left", closed="left")
    aggregated = pd.concat(
        {
            **{f"{side}_open": resampler[f"{side}_open"].first() for side in ("bid", "ask")},
            **{f"{side}_high": resampler[f"{side}_high"].max() for side in ("bid", "ask")},
            **{f"{side}_low": resampler[f"{side}_low"].min() for side in ("bid", "ask")},
            **{f"{side}_close": resampler[f"{side}_close"].last() for side in ("bid", "ask")},
        },
        axis=1,
    )
    return aggregated.dropna()


@pytest.mark.parametrize("minutes", [5, 15, 60])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_aggregation_equals_pandas_resample_with_holes(minutes: int, seed: int) -> None:
    frame = minute_frame(seed)

    result = bars.aggregate_minutes(fx.bars_to_matrix(frame), minutes * 60)
    expected = reference(frame, minutes)

    assert result.shape[0] == len(expected)
    assert np.array_equal(result[:, fx.BAR_TIME], fx.bars_to_matrix(expected)[:, fx.BAR_TIME])
    for offset, column in enumerate(fx.BAR_COLUMNS, start=1):
        assert np.allclose(result[:, offset], expected[column].to_numpy(), rtol=0, atol=1e-12), column


def test_an_empty_bucket_produces_no_bar_instead_of_a_flat_one() -> None:
    frame = minute_frame(4)
    result = bars.aggregate_minutes(fx.bars_to_matrix(frame), 3600)
    hours = result[:, fx.BAR_TIME]

    assert np.all(np.diff(hours) >= 3600)
    assert np.any(np.diff(hours) > 3600)  # the five-hour hole stays a hole


def test_bars_are_stamped_with_the_epoch_aligned_open_and_keep_the_high_above_the_low() -> None:
    result = bars.aggregate_minutes(fx.bars_to_matrix(minute_frame(5)), 900)

    assert np.all(result[:, fx.BAR_TIME] % 900 == 0)
    assert np.all(result[:, fx.BID_HIGH] >= result[:, fx.BID_LOW])
    assert np.all(result[:, fx.ASK_HIGH] >= result[:, fx.ASK_LOW])
    assert np.all(result[:, fx.ASK_CLOSE] >= result[:, fx.BID_CLOSE])


def test_the_bucket_boundaries_can_be_reused_for_another_price_path() -> None:
    matrix = fx.bars_to_matrix(minute_frame(6))
    starts = bars.bucket_starts(matrix[:, fx.BAR_TIME], 300)
    shifted = matrix.copy()
    shifted[:, 1:] += 0.01

    first = bars.aggregate(matrix, starts, 300)
    second = bars.aggregate(shifted, starts, 300)

    assert np.array_equal(first[:, fx.BAR_TIME], second[:, fx.BAR_TIME])
    assert np.allclose(second[:, 1:] - first[:, 1:], 0.01)


def test_one_minute_bars_pass_through_and_bad_inputs_are_rejected() -> None:
    matrix = fx.bars_to_matrix(minute_frame(7, minutes=200))

    assert np.array_equal(bars.aggregate_minutes(matrix, 60), matrix)
    with pytest.raises(ValueError, match="whole number of minutes"):
        bars.bucket_starts(matrix[:, fx.BAR_TIME], 90)
    with pytest.raises(ValueError, match="strictly increasing"):
        bars.bucket_starts(matrix[::-1, fx.BAR_TIME].copy(), 300)
    assert bars.bucket_starts(np.empty(0), 300).size == 0

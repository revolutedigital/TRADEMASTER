"""Tests for interval-aware walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.walk_forward import candles_per_day, run_walk_forward


def _candles(rows: int) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 100.0),
        }
    )


def test_candles_per_day_matches_requested_interval() -> None:
    assert candles_per_day("15m") == 96
    assert candles_per_day("1h") == 24
    assert candles_per_day("4h") == 6
    assert candles_per_day("1d") == 1
    with pytest.raises(ValueError, match="unsupported"):
        candles_per_day("2h")


def test_one_hour_walk_forward_uses_one_hour_candle_cadence() -> None:
    frame = _candles(1_800)
    signals = pd.Series(np.zeros(len(frame)))

    result = run_walk_forward(
        df=frame,
        signals=signals,
        train_days=60,
        test_days=15,
        step_days=15,
        allow_short=False,
        candles_per_day=candles_per_day("1h"),
    )

    assert len(result.windows) == 1

"""Tests for the bounded closed-candle pattern library."""

import numpy as np
import pandas as pd

from app.services.market.pattern_intelligence import assess_closed_candle_pattern


def _candles(closes: np.ndarray, *, final_quote_volume: float = 150_000.0) -> pd.DataFrame:
    quote_volume = np.full(len(closes), 100_000.0)
    quote_volume[-1] = final_quote_volume
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": quote_volume / closes,
            "quote_volume": quote_volume,
            "taker_buy_quote": quote_volume * 0.58,
        }
    )


def test_uptrend_with_confirmed_flow_is_classified_as_trend_continuation() -> None:
    assessment = assess_closed_candle_pattern(_candles(np.linspace(100, 160, 260)))

    assert assessment.regime == "UPTREND"
    assert assessment.pattern == "TREND_CONTINUATION"
    assert assessment.flow_data_available is True
    assert assessment.taker_buy_imbalance is not None and assessment.taker_buy_imbalance > 0


def test_downtrend_is_observation_only_for_spot_long_only_research() -> None:
    assessment = assess_closed_candle_pattern(_candles(np.linspace(160, 100, 260)))

    assert assessment.regime == "DOWNTREND"
    assert assessment.pattern == "OBSERVATION_ONLY"

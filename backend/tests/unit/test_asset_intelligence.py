import numpy as np
import pandas as pd

from app.services.asset_intelligence import (
    _candidates_for_trend,
    _choose_candidate,
    _market_study,
    _predictive_model_blocker,
)
from app.services.market.spot_asset_catalog import SpotAsset


def _candles(closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC"),
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(len(closes), 1000.0),
            "close_time": pd.date_range("2025-01-01 00:59", periods=len(closes), freq="h", tz="UTC"),
            "quote_volume": np.full(len(closes), 100_000.0),
            "trade_count": np.full(len(closes), 100, dtype=int),
        }
    )


def test_market_study_detects_uptrend_and_reports_liquidity() -> None:
    asset = SpotAsset(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        quote_volume_24h=1_500_000.0,
        price_change_pct_24h=4.2,
    )
    study = _market_study(_candles(np.linspace(100, 200, 300)), asset)

    assert study["trend"] == "UPTREND"
    assert study["candles"] == 300
    assert study["liquidity_quote_volume_24h"] == 1_500_000.0
    assert study["volatility_pct"] >= 0


def test_automatic_candidates_are_bounded_to_spot_long_only_strategies() -> None:
    candidates = _candidates_for_trend("RANGE")

    assert len(candidates) == 3
    assert all(candidate.strategy.execution_profile == "spot_long_only" for candidate in candidates)
    assert all(candidate.strategy.min_confirmations <= len(candidate.strategy.indicators) for candidate in candidates)


def test_predictive_model_rejects_unreliable_or_bearish_spot_recommendations() -> None:
    assert _predictive_model_blocker({"trained": False}) is not None
    assert _predictive_model_blocker({"trained": True, "validation_accuracy": 0.39, "latest_signal": "BUY"}) is not None
    assert _predictive_model_blocker({"trained": True, "validation_accuracy": 0.55, "latest_signal": "SELL"}) is not None
    assert _predictive_model_blocker({"trained": True, "validation_accuracy": 0.55, "latest_signal": "BUY"}) is None


def test_neutral_predictive_model_requires_stronger_technical_confirmation() -> None:
    candles = _candles(np.linspace(100, 200, 300))

    buy_candidate = _choose_candidate(candles, "UPTREND", "BUY")
    neutral_candidate = _choose_candidate(candles, "UPTREND", "HOLD")

    assert neutral_candidate.candidate.signal_threshold >= buy_candidate.candidate.signal_threshold

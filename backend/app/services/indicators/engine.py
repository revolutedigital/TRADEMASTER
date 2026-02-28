"""Technical indicator computation orchestrator."""

import pandas as pd

from app.core.logging import get_logger
from app.services.indicators.trend import compute_trend_indicators
from app.services.indicators.momentum import compute_momentum_indicators
from app.services.indicators.volatility import compute_volatility_indicators
from app.services.indicators.volume import compute_volume_indicators

logger = get_logger(__name__)


class IndicatorEngine:
    """Computes 50+ technical indicators on OHLCV DataFrames.

    All methods are stateless: they take a DataFrame and return an enriched one.
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicator categories on the given OHLCV DataFrame.

        Expected columns: open, high, low, close, volume
        Returns: DataFrame with all original columns plus indicator columns.
        """
        if df.empty or len(df) < 30:
            logger.warning("insufficient_data_for_indicators", rows=len(df))
            return df

        df = df.copy()
        df = compute_trend_indicators(df)
        df = compute_momentum_indicators(df)
        df = compute_volatility_indicators(df)
        df = compute_volume_indicators(df)

        return df

    def compute_selective(
        self, df: pd.DataFrame, categories: list[str] | None = None
    ) -> pd.DataFrame:
        """Compute only requested indicator categories.

        categories: list of "trend", "momentum", "volatility", "volume"
        """
        if df.empty:
            return df

        df = df.copy()
        dispatch = {
            "trend": compute_trend_indicators,
            "momentum": compute_momentum_indicators,
            "volatility": compute_volatility_indicators,
            "volume": compute_volume_indicators,
        }

        for cat in categories or dispatch.keys():
            fn = dispatch.get(cat)
            if fn:
                df = fn(df)

        return df

    @staticmethod
    def list_indicators() -> dict[str, list[str]]:
        """Return a dict of category -> indicator names."""
        return {
            "trend": [
                "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "ema_55",
                "macd", "macd_signal", "macd_hist",
                "adx", "plus_di", "minus_di",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
                "ichimoku_senkou_b",
                "supertrend", "supertrend_direction",
                "hma_20",
            ],
            "momentum": [
                "rsi_14", "stoch_k", "stoch_d",
                "cci_20", "williams_r",
                "roc_12", "ultimate_osc", "awesome_osc",
                "mfi_14",
            ],
            "volatility": [
                "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
                "atr_14", "atr_normalized",
                "keltner_upper", "keltner_lower",
                "donchian_upper", "donchian_lower",
                "historical_vol_20",
            ],
            "volume": [
                "obv", "obv_sma",
                "vwap",
                "cmf_20",
                "force_index",
                "volume_sma_20", "volume_ratio",
            ],
        }


indicator_engine = IndicatorEngine()

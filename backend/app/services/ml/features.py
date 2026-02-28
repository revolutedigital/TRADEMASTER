"""Feature engineering pipeline: transforms raw OHLCV + indicators into ML features."""

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.indicators.engine import indicator_engine

logger = get_logger(__name__)


class FeatureEngineer:
    """Generates 120+ features from raw OHLCV data for ML models."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline.

        Input: DataFrame with open, high, low, close, volume columns.
        Output: DataFrame with 120+ feature columns.
        """
        if df.empty or len(df) < 200:
            logger.warning("insufficient_data_for_features", rows=len(df))
            return df

        df = df.copy()

        # 1. Technical indicators (50+ columns)
        df = indicator_engine.compute_all(df)

        # 2. Price features (20+ columns)
        df = self._price_features(df)

        # 3. Return features
        df = self._return_features(df)

        # 4. Volume features
        df = self._volume_features(df)

        # 5. Volatility features
        df = self._volatility_features(df)

        # 6. Lag features
        df = self._lag_features(df)

        # 7. Cross features (interactions)
        df = self._cross_features(df)

        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Price relative to moving averages
        for ma_col in ["sma_20", "sma_50", "sma_200", "ema_9", "ema_21"]:
            if ma_col in df.columns:
                df[f"price_vs_{ma_col}"] = (close - df[ma_col]) / df[ma_col].replace(0, np.nan)

        # Price percentile rank
        df["price_pctrank_20"] = close.rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )
        df["price_pctrank_50"] = close.rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )

        # High/low range
        df["hl_range"] = (df["high"] - df["low"]) / close
        df["gap"] = (df["open"] - close.shift()) / close.shift().replace(0, np.nan)

        # Price position in day's range
        hl_denom = (df["high"] - df["low"]).replace(0, np.nan)
        df["close_position"] = (close - df["low"]) / hl_denom

        return df

    def _return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Simple returns at multiple horizons
        for period in [1, 3, 5, 10, 20]:
            df[f"return_{period}"] = close.pct_change(period)

        # Log returns
        df["log_return_1"] = np.log(close / close.shift(1))
        df["log_return_5"] = np.log(close / close.shift(5))

        # Cumulative returns
        df["cum_return_5"] = close.pct_change(5)
        df["cum_return_20"] = close.pct_change(20)

        # Return momentum (acceleration)
        ret_1 = close.pct_change(1)
        df["return_acceleration"] = ret_1 - ret_1.shift(1)

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        volume = df["volume"]

        # Volume momentum
        df["volume_change_1"] = volume.pct_change(1)
        df["volume_change_5"] = volume.pct_change(5)

        # Volume percentile
        df["volume_pctrank_20"] = volume.rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )

        # Price-volume correlation (rolling 20)
        df["price_volume_corr"] = (
            df["close"].rolling(20).corr(volume)
        )

        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        log_ret = np.log(close / close.shift())

        # Rolling volatility at multiple windows
        df["volatility_5"] = log_ret.rolling(5).std() * np.sqrt(252)
        df["volatility_10"] = log_ret.rolling(10).std() * np.sqrt(252)
        df["volatility_20"] = log_ret.rolling(20).std() * np.sqrt(252)

        # Volatility ratio (short/long)
        vol_20 = df["volatility_20"].replace(0, np.nan)
        df["vol_ratio_5_20"] = df["volatility_5"] / vol_20
        df["vol_ratio_10_20"] = df["volatility_10"] / vol_20

        # Parkinson volatility estimator
        hl_ratio = np.log(df["high"] / df["low"])
        df["parkinson_vol"] = (
            hl_ratio.pow(2).rolling(20).mean() / (4 * np.log(2))
        ).apply(np.sqrt)

        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of key indicators."""
        lag_cols = ["rsi_14", "macd_hist", "bb_pct", "volume_ratio", "atr_normalized"]

        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 3, 5]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    def _cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between indicators."""
        # RSI x Volume
        if "rsi_14" in df.columns and "volume_ratio" in df.columns:
            df["rsi_volume_interaction"] = df["rsi_14"] * df["volume_ratio"]

        # MACD x ADX (trend strength)
        if "macd_hist" in df.columns and "adx" in df.columns:
            df["macd_adx_interaction"] = df["macd_hist"] * df["adx"]

        # Bollinger %B x RSI (mean reversion signal)
        if "bb_pct" in df.columns and "rsi_14" in df.columns:
            df["bb_rsi_interaction"] = df["bb_pct"] * df["rsi_14"]

        # EMA crossover signals
        if "ema_9" in df.columns and "ema_21" in df.columns:
            df["ema_9_21_cross"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"].replace(
                0, np.nan
            )

        # Supertrend alignment
        if "supertrend_direction" in df.columns and "macd_hist" in df.columns:
            df["trend_alignment"] = df["supertrend_direction"] * np.sign(df["macd_hist"])

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of feature column names (excludes raw OHLCV and meta columns)."""
        exclude = {
            "open_time", "close_time", "open", "high", "low", "close", "volume",
            "quote_volume", "trade_count", "target",
        }
        return [c for c in df.columns if c not in exclude]


feature_engineer = FeatureEngineer()

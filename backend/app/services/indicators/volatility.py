"""Volatility indicators: Bollinger Bands, ATR, Keltner, Donchian, Historical Vol."""

import numpy as np
import pandas as pd


def compute_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- Bollinger Bands (20, 2) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_middle"] = bb_mid
    df["bb_lower"] = bb_mid - 2 * bb_std
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = bb_range / bb_mid
    df["bb_pct"] = (close - df["bb_lower"]) / bb_range

    # --- ATR (14) ---
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_normalized"] = df["atr_14"] / close

    # --- Keltner Channel (20, 1.5) ---
    ema_20 = close.ewm(span=20, adjust=False).mean()
    df["keltner_upper"] = ema_20 + 1.5 * df["atr_14"]
    df["keltner_lower"] = ema_20 - 1.5 * df["atr_14"]

    # --- Donchian Channel (20) ---
    df["donchian_upper"] = high.rolling(20).max()
    df["donchian_lower"] = low.rolling(20).min()

    # --- Historical Volatility (20-day) ---
    log_returns = np.log(close / close.shift())
    df["historical_vol_20"] = log_returns.rolling(20).std() * np.sqrt(252)

    return df

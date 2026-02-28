"""Volume indicators: OBV, VWAP, CMF, Force Index."""

import numpy as np
import pandas as pd


def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- On-Balance Volume (OBV) ---
    obv_direction = np.where(close > close.shift(), 1, np.where(close < close.shift(), -1, 0))
    df["obv"] = (volume * obv_direction).cumsum()
    df["obv_sma"] = df["obv"].rolling(20).mean()

    # --- VWAP (session-based, using rolling 20 as proxy) ---
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).rolling(20).sum()
    cumulative_vol = volume.rolling(20).sum().replace(0, np.nan)
    df["vwap"] = cumulative_tp_vol / cumulative_vol

    # --- Chaikin Money Flow (20) ---
    mfv_denom = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / mfv_denom
    money_flow_volume = money_flow_multiplier * volume
    df["cmf_20"] = money_flow_volume.rolling(20).sum() / cumulative_vol

    # --- Force Index (13-period EMA) ---
    df["force_index"] = (close.diff() * volume).ewm(span=13, adjust=False).mean()

    # --- Volume SMA and Ratio ---
    df["volume_sma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

    return df

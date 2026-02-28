"""Momentum indicators: RSI, Stochastic, CCI, Williams %R, ROC, etc."""

import numpy as np
import pandas as pd


def compute_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- RSI (14) ---
    df["rsi_14"] = _rsi(close, period=14)

    # --- Stochastic Oscillator (14, 3, 3) ---
    lowest_14 = low.rolling(14).min()
    highest_14 = high.rolling(14).max()
    denom = (highest_14 - lowest_14).replace(0, np.nan)
    df["stoch_k"] = 100 * (close - lowest_14) / denom
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- CCI (20) ---
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # --- Williams %R (14) ---
    df["williams_r"] = -100 * (highest_14 - close) / denom

    # --- Rate of Change (12) ---
    df["roc_12"] = close.pct_change(12) * 100

    # --- Ultimate Oscillator (7, 14, 28) ---
    df["ultimate_osc"] = _ultimate_oscillator(high, low, close)

    # --- Awesome Oscillator ---
    midpoint = (high + low) / 2
    df["awesome_osc"] = midpoint.rolling(5).mean() - midpoint.rolling(34).mean()

    # --- Money Flow Index (14) ---
    df["mfi_14"] = _mfi(high, low, close, volume, period=14)

    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ultimate_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    prev_close = close.shift()
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)

    return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7


def _mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    delta = typical_price.diff()

    positive_flow = money_flow.where(delta > 0, 0.0).rolling(period).sum()
    negative_flow = money_flow.where(delta <= 0, 0.0).rolling(period).sum()

    mfr = positive_flow / negative_flow.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))

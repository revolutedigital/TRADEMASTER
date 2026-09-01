"""Deterministic technical-strategy signals for research backtests.

The signal generator deliberately has no exchange dependency and never emits an
execution instruction. It turns the configuration captured with a backtest
into a reproducible series of long-entry (+1), long-exit (-1), or hold (0).
"""

from dataclasses import dataclass

import pandas as pd

from app.schemas.trading import TechnicalStrategyConfig


@dataclass(frozen=True)
class TechnicalStrategyDefinition:
    """Normalized technical strategy details used in a research result."""

    name: str
    indicators: tuple[str, ...]
    min_confirmations: int


def build_technical_strategy_signals(
    candles: pd.DataFrame,
    strategy: TechnicalStrategyConfig,
) -> tuple[pd.Series, TechnicalStrategyDefinition]:
    """Generate deterministic entry/exit signals from selected indicators.

    An indicator votes only at a new crossover or band/threshold re-entry. A
    trade signal exists only when the required number of selected indicators
    votes in the same direction on the same candle. This avoids treating a
    persistently overbought/oversold condition as a new trade on every bar.
    """
    if "close" not in candles:
        raise ValueError("candles must include a close column")

    close = pd.to_numeric(candles["close"], errors="coerce")
    votes: dict[str, pd.Series] = {}

    for indicator in strategy.indicators:
        parameters = strategy.indicator_params.get(indicator, {})
        if indicator == "sma":
            votes[indicator] = _moving_average_crossover(
                close,
                short_period=int(parameters.get("sma_short", 10)),
                long_period=int(parameters.get("sma_long", 30)),
                exponential=False,
            )
        elif indicator == "ema":
            votes[indicator] = _moving_average_crossover(
                close,
                short_period=int(parameters.get("ema_short", 12)),
                long_period=int(parameters.get("ema_long", 26)),
                exponential=True,
            )
        elif indicator == "rsi":
            votes[indicator] = _rsi_reentry(
                close,
                period=int(parameters.get("rsi_period", 14)),
                overbought=float(parameters.get("rsi_overbought", 70)),
                oversold=float(parameters.get("rsi_oversold", 30)),
            )
        elif indicator == "macd":
            votes[indicator] = _macd_crossover(
                close,
                fast_period=int(parameters.get("macd_fast", 12)),
                slow_period=int(parameters.get("macd_slow", 26)),
                signal_period=int(parameters.get("macd_signal", 9)),
            )
        elif indicator == "bollinger":
            votes[indicator] = _bollinger_reentry(
                close,
                period=int(parameters.get("bb_period", 20)),
                stddev=float(parameters.get("bb_std", 2)),
            )
        elif indicator == "engulfing":
            votes[indicator] = _engulfing_reversal(candles)
        elif indicator == "breakout":
            votes[indicator] = _breakout_signal(
                candles,
                lookback=int(parameters.get("breakout_lookback", 20)),
            )

    vote_frame = pd.DataFrame(votes, index=candles.index).fillna(0.0)
    positive_votes = (vote_frame > 0).sum(axis=1)
    negative_votes = (vote_frame < 0).sum(axis=1)

    signals = pd.Series(0.0, index=candles.index, dtype=float)
    bullish = (
        (positive_votes >= strategy.min_confirmations)
        & (positive_votes > negative_votes)
    )
    bearish = (
        (negative_votes >= strategy.min_confirmations)
        & (negative_votes > positive_votes)
    )
    signals.loc[bullish] = 1.0
    signals.loc[bearish] = -1.0

    definition = TechnicalStrategyDefinition(
        name="Technical ensemble (Spot long-only)",
        indicators=tuple(strategy.indicators),
        min_confirmations=strategy.min_confirmations,
    )
    return signals, definition


def _moving_average_crossover(
    close: pd.Series,
    *,
    short_period: int,
    long_period: int,
    exponential: bool,
) -> pd.Series:
    if exponential:
        short_average = close.ewm(span=short_period, adjust=False, min_periods=short_period).mean()
        long_average = close.ewm(span=long_period, adjust=False, min_periods=long_period).mean()
    else:
        short_average = close.rolling(short_period, min_periods=short_period).mean()
        long_average = close.rolling(long_period, min_periods=long_period).mean()
    return _cross_signal(short_average, long_average)


def _rsi_reentry(
    close: pd.Series,
    *,
    period: int,
    overbought: float,
    oversold: float,
) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + relative_strength))

    signals = pd.Series(0.0, index=close.index, dtype=float)
    signals.loc[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1.0
    signals.loc[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1.0
    return signals


def _macd_crossover(
    close: pd.Series,
    *,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> pd.Series:
    fast = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd = fast - slow
    signal = macd.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    return _cross_signal(macd, signal)


def _bollinger_reentry(close: pd.Series, *, period: int, stddev: float) -> pd.Series:
    middle = close.rolling(period, min_periods=period).mean()
    deviation = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + deviation * stddev
    lower = middle - deviation * stddev

    signals = pd.Series(0.0, index=close.index, dtype=float)
    signals.loc[(close > lower) & (close.shift(1) <= lower.shift(1))] = 1.0
    signals.loc[(close < upper) & (close.shift(1) >= upper.shift(1))] = -1.0
    return signals


def _engulfing_reversal(candles: pd.DataFrame) -> pd.Series:
    """Emit a single signal for bullish or bearish real-body engulfing candles."""
    _require_columns(candles, "open", "close")
    open_price = pd.to_numeric(candles["open"], errors="coerce")
    close = pd.to_numeric(candles["close"], errors="coerce")
    previous_open = open_price.shift(1)
    previous_close = close.shift(1)

    bullish = (
        (close > open_price)
        & (previous_close < previous_open)
        & (open_price <= previous_close)
        & (close >= previous_open)
    )
    bearish = (
        (close < open_price)
        & (previous_close > previous_open)
        & (open_price >= previous_close)
        & (close <= previous_open)
    )

    signals = pd.Series(0.0, index=candles.index, dtype=float)
    signals.loc[bullish] = 1.0
    signals.loc[bearish] = -1.0
    return signals


def _breakout_signal(candles: pd.DataFrame, *, lookback: int) -> pd.Series:
    """Emit once when close breaks the preceding rolling high or low."""
    _require_columns(candles, "high", "low", "close")
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    close = pd.to_numeric(candles["close"], errors="coerce")
    previous_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
    previous_low = low.shift(1).rolling(lookback, min_periods=lookback).min()
    above_range = close > previous_high
    below_range = close < previous_low

    signals = pd.Series(0.0, index=candles.index, dtype=float)
    signals.loc[above_range & ~above_range.shift(1, fill_value=False)] = 1.0
    signals.loc[below_range & ~below_range.shift(1, fill_value=False)] = -1.0
    return signals


def _require_columns(candles: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in candles]
    if missing:
        raise ValueError(f"candles must include {', '.join(missing)}")


def _cross_signal(left: pd.Series, right: pd.Series) -> pd.Series:
    signals = pd.Series(0.0, index=left.index, dtype=float)
    signals.loc[(left > right) & (left.shift(1) <= right.shift(1))] = 1.0
    signals.loc[(left < right) & (left.shift(1) >= right.shift(1))] = -1.0
    return signals

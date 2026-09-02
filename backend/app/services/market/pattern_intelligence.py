"""Closed-candle, explainable pattern classification for asset research.

The library deliberately recognizes a small number of falsifiable setups. It
does not infer a story from every chart: unstable, weak-flow, or ambiguous
conditions remain observation-only and cannot create an automatic strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


FLOW_LOOKBACK = 48
TREND_SHORT_WINDOW = 50
TREND_LONG_WINDOW = 200
COMPRESSION_LOOKBACK = 120
BREAKOUT_LOOKBACK = 20


@dataclass(frozen=True)
class PatternAssessment:
    """An immutable classification based only on completed candle data."""

    regime: str
    pattern: str
    confidence: float
    relative_volume: float
    taker_buy_imbalance: float | None
    flow_data_available: bool
    explanation: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_closed_candle_pattern(candles: pd.DataFrame) -> PatternAssessment:
    """Classify the latest closed-candle context from price and quote flow.

    Quote-volume is used rather than base quantity so values stay comparable
    across assets. Taker-buy imbalance is optional: old records without it
    retain a robust relative-volume check but are never represented as order
    book data.
    """
    required = {"high", "low", "close", "volume"}
    missing = sorted(required - set(candles.columns))
    if missing:
        raise ValueError(f"candles must include {', '.join(missing)}")
    if len(candles) < TREND_LONG_WINDOW:
        raise ValueError("at least 200 closed candles are required for pattern assessment")

    frame = candles.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    quote_volume = _quote_volume(frame, close)
    if close.isna().any() or high.isna().any() or low.isna().any() or quote_volume.isna().any():
        raise ValueError("candles contain invalid price or volume values")

    relative_volume = _relative_volume(quote_volume)
    imbalance = _taker_buy_imbalance(frame, quote_volume)
    flow_available = imbalance is not None
    short_average = float(close.iloc[-TREND_SHORT_WINDOW:].mean())
    long_average = float(close.iloc[-TREND_LONG_WINDOW:].mean())
    latest_price = float(close.iloc[-1])

    returns = close.pct_change()
    stress = _is_stress(returns, relative_volume)
    trend_separation = abs(short_average / long_average - 1.0) if long_average > 0 else 0.0
    compression = _is_compression(close) and trend_separation < 0.03
    breakout = _is_upside_breakout(high, close)
    lower_band_reentry = _is_lower_band_reentry(close)
    buy_flow_ok = imbalance is None or imbalance >= -0.10

    if stress:
        return PatternAssessment(
            regime="STRESS",
            pattern="OBSERVATION_ONLY",
            confidence=_clamp(0.55 + min(relative_volume, 2.0) * 0.10),
            relative_volume=relative_volume,
            taker_buy_imbalance=imbalance,
            flow_data_available=flow_available,
            explanation="Volatilidade recente está fora do regime normal; o ativo fica somente em observação.",
        )

    if compression:
        if breakout and relative_volume >= 1.10 and buy_flow_ok:
            return PatternAssessment(
                regime="COMPRESSION",
                pattern="COMPRESSION_BREAKOUT",
                confidence=_clamp(0.45 + min(relative_volume, 2.5) * 0.16 + _flow_bonus(imbalance)),
                relative_volume=relative_volume,
                taker_buy_imbalance=imbalance,
                flow_data_available=flow_available,
                explanation="Compressão de volatilidade rompeu a máxima recente com confirmação de volume.",
            )
        return PatternAssessment(
            regime="COMPRESSION",
            pattern="OBSERVATION_ONLY",
            confidence=_clamp(0.35 + min(relative_volume, 2.0) * 0.10),
            relative_volume=relative_volume,
            taker_buy_imbalance=imbalance,
            flow_data_available=flow_available,
            explanation="Há compressão, mas ainda não existe rompimento com fluxo suficiente para entrada.",
        )

    if latest_price > short_average > long_average:
        if relative_volume >= 0.90 and buy_flow_ok:
            return PatternAssessment(
                regime="UPTREND",
                pattern="TREND_CONTINUATION",
                confidence=_clamp(
                    0.40
                    + min((latest_price / long_average - 1.0) * 18, 0.22)
                    + min(relative_volume, 2.0) * 0.10
                    + _flow_bonus(imbalance)
                ),
                relative_volume=relative_volume,
                taker_buy_imbalance=imbalance,
                flow_data_available=flow_available,
                explanation="Preço e médias apontam tendência de alta; o fluxo confirma continuação, não reversão.",
            )
        return PatternAssessment(
            regime="UPTREND",
            pattern="OBSERVATION_ONLY",
            confidence=_clamp(0.35 + min(relative_volume, 1.5) * 0.08),
            relative_volume=relative_volume,
            taker_buy_imbalance=imbalance,
            flow_data_available=flow_available,
            explanation="A tendência é positiva, mas o fluxo atual não confirma uma nova entrada.",
        )

    if latest_price < short_average < long_average:
        return PatternAssessment(
            regime="DOWNTREND",
            pattern="OBSERVATION_ONLY",
            confidence=_clamp(0.45 + min((long_average / latest_price - 1.0) * 15, 0.20)),
            relative_volume=relative_volume,
            taker_buy_imbalance=imbalance,
            flow_data_available=flow_available,
            explanation="Regime de baixa: Spot long-only não procura uma entrada contra a tendência.",
        )

    if lower_band_reentry and relative_volume >= 0.85 and buy_flow_ok:
        return PatternAssessment(
            regime="RANGE",
            pattern="MEAN_REVERSION",
            confidence=_clamp(0.38 + min(relative_volume, 2.0) * 0.12 + _flow_bonus(imbalance)),
            relative_volume=relative_volume,
            taker_buy_imbalance=imbalance,
            flow_data_available=flow_available,
            explanation="Mercado lateral reentrou na banda inferior com confirmação de volume comprador.",
        )

    return PatternAssessment(
        regime="RANGE",
        pattern="OBSERVATION_ONLY",
        confidence=_clamp(0.30 + min(relative_volume, 1.5) * 0.08),
        relative_volume=relative_volume,
        taker_buy_imbalance=imbalance,
        flow_data_available=flow_available,
        explanation="Não há padrão explicável com relação risco-retorno suficiente neste momento.",
    )


def _quote_volume(frame: pd.DataFrame, close: pd.Series) -> pd.Series:
    if "quote_volume" in frame:
        quote_volume = pd.to_numeric(frame["quote_volume"], errors="coerce")
        if quote_volume.notna().all() and (quote_volume > 0).all():
            return quote_volume
    return pd.to_numeric(frame["volume"], errors="coerce") * close


def _relative_volume(quote_volume: pd.Series) -> float:
    trailing = quote_volume.iloc[-(FLOW_LOOKBACK + 1) : -1]
    median = float(trailing.median()) if not trailing.empty else 0.0
    if not math.isfinite(median) or median <= 0:
        return 0.0
    return round(max(0.0, float(quote_volume.iloc[-1]) / median), 4)


def _taker_buy_imbalance(frame: pd.DataFrame, quote_volume: pd.Series) -> float | None:
    if "taker_buy_quote" not in frame:
        return None
    taker_buy_quote = pd.to_numeric(frame["taker_buy_quote"], errors="coerce")
    recent = taker_buy_quote.iloc[-FLOW_LOOKBACK:]
    if recent.isna().any() or not (recent > 0).any():
        return None
    current_quote = float(quote_volume.iloc[-1])
    current_taker_buy = float(taker_buy_quote.iloc[-1])
    if current_quote <= 0 or not math.isfinite(current_taker_buy):
        return None
    return round(_clamp(2 * current_taker_buy / current_quote - 1.0, lower=-1.0, upper=1.0), 4)


def _is_stress(returns: pd.Series, relative_volume: float) -> bool:
    current_volatility = float(returns.iloc[-24:].std(ddof=0))
    trailing_volatility = (
        returns.rolling(24).std(ddof=0).iloc[-COMPRESSION_LOOKBACK - 1 : -1].dropna()
    )
    if not math.isfinite(current_volatility) or len(trailing_volatility) < 30:
        return False
    stress_threshold = float(trailing_volatility.quantile(0.90)) * 1.25
    return current_volatility > stress_threshold and relative_volume >= 1.25


def _is_compression(close: pd.Series) -> bool:
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    width = (4 * deviation / middle.replace(0, np.nan)).dropna()
    history = width.iloc[-COMPRESSION_LOOKBACK - 1 : -1]
    return len(history) >= 30 and float(width.iloc[-1]) <= float(history.quantile(0.25))


def _is_upside_breakout(high: pd.Series, close: pd.Series) -> bool:
    previous_high = high.iloc[-(BREAKOUT_LOOKBACK + 1) : -1].max()
    return bool(
        math.isfinite(float(previous_high)) and float(close.iloc[-1]) > float(previous_high)
    )


def _is_lower_band_reentry(close: pd.Series) -> bool:
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    lower = middle - 2 * deviation
    return bool(close.iloc[-1] > lower.iloc[-1] and close.iloc[-2] <= lower.iloc[-2])


def _flow_bonus(imbalance: float | None) -> float:
    if imbalance is None:
        return 0.0
    return max(0.0, min(imbalance, 0.5)) * 0.18


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return round(max(lower, min(upper, value)), 4)

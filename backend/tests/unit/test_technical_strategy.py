"""Tests for reproducible technical-strategy research."""

import pandas as pd
import pytest
from pydantic import ValidationError

from app.api.v1.backtest import _backtest_response
from app.schemas.trading import StrategySummary, TechnicalStrategyConfig
from app.services.backtest.engine import BacktestEngine
from app.services.backtest.technical_strategy import build_technical_strategy_signals


def test_technical_strategy_emits_a_signal_only_on_a_new_crossover() -> None:
    candles = pd.DataFrame({"close": [10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12]})
    strategy = TechnicalStrategyConfig(
        kind="technical_ensemble",
        indicators=["sma"],
        indicator_params={"sma": {"sma_short": 2, "sma_long": 5}},
    )

    signals, definition = build_technical_strategy_signals(candles, strategy)

    assert definition.name == "Technical ensemble (Spot long-only)"
    assert signals.iloc[6] == 1.0
    assert signals.iloc[-1] == 0.0


def test_strategy_configuration_rejects_parameters_for_unselected_indicator() -> None:
    with pytest.raises(ValidationError, match="unselected indicator"):
        TechnicalStrategyConfig(
            kind="technical_ensemble",
            indicators=["sma"],
            indicator_params={"rsi": {"rsi_period": 14}},
        )


def test_engulfing_strategy_detects_bullish_and_bearish_reversals() -> None:
    candles = pd.DataFrame(
        {
            "open": [10.0, 7.0, 10.0, 13.0],
            "close": [8.0, 11.0, 12.0, 9.0],
        }
    )
    strategy = TechnicalStrategyConfig(
        kind="technical_ensemble",
        indicators=["engulfing"],
    )

    signals, _definition = build_technical_strategy_signals(candles, strategy)

    assert signals.tolist() == [0.0, 1.0, 0.0, -1.0]


def test_breakout_strategy_emits_only_on_new_range_break() -> None:
    candles = pd.DataFrame(
        {
            "high": [10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 12.0],
            "low": [9.0, 9.0, 9.0, 9.0, 9.0, 10.0, 11.0],
            "close": [9.5, 9.5, 9.5, 9.5, 9.5, 10.5, 11.5],
        }
    )
    strategy = TechnicalStrategyConfig(
        kind="technical_ensemble",
        indicators=["breakout"],
        indicator_params={"breakout": {"breakout_lookback": 5}},
    )

    signals, _definition = build_technical_strategy_signals(candles, strategy)

    assert signals.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_strategy_configuration_rejects_invalid_breakout_window() -> None:
    with pytest.raises(ValidationError, match="breakout_lookback"):
        TechnicalStrategyConfig(
            kind="technical_ensemble",
            indicators=["breakout"],
            indicator_params={"breakout": {"breakout_lookback": 4}},
        )


def test_spot_long_only_backtest_closes_on_sell_signal_without_opening_short() -> None:
    candles = pd.DataFrame(
        {
            "close": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "atr_14": [1.0] * 100,
        }
    )
    signals = pd.Series(0.0, index=candles.index)
    signals.iloc[60] = 1.0
    signals.iloc[61] = -1.0

    result = BacktestEngine(
        initial_capital=1_000,
        signal_threshold=0.3,
        allow_short=False,
    ).run(candles, signals=signals)

    assert len(result.trades) == 1
    assert result.trades[0].side == "LONG"
    assert result.trades[0].exit_reason == "signal"


def test_backtest_response_exposes_complete_metrics_and_strategy_audit() -> None:
    candles = pd.DataFrame(
        {
            "close": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "atr_14": [1.0] * 100,
        }
    )
    signals = pd.Series(0.0, index=candles.index)
    signals.iloc[60] = 1.0
    signals.iloc[61] = -1.0
    result = BacktestEngine(initial_capital=1_000, allow_short=False).run(
        candles, signals=signals
    )

    response = _backtest_response(
        result,
        result.equity_curve,
        StrategySummary(
            name="Technical ensemble (Spot long-only)",
            execution_profile="spot_long_only",
            indicators=["sma"],
            min_confirmations=1,
        ),
    )

    assert response.total_return == result.metrics.total_return
    assert response.max_drawdown_pct == result.metrics.max_drawdown_pct
    assert response.strategy.execution_profile == "spot_long_only"

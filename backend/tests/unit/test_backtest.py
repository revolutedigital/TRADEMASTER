"""Tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.engine import BacktestEngine
from app.services.portfolio.pnl import PnLCalculator


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for backtesting."""
    np.random.seed(42)
    n = 500
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000) + 100

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_backtest_runs_with_signals(sample_ohlcv):
    engine = BacktestEngine(initial_capital=10000)
    signals = pd.Series(np.random.uniform(-0.5, 0.5, len(sample_ohlcv)))

    result = engine.run(sample_ohlcv, signals=signals)

    assert result.metrics.total_trades >= 0
    assert len(result.equity_curve) > 0
    assert result.equity_curve[0] == 10000


def test_backtest_no_trades_with_weak_signals(sample_ohlcv):
    engine = BacktestEngine(initial_capital=10000, signal_threshold=0.9)
    # All signals below threshold
    signals = pd.Series(np.zeros(len(sample_ohlcv)))

    result = engine.run(sample_ohlcv, signals=signals)
    assert result.metrics.total_trades == 0


def test_backtest_params_recorded(sample_ohlcv):
    engine = BacktestEngine(
        initial_capital=5000,
        signal_threshold=0.4,
        atr_stop_multiplier=3.0,
    )
    signals = pd.Series(np.random.uniform(-0.5, 0.5, len(sample_ohlcv)))
    result = engine.run(sample_ohlcv, signals=signals)

    assert result.params["initial_capital"] == 5000
    assert result.params["signal_threshold"] == 0.4
    assert result.params["atr_stop_multiplier"] == 3.0


def test_backtest_requires_model_or_signals(sample_ohlcv):
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="model or signals"):
        engine.run(sample_ohlcv)


def test_completed_candle_signal_executes_at_the_following_open():
    candles = pd.DataFrame(
        {
            "open": [100.0] * 61 + [120.0] * 4,
            "high": [101.0] * 61 + [121.0] * 4,
            "low": [99.0] * 61 + [119.0] * 4,
            "close": [100.0] * 61 + [120.0] * 4,
            "atr_14": [1.0] * 65,
        }
    )
    signals = pd.Series(0.0, index=candles.index)
    signals.iloc[60] = 1.0

    result = BacktestEngine(
        initial_capital=1_000,
        signal_threshold=0.3,
        allow_short=False,
    ).run(candles, signals=signals)

    assert len(result.trades) == 1
    assert result.trades[0].entry_idx == 61
    assert result.trades[0].entry_price == pytest.approx(120.0 * 1.0005)
    assert result.params["signal_execution_lag_bars"] == 1
    assert result.params["signal_execution_price"] == "next_open"


# ---- PnL Calculator ----

def test_pnl_metrics_winning_trades():
    pnls = [100, 200, -50, 150, -30, 80, -40, 120]
    calc = PnLCalculator()
    metrics = calc.calculate_metrics(pnls, initial_equity=10000)

    assert metrics.total_trades == 8
    assert metrics.winning_trades == 5
    assert metrics.losing_trades == 3
    assert metrics.win_rate == 5 / 8
    assert metrics.total_return == sum(pnls)
    assert metrics.profit_factor > 1.0
    assert metrics.max_drawdown >= 0


def test_pnl_metrics_all_losses():
    pnls = [-100, -200, -50]
    calc = PnLCalculator()
    metrics = calc.calculate_metrics(pnls, initial_equity=10000)

    assert metrics.win_rate == 0
    assert metrics.total_return < 0
    assert metrics.profit_factor == 0


def test_pnl_metrics_empty():
    calc = PnLCalculator()
    metrics = calc.calculate_metrics([], initial_equity=10000)
    assert metrics.total_trades == 0
    assert metrics.total_return == 0

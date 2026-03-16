"""Walk-forward validation: rolling train/test windows to detect overfitting.

Instead of a single backtest, splits data into rolling windows:
  [Train 60 days | Test 15 days] → slide forward → repeat

Reports aggregate metrics across all test windows = realistic out-of-sample performance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.backtest.engine import BacktestEngine
from app.services.portfolio.pnl import PnLCalculator

logger = get_logger(__name__)

# 15m candles per day = 96
CANDLES_PER_DAY_15M = 96


@dataclass
class WalkForwardWindow:
    """Result of a single walk-forward window."""

    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_trades: int
    test_trades: int
    train_return_pct: float
    train_sharpe: float
    test_win_rate: float
    test_return_pct: float
    test_sharpe: float
    test_max_dd_pct: float
    test_profit_factor: float


@dataclass
class WalkForwardResult:
    """Aggregate result of walk-forward validation."""

    windows: list[WalkForwardWindow]
    total_test_trades: int
    avg_win_rate: float
    avg_return_pct: float
    avg_sharpe: float
    avg_max_dd_pct: float
    avg_profit_factor: float
    consistency_score: float  # % of windows that were profitable
    overfitting_score: float  # Train performance vs test performance ratio


def run_walk_forward(
    df: pd.DataFrame,
    signals: pd.Series,
    train_days: int = 60,
    test_days: int = 15,
    step_days: int = 15,
    initial_capital: float = 10000.0,
    signal_threshold: float = 0.3,
) -> WalkForwardResult:
    """Run walk-forward validation on historical data with pre-computed signals.

    Args:
        df: OHLCV DataFrame
        signals: Pre-computed signal series (same index as df)
        train_days: Training window in days
        test_days: Test window in days
        step_days: Step size between windows
        initial_capital: Starting capital per window
        signal_threshold: Minimum signal strength to trade
    """
    train_candles = train_days * CANDLES_PER_DAY_15M
    test_candles = test_days * CANDLES_PER_DAY_15M
    step_candles = step_days * CANDLES_PER_DAY_15M

    total_candles = len(df)
    if total_candles < train_candles + test_candles:
        logger.warning(
            "walk_forward_insufficient_data",
            total=total_candles,
            needed=train_candles + test_candles,
        )
        return WalkForwardResult(
            windows=[], total_test_trades=0,
            avg_win_rate=0, avg_return_pct=0, avg_sharpe=0,
            avg_max_dd_pct=0, avg_profit_factor=0,
            consistency_score=0, overfitting_score=0,
        )

    windows: list[WalkForwardWindow] = []
    pnl_calc = PnLCalculator()

    start = 0
    window_idx = 0

    while start + train_candles + test_candles <= total_candles:
        train_start = start
        train_end = start + train_candles
        test_start = train_end
        test_end = min(train_end + test_candles, total_candles)

        # Run backtest on training window
        train_df = df.iloc[train_start:train_end].reset_index(drop=True)
        train_signals = signals.iloc[train_start:train_end].reset_index(drop=True)

        train_engine = BacktestEngine(
            initial_capital=initial_capital,
            signal_threshold=signal_threshold,
        )
        train_result = train_engine.run(train_df, signals=train_signals)

        # Run backtest on test window (out-of-sample)
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)
        test_signals = signals.iloc[test_start:test_end].reset_index(drop=True)

        test_engine = BacktestEngine(
            initial_capital=initial_capital,
            signal_threshold=signal_threshold,
        )
        test_result = test_engine.run(test_df, signals=test_signals)

        window = WalkForwardWindow(
            window_idx=window_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_trades=train_result.metrics.total_trades,
            test_trades=test_result.metrics.total_trades,
            train_return_pct=train_result.metrics.total_return_pct,
            train_sharpe=train_result.metrics.sharpe_ratio,
            test_win_rate=test_result.metrics.win_rate,
            test_return_pct=test_result.metrics.total_return_pct,
            test_sharpe=test_result.metrics.sharpe_ratio,
            test_max_dd_pct=test_result.metrics.max_drawdown_pct,
            test_profit_factor=test_result.metrics.profit_factor,
        )
        windows.append(window)

        logger.info(
            "walk_forward_window",
            idx=window_idx,
            test_trades=test_result.metrics.total_trades,
            test_return=round(test_result.metrics.total_return_pct * 100, 2),
            test_sharpe=round(test_result.metrics.sharpe_ratio, 2),
        )

        start += step_candles
        window_idx += 1

    if not windows:
        return WalkForwardResult(
            windows=[], total_test_trades=0,
            avg_win_rate=0, avg_return_pct=0, avg_sharpe=0,
            avg_max_dd_pct=0, avg_profit_factor=0,
            consistency_score=0, overfitting_score=0,
        )

    # Aggregate metrics across all test windows
    total_test_trades = sum(w.test_trades for w in windows)
    traded_windows = [w for w in windows if w.test_trades > 0]

    if not traded_windows:
        return WalkForwardResult(
            windows=windows, total_test_trades=0,
            avg_win_rate=0, avg_return_pct=0, avg_sharpe=0,
            avg_max_dd_pct=0, avg_profit_factor=0,
            consistency_score=0, overfitting_score=0,
        )

    avg_win_rate = float(np.mean([w.test_win_rate for w in traded_windows]))
    avg_return = float(np.mean([w.test_return_pct for w in traded_windows]))
    avg_sharpe = float(np.mean([w.test_sharpe for w in traded_windows]))
    avg_dd = float(np.mean([w.test_max_dd_pct for w in traded_windows]))
    avg_pf = float(np.mean([w.test_profit_factor for w in traded_windows]))

    profitable_windows = sum(1 for w in traded_windows if w.test_return_pct > 0)
    consistency = profitable_windows / len(traded_windows)

    # Overfitting score: measures Sharpe ratio degradation from in-sample
    # to out-of-sample.  Score near 0 = no overfitting, near/above 1 = severe.
    # Formula: 1 - (avg OOS Sharpe / avg IS Sharpe)
    is_sharpes = [w.train_sharpe for w in traded_windows if w.train_sharpe > 0]
    oos_sharpes = [w.test_sharpe for w in traded_windows if w.train_sharpe > 0]
    avg_is_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    if avg_is_sharpe > 0:
        overfitting = max(0.0, min(1.0, 1.0 - (avg_oos_sharpe / avg_is_sharpe)))
    else:
        overfitting = 1.0 if avg_oos_sharpe <= 0 else 0.0

    result = WalkForwardResult(
        windows=windows,
        total_test_trades=total_test_trades,
        avg_win_rate=round(avg_win_rate, 4),
        avg_return_pct=round(avg_return, 4),
        avg_sharpe=round(avg_sharpe, 4),
        avg_max_dd_pct=round(avg_dd, 4),
        avg_profit_factor=round(avg_pf, 4),
        consistency_score=round(consistency, 4),
        overfitting_score=round(overfitting, 4),
    )

    logger.info(
        "walk_forward_complete",
        windows=len(windows),
        total_test_trades=total_test_trades,
        avg_return_pct=round(avg_return * 100, 2),
        consistency=round(consistency * 100, 1),
    )

    return result

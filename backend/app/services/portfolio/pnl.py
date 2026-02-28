"""P&L and performance metrics calculation."""

from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive trading performance metrics."""

    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    expectancy: float  # avg profit per trade


class PnLCalculator:
    """Calculates performance metrics from trade history."""

    def calculate_metrics(
        self,
        pnl_series: list[float],
        initial_equity: float,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics from a list of trade P&Ls.

        Args:
            pnl_series: List of individual trade P&Ls (positive = profit).
            initial_equity: Starting capital.
            risk_free_rate: Annual risk-free rate (default 0).
        """
        if not pnl_series:
            return self._empty_metrics()

        pnls = np.array(pnl_series)
        total_pnl = float(pnls.sum())

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = float(wins.mean()) if len(wins) > 0 else 0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve for drawdown
        equity_curve = np.cumsum(pnls) + initial_equity
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        max_dd = float(drawdown.max())
        max_dd_pct = max_dd / float(peak[drawdown.argmax()]) if peak[drawdown.argmax()] > 0 else 0

        # Returns for Sharpe/Sortino
        returns = pnls / initial_equity
        avg_return = float(returns.mean())
        std_return = float(returns.std()) if len(returns) > 1 else 0

        # Annualize (assume ~252 trading days)
        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        # Sortino (only downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = float(negative_returns.std()) if len(negative_returns) > 1 else 0
        sortino = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Calmar ratio
        annual_return = total_pnl / initial_equity
        calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0

        # Expectancy
        expectancy = float(pnls.mean())

        return PerformanceMetrics(
            total_return=total_pnl,
            total_return_pct=total_pnl / initial_equity if initial_equity > 0 else 0,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=float(calmar),
            expectancy=expectancy,
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=0, total_return_pct=0, total_trades=0,
            winning_trades=0, losing_trades=0, win_rate=0,
            avg_win=0, avg_loss=0, profit_factor=0,
            sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, max_drawdown_pct=0,
            calmar_ratio=0, expectancy=0,
        )


pnl_calculator = PnLCalculator()

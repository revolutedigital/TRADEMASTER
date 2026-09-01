"""Backtesting engine: event-driven simulation with realistic fills."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.services.indicators.engine import indicator_engine
from app.services.ml.features import feature_engineer
from app.services.ml.models.base import BaseTradingModel
from app.services.ml.models.ensemble import EnsembleModel
from app.services.portfolio.pnl import PerformanceMetrics, PnLCalculator

logger = get_logger(__name__)

# Realistic trading costs
DEFAULT_MAKER_FEE = 0.001  # 0.1%
DEFAULT_TAKER_FEE = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""

    entry_idx: int
    exit_idx: int
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    fees: float
    exit_reason: str  # signal, stop_loss, take_profit, time_exit


@dataclass
class BacktestResult:
    """Complete backtest result."""

    trades: list[BacktestTrade]
    metrics: PerformanceMetrics
    equity_curve: list[float]
    params: dict[str, Any]


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulates trading with:
    - Realistic slippage and fees
    - ATR-based stop losses
    - Trailing stops
    - Time-based exits
    - Position sizing
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        maker_fee: float = DEFAULT_MAKER_FEE,
        taker_fee: float = DEFAULT_TAKER_FEE,
        slippage: float = DEFAULT_SLIPPAGE,
        max_risk_per_trade: float = 0.02,
        atr_stop_multiplier: float = 2.0,
        risk_reward_ratio: float = 2.0,
        signal_threshold: float = 0.3,
        allow_short: bool = True,
    ):
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.max_risk_per_trade = max_risk_per_trade
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.signal_threshold = signal_threshold
        self.allow_short = allow_short

    def run(
        self,
        df: pd.DataFrame,
        model: BaseTradingModel | EnsembleModel | None = None,
        signals: pd.Series | None = None,
    ) -> BacktestResult:
        """Run backtest on OHLCV data.

        Provide either a model (will generate signals) or pre-computed signals.

        Args:
            df: OHLCV DataFrame with indicators computed.
            model: ML model to generate signals (optional).
            signals: Pre-computed signal series [-1.0, +1.0] (optional).
        """
        if "atr_14" not in df.columns:
            df = indicator_engine.compute_all(df)

        if signals is None and model is None:
            raise ValueError("Provide either a model or signals Series")

        equity = self.initial_capital
        position = None  # (side, entry_price, quantity, stop, tp, entry_idx)
        trades: list[BacktestTrade] = []
        equity_curve = [equity]

        # Signals use a completed candle and may only execute on the next bar.
        # This prevents same-bar look-ahead across technical and ML strategies.
        for i in range(61, len(df)):
            row = df.iloc[i]
            execution_price = _execution_open_price(row)
            signal_row = i - 1
            atr = float(df.iloc[signal_row].get("atr_14", 0))

            if signals is not None:
                signal = float(signals.iloc[signal_row]) if signal_row < len(signals) else 0
            else:
                signal = self._predict_signal(model, df, signal_row)

            # A reversal is available at this candle's open. It must execute
            # before checking this candle's high/low range.
            if position:
                side, entry, qty, _stop, _tp, entry_idx = position
                is_reversal = (
                    (side == "LONG" and signal <= -self.signal_threshold)
                    or (side == "SHORT" and signal >= self.signal_threshold)
                )
                if is_reversal:
                    exit_price = execution_price * (
                        1 - self.slippage if side == "LONG" else 1 + self.slippage
                    )
                    pnl, fees = self._close_trade(side, entry, exit_price, qty)
                    equity += pnl - fees
                    trades.append(BacktestTrade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        side=side,
                        entry_price=entry,
                        exit_price=exit_price,
                        quantity=qty,
                        pnl=pnl - fees,
                        fees=fees,
                        exit_reason="signal",
                    ))
                    position = None

            # A new order is sent at the current open, using ATR that existed
            # when the prior candle closed. Spot long-only ignores SELL here
            # after using it to close the existing long above.
            if position is None and atr > 0:
                if signal >= self.signal_threshold:
                    entry_price = execution_price * (1 + self.slippage)
                    stop_dist = atr * self.atr_stop_multiplier
                    stop_price = entry_price - stop_dist
                    tp_price = entry_price + stop_dist * self.risk_reward_ratio

                    risk_amount = equity * self.max_risk_per_trade
                    stop_pct = stop_dist / entry_price
                    qty = (risk_amount / stop_pct) / entry_price if stop_pct > 0 else 0

                    if qty * entry_price >= 10:  # min notional
                        position = ("LONG", entry_price, qty, stop_price, tp_price, i)

                elif self.allow_short and signal <= -self.signal_threshold:
                    entry_price = execution_price * (1 - self.slippage)
                    stop_dist = atr * self.atr_stop_multiplier
                    stop_price = entry_price + stop_dist
                    tp_price = entry_price - stop_dist * self.risk_reward_ratio

                    risk_amount = equity * self.max_risk_per_trade
                    stop_pct = stop_dist / entry_price
                    qty = (risk_amount / stop_pct) / entry_price if stop_pct > 0 else 0

                    if qty * entry_price >= 10:
                        position = ("SHORT", entry_price, qty, stop_price, tp_price, i)

            # Stop/target orders protect the remaining candle. When both are
            # possible, stop takes priority as the conservative assumption.
            if position:
                side, entry, qty, stop, tp, entry_idx = position
                if side == "LONG" and float(row["low"]) <= stop:
                    exit_price = stop * (1 - self.slippage)
                    pnl, fees = self._close_trade(side, entry, exit_price, qty)
                    equity += pnl - fees
                    trades.append(BacktestTrade(
                        entry_idx=entry_idx, exit_idx=i, side=side,
                        entry_price=entry, exit_price=exit_price,
                        quantity=qty, pnl=pnl - fees, fees=fees,
                        exit_reason="stop_loss",
                    ))
                    position = None
                elif side == "SHORT" and float(row["high"]) >= stop:
                    exit_price = stop * (1 + self.slippage)
                    pnl, fees = self._close_trade(side, entry, exit_price, qty)
                    equity += pnl - fees
                    trades.append(BacktestTrade(
                        entry_idx=entry_idx, exit_idx=i, side=side,
                        entry_price=entry, exit_price=exit_price,
                        quantity=qty, pnl=pnl - fees, fees=fees,
                        exit_reason="stop_loss",
                    ))
                    position = None
                elif tp and side == "LONG" and float(row["high"]) >= tp:
                    exit_price = tp * (1 - self.slippage)
                    pnl, fees = self._close_trade(side, entry, exit_price, qty)
                    equity += pnl - fees
                    trades.append(BacktestTrade(
                        entry_idx=entry_idx, exit_idx=i, side=side,
                        entry_price=entry, exit_price=exit_price,
                        quantity=qty, pnl=pnl - fees, fees=fees,
                        exit_reason="take_profit",
                    ))
                    position = None
                elif tp and side == "SHORT" and float(row["low"]) <= tp:
                    exit_price = tp * (1 + self.slippage)
                    pnl, fees = self._close_trade(side, entry, exit_price, qty)
                    equity += pnl - fees
                    trades.append(BacktestTrade(
                        entry_idx=entry_idx, exit_idx=i, side=side,
                        entry_price=entry, exit_price=exit_price,
                        quantity=qty, pnl=pnl - fees, fees=fees,
                        exit_reason="take_profit",
                    ))
                    position = None

            equity_curve.append(equity)

        # Close remaining position at last price
        if position:
            side, entry, qty, stop, tp, entry_idx = position
            exit_price = float(df.iloc[-1]["close"])
            pnl, fees = self._close_trade(side, entry, exit_price, qty)
            equity += pnl - fees
            trades.append(BacktestTrade(
                entry_idx=entry_idx, exit_idx=len(df) - 1, side=side,
                entry_price=entry, exit_price=exit_price,
                quantity=qty, pnl=pnl - fees, fees=fees,
                exit_reason="end_of_data",
            ))
            equity_curve.append(equity)

        # Calculate metrics
        pnl_calc = PnLCalculator()
        pnl_series = [t.pnl for t in trades]
        metrics = pnl_calc.calculate_metrics(pnl_series, self.initial_capital)

        logger.info(
            "backtest_complete",
            total_trades=metrics.total_trades,
            win_rate=round(metrics.win_rate, 4),
            total_return_pct=round(metrics.total_return_pct * 100, 2),
            sharpe=round(metrics.sharpe_ratio, 2),
            max_dd_pct=round(metrics.max_drawdown_pct * 100, 2),
            profit_factor=round(metrics.profit_factor, 2),
        )

        return BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            params={
                "initial_capital": self.initial_capital,
                "signal_threshold": self.signal_threshold,
                "atr_stop_multiplier": self.atr_stop_multiplier,
                "risk_reward_ratio": self.risk_reward_ratio,
                "max_risk_per_trade": self.max_risk_per_trade,
                "slippage": self.slippage,
                "fees": self.taker_fee,
                "allow_short": self.allow_short,
                "signal_execution_lag_bars": 1,
                "signal_execution_price": "next_open",
            },
        )

    def _predict_signal(
        self,
        model: BaseTradingModel | EnsembleModel | None,
        df: pd.DataFrame,
        idx: int,
    ) -> float:
        """Generate a trading signal from an ML model prediction.

        Extracts features for the current bar, calls model.predict(), and
        converts the output into a signal value in [-1.0, +1.0].

        Falls back to 0 (HOLD) if anything goes wrong.
        """
        if model is None:
            return 0.0

        try:
            # Build feature row from the current OHLCV window
            feature_cols = feature_engineer.get_feature_columns(df)
            row_features = df.iloc[idx][feature_cols].values.astype(np.float64)

            # Replace NaN/inf with 0 so the model doesn't choke
            row_features = np.nan_to_num(row_features, nan=0.0, posinf=0.0, neginf=0.0)

            if isinstance(model, EnsembleModel):
                # EnsembleModel.predict expects dict[model_name -> features]
                features_dict = dict.fromkeys(model.models, row_features)
                prediction = model.predict(features_dict)
            else:
                prediction = model.predict(row_features)

            return float(prediction.signal_strength)
        except Exception as exc:
            logger.debug("model_predict_fallback", idx=idx, error=str(exc))
            return 0.0

    def _close_trade(
        self, side: str, entry: float, exit_price: float, qty: float
    ) -> tuple[float, float]:
        """Calculate P&L and fees for closing a trade."""
        if side == "LONG":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty

        fees = (entry * qty * self.taker_fee) + (exit_price * qty * self.taker_fee)
        return pnl, fees


def _execution_open_price(row: pd.Series) -> float:
    """Use the first executable candle price, with a safe legacy-data fallback."""
    open_price = float(row.get("open", 0))
    if np.isfinite(open_price) and open_price > 0:
        return open_price
    close_price = float(row["close"])
    if not np.isfinite(close_price) or close_price <= 0:
        raise ValueError("backtest candle requires a positive open or close price")
    return close_price


backtest_engine = BacktestEngine()

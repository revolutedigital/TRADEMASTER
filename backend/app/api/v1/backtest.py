"""Backtest API endpoints."""

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.models.backtest import BacktestResult as BacktestResultModel
from app.schemas.trading import BacktestRequest, BacktestResponse

router = APIRouter()


class BacktestHistoryItem(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    symbol: str
    interval: str
    initial_capital: float
    signal_threshold: float
    total_trades: int
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    created_at: str


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    req: BacktestRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Run a backtest with ML model signals. Requires authentication."""
    from app.services.backtest.engine import BacktestEngine
    from app.services.market.data_collector import market_data_collector

    # Load historical data
    df = await market_data_collector.get_latest_candles(
        db=db,
        symbol=req.symbol,
        interval=req.interval,
        limit=5000,
    )

    if df.empty or len(df) < 200:
        return BacktestResponse(
            total_trades=0, win_rate=0, total_return_pct=0,
            sharpe_ratio=0, max_drawdown_pct=0, profit_factor=0,
            expectancy=0, equity_curve=[req.initial_capital],
        )

    # Generate signals using ML models
    signals = await _generate_ml_signals(df, req.symbol)

    engine = BacktestEngine(
        initial_capital=req.initial_capital,
        signal_threshold=req.signal_threshold,
        atr_stop_multiplier=req.atr_stop_multiplier,
        risk_reward_ratio=req.risk_reward_ratio,
    )
    result = engine.run(df, signals=signals)

    # Persist result
    equity_curve = result.equity_curve[-500:]
    db_result = BacktestResultModel(
        symbol=req.symbol,
        interval=req.interval,
        initial_capital=req.initial_capital,
        signal_threshold=req.signal_threshold,
        atr_stop_multiplier=req.atr_stop_multiplier,
        risk_reward_ratio=req.risk_reward_ratio,
        total_trades=result.metrics.total_trades,
        win_rate=result.metrics.win_rate,
        total_return_pct=result.metrics.total_return_pct,
        sharpe_ratio=result.metrics.sharpe_ratio,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        profit_factor=result.metrics.profit_factor,
        expectancy=result.metrics.expectancy,
        equity_curve_json=json.dumps(equity_curve),
    )
    db.add(db_result)
    await db.commit()

    return BacktestResponse(
        total_trades=result.metrics.total_trades,
        win_rate=result.metrics.win_rate,
        total_return_pct=result.metrics.total_return_pct,
        sharpe_ratio=result.metrics.sharpe_ratio,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        profit_factor=result.metrics.profit_factor,
        expectancy=result.metrics.expectancy,
        equity_curve=equity_curve,
    )


@router.post("/walk-forward")
async def run_walk_forward_validation(
    req: BacktestRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Run walk-forward validation with rolling train/test windows.

    Tests signal robustness by backtesting on multiple out-of-sample windows.
    """
    from app.services.backtest.walk_forward import run_walk_forward
    from app.services.market.data_collector import market_data_collector

    df = await market_data_collector.get_latest_candles(
        db=db, symbol=req.symbol, interval=req.interval, limit=5000,
    )

    if df.empty or len(df) < 500:
        return {
            "error": "Need at least 500 candles for walk-forward validation",
            "available": len(df) if not df.empty else 0,
        }

    signals = await _generate_ml_signals(df, req.symbol)

    result = run_walk_forward(
        df=df,
        signals=signals,
        train_days=60,
        test_days=15,
        step_days=15,
        initial_capital=req.initial_capital,
        signal_threshold=req.signal_threshold,
    )

    return {
        "total_windows": len(result.windows),
        "total_test_trades": result.total_test_trades,
        "avg_win_rate": result.avg_win_rate,
        "avg_return_pct": result.avg_return_pct,
        "avg_sharpe": result.avg_sharpe,
        "avg_max_dd_pct": result.avg_max_dd_pct,
        "avg_profit_factor": result.avg_profit_factor,
        "consistency_score": result.consistency_score,
        "windows": [
            {
                "idx": w.window_idx,
                "test_trades": w.test_trades,
                "test_win_rate": w.test_win_rate,
                "test_return_pct": w.test_return_pct,
                "test_sharpe": w.test_sharpe,
            }
            for w in result.windows
        ],
    }


@router.get("/history", response_model=list[BacktestHistoryItem])
async def backtest_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Get backtest run history, newest first."""
    query = (
        select(BacktestResultModel)
        .order_by(BacktestResultModel.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    rows = result.scalars().all()
    return [
        BacktestHistoryItem(
            id=r.id,
            symbol=r.symbol,
            interval=r.interval,
            initial_capital=float(r.initial_capital),
            signal_threshold=float(r.signal_threshold),
            total_trades=r.total_trades,
            win_rate=r.win_rate,
            total_return_pct=r.total_return_pct,
            sharpe_ratio=r.sharpe_ratio,
            max_drawdown_pct=r.max_drawdown_pct,
            profit_factor=r.profit_factor,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Get a specific backtest result by ID."""
    result = await db.execute(
        select(BacktestResultModel).where(BacktestResultModel.id == backtest_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Backtest not found")

    equity_curve = json.loads(row.equity_curve_json) if row.equity_curve_json else []
    return BacktestResponse(
        total_trades=row.total_trades,
        win_rate=row.win_rate,
        total_return_pct=row.total_return_pct,
        sharpe_ratio=row.sharpe_ratio,
        max_drawdown_pct=row.max_drawdown_pct,
        profit_factor=row.profit_factor,
        expectancy=row.expectancy,
        equity_curve=equity_curve,
    )


@router.post("/monte-carlo/{backtest_id}")
async def run_monte_carlo_validation(
    backtest_id: int,
    n_simulations: int = 1000,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Run Monte Carlo significance test on a completed backtest.

    Shuffles trade order N times to determine if results are statistically
    significant or just lucky ordering.
    """
    from app.services.backtest.monte_carlo import run_monte_carlo

    result = await db.execute(
        select(BacktestResultModel).where(BacktestResultModel.id == backtest_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Backtest not found")

    # Extract trade returns from equity curve
    equity_curve = json.loads(row.equity_curve_json) if row.equity_curve_json else []
    if len(equity_curve) < 5:
        raise HTTPException(400, "Backtest has too few trades for Monte Carlo analysis")

    # Compute per-step returns from equity curve
    eq = np.array(equity_curve, dtype=float)
    trade_returns = list(np.diff(eq) / eq[:-1])

    mc_result = run_monte_carlo(
        trade_returns=trade_returns,
        initial_capital=float(row.initial_capital),
        n_simulations=min(n_simulations, 10000),
    )

    return {
        "backtest_id": backtest_id,
        "original_return_pct": mc_result.original_return_pct,
        "original_sharpe": mc_result.original_sharpe,
        "original_max_dd_pct": mc_result.original_max_dd_pct,
        "p_value_return": mc_result.p_value_return,
        "p_value_sharpe": mc_result.p_value_sharpe,
        "is_significant_95": mc_result.is_significant_95,
        "is_significant_99": mc_result.is_significant_99,
        "sim_mean_return": mc_result.sim_mean_return,
        "sim_std_return": mc_result.sim_std_return,
        "confidence_intervals": {
            "ci_90": mc_result.ci_90,
            "ci_95": mc_result.ci_95,
            "ci_99": mc_result.ci_99,
        },
        "worst_case_return_pct": mc_result.worst_return_pct,
        "best_case_return_pct": mc_result.best_return_pct,
        "worst_max_drawdown_pct": mc_result.worst_max_dd_pct,
        "n_simulations": mc_result.n_simulations,
        "n_trades": mc_result.n_trades,
    }


async def _generate_ml_signals(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Generate signals for backtest using ML models or technical signal fallback.

    Priority:
    1. ML ensemble (if models loaded)
    2. Technical multi-indicator signal (same as live trading engine)
    """
    from app.services.ml.pipeline import MLPipeline
    from app.services.ml.features import feature_engineer

    # Try to load ML pipeline
    pipeline = MLPipeline()
    await pipeline.load_models(symbol)

    if pipeline._ensemble is not None:
        # ML models available — use them
        signals = pd.Series(0.0, index=df.index)
        window_size = 200
        for i in range(window_size, len(df)):
            window_df = df.iloc[max(0, i - 300):i + 1]
            try:
                prediction = await pipeline.predict(window_df, symbol)
                if prediction is not None:
                    signals.iloc[i] = prediction.signal_strength
            except Exception:
                continue
        return signals

    # No ML models — use the same multi-indicator technical signal as live engine
    return _generate_technical_signals(df, symbol)


def _generate_technical_signals(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Generate signals using the same multi-indicator system as the live trading engine.

    Indicators: SMA(10/30) crossover, RSI(14), MACD(12,26,9), Bollinger Bands(20,2)
    Filters: Trend (SMA 50), Volatility (ATR > 0.3%)
    """
    from app.services.trading_engine import TradingEngine

    signals = pd.Series(0.0, index=df.index)
    engine = TradingEngine()

    min_candles = 30  # Same as MIN_CANDLES_FOR_SIGNAL in trading engine

    for i in range(min_candles, len(df)):
        window = df.iloc[max(0, i - 300):i + 1]
        if len(window) < min_candles:
            continue

        prediction = engine._technical_signal(window, symbol)
        if prediction is not None:
            signals.iloc[i] = prediction.signal_strength

    return signals

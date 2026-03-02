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
from app.services.backtest.engine import BacktestEngine
from app.services.market.data_collector import market_data_collector
from app.services.ml.features import feature_engineer
from app.services.ml.pipeline import MLPipeline

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


async def _generate_ml_signals(df: pd.DataFrame, symbol: str) -> pd.Series:
    """Generate signals for backtest using loaded ML models.

    Runs inference on each row using a sliding window approach.
    Falls back to feature-based heuristic if models are not available.
    """
    # Try to load ML pipeline
    pipeline = MLPipeline()
    await pipeline.load_models(symbol)

    # Build features
    df_features = feature_engineer.build_features(df)

    if pipeline._ensemble is None:
        # No models available - use simple momentum heuristic as fallback
        signals = pd.Series(0.0, index=df.index)
        if "rsi_14" in df_features.columns and "macd" in df_features.columns:
            rsi = df_features["rsi_14"].fillna(50)
            macd = df_features["macd"].fillna(0)
            signals = ((50 - rsi) / 100 + macd / df_features["close"].abs().clip(lower=1)) / 2
            signals = signals.clip(-1, 1).fillna(0)
        return signals

    # Run ML inference for each valid window
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

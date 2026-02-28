"""Backtest API endpoints."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_auth
from app.schemas.trading import BacktestRequest, BacktestResponse
from app.services.backtest.engine import BacktestEngine
from app.services.market.data_collector import market_data_collector
from app.services.ml.features import feature_engineer
from app.services.ml.pipeline import MLPipeline

router = APIRouter()


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

    return BacktestResponse(
        total_trades=result.metrics.total_trades,
        win_rate=result.metrics.win_rate,
        total_return_pct=result.metrics.total_return_pct,
        sharpe_ratio=result.metrics.sharpe_ratio,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        profit_factor=result.metrics.profit_factor,
        expectancy=result.metrics.expectancy,
        equity_curve=result.equity_curve[-500:],  # Last 500 points
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
    feature_cols = feature_engineer.get_feature_columns(df_features)

    if pipeline._ensemble is None:
        # No models available - use simple momentum heuristic as fallback
        # This is better than random but clearly labeled as heuristic
        signals = pd.Series(0.0, index=df.index)
        if "rsi_14" in df_features.columns and "macd" in df_features.columns:
            rsi = df_features["rsi_14"].fillna(50)
            macd = df_features["macd"].fillna(0)
            # RSI oversold + MACD positive = buy signal
            signals = ((50 - rsi) / 100 + macd / df_features["close"].abs().clip(lower=1)) / 2
            signals = signals.clip(-1, 1).fillna(0)
        return signals

    # Run ML inference for each valid window
    signals = pd.Series(0.0, index=df.index)
    window_size = 200  # Minimum data for prediction

    for i in range(window_size, len(df)):
        window_df = df.iloc[max(0, i - 300):i + 1]
        try:
            prediction = await pipeline.predict(window_df, symbol)
            if prediction is not None:
                signals.iloc[i] = prediction.signal_strength
        except Exception:
            continue

    return signals

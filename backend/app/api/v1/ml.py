"""ML model management, explainability, and tracking endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db, require_auth
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/feature-importance/{symbol}")
async def get_feature_importance(symbol: str, _user: dict = Depends(require_auth)):
    """Get feature importance for a symbol's ML models."""
    symbol = symbol.upper()
    if symbol not in settings.symbols_list:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not configured")

    from app.services.ml.pipeline import ml_pipeline

    importance = {}

    # Try XGBoost importance
    models = getattr(ml_pipeline, '_models', {})
    symbol_models = models.get(symbol, {})

    if 'xgboost' in symbol_models:
        xgb = symbol_models['xgboost']
        if hasattr(xgb, '_model') and xgb._model is not None:
            try:
                booster = xgb._model.get_booster()
                raw = booster.get_score(importance_type='gain')
                total = sum(raw.values()) or 1
                importance['xgboost'] = {k: round(v / total, 4) for k, v in
                                         sorted(raw.items(), key=lambda x: x[1], reverse=True)[:15]}
            except Exception as e:
                importance['xgboost'] = {"error": str(e)}

    return {
        "symbol": symbol,
        "feature_importance": importance,
        "models_loaded": list(symbol_models.keys()) if symbol_models else [],
    }


@router.get("/calibration/{symbol}")
async def get_calibration_stats(symbol: str, _user: dict = Depends(require_auth)):
    """Get confidence calibration statistics for a symbol."""
    from app.services.ml.calibration import get_calibrator

    calibrator = get_calibrator(symbol.upper())
    return {
        "symbol": symbol.upper(),
        "calibration_stats": calibrator.get_calibration_stats(),
    }


@router.get("/ensemble/weights")
async def get_ensemble_weights(_user: dict = Depends(require_auth)):
    """Get current ensemble model weights."""
    from app.services.ml.ensemble import ensemble_voter
    return {
        "weights": ensemble_voter._weights,
        "default_weights": ensemble_voter.DEFAULT_WEIGHTS,
    }


@router.get("/drift/{symbol}")
async def get_drift_status(symbol: str, _user: dict = Depends(require_auth)):
    """Get model drift detection status for a symbol."""
    symbol = symbol.upper()
    try:
        from app.services.ml.drift_detector import drift_detector
        status = drift_detector.get_status()
        return {
            "symbol": symbol,
            "drift_status": status,
        }
    except Exception as e:
        return {"symbol": symbol, "drift_status": {"error": str(e)}}


@router.get("/models")
async def list_models(_user: dict = Depends(require_auth)):
    """List all loaded ML models and their status."""
    from app.services.ml.pipeline import ml_pipeline

    models = getattr(ml_pipeline, '_models', {})
    result = {}

    for symbol, symbol_models in models.items():
        result[symbol] = {
            model_type: {
                "loaded": True,
                "type": type(model).__name__,
            }
            for model_type, model in symbol_models.items()
        }

    return {"models": result, "total_symbols": len(result)}


@router.get("/transformer/{symbol}")
async def get_transformer_prediction(symbol: str, _user: dict = Depends(require_auth)):
    """Get multi-horizon prediction from transformer model."""
    symbol = symbol.upper()
    from app.services.ml.models.transformer_model import transformer_predictor
    from app.services.ml.features import feature_engineer
    from app.models.base import async_session_factory
    from app.services.market.data_collector import market_data_collector

    async with async_session_factory() as db:
        df = await market_data_collector.get_latest_candles(db=db, symbol=symbol, interval="1h", limit=100)

    if df.empty or len(df) < 30:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction")

    df_feat = feature_engineer.build_features(df)
    feature_cols = feature_engineer.get_feature_columns(df_feat)
    features = df_feat[feature_cols].dropna().values[-60:]

    result = transformer_predictor.predict(features)
    return {
        "symbol": symbol,
        "horizons": result.horizons,
        "predictions": result.predictions,
        "confidence": result.confidence,
        "attention_top_5": sorted(
            enumerate(result.attention_weights), key=lambda x: x[1], reverse=True
        )[:5],
    }


@router.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str, _user: dict = Depends(require_auth)):
    """Get market sentiment analysis for a symbol."""
    symbol = symbol.upper()
    from app.services.ml.sentiment import sentiment_analyzer
    from app.models.base import async_session_factory
    from app.services.market.data_collector import market_data_collector

    async with async_session_factory() as db:
        df = await market_data_collector.get_latest_candles(db=db, symbol=symbol, interval="1h", limit=100)

    if df.empty or len(df) < 20:
        raise HTTPException(status_code=400, detail="Insufficient data for sentiment analysis")

    prices = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    result = sentiment_analyzer.analyze_from_market_data(prices, volumes)
    return {
        "symbol": symbol,
        "overall": result.overall,
        "interpretation": result.interpretation,
        "confidence": result.confidence,
        "sources": result.sources,
        "timestamp": result.timestamp,
    }


@router.get("/synthetic-scenarios")
async def list_synthetic_scenarios(_user: dict = Depends(require_auth)):
    """List available synthetic data scenarios for training augmentation."""
    from app.services.ml.synthetic_data import synthetic_generator
    scenarios = synthetic_generator.generate_all_scenarios()
    return {
        "scenarios": [
            {"name": s.name, "description": s.description, "candles": s.candles}
            for s in scenarios
        ]
    }


# ========================================
# ML Tracking endpoints
# ========================================


@router.get("/tracking/runs")
async def list_training_runs(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """List ML training runs with optional filters."""
    from app.services.ml.tracking import ml_tracker

    runs = await ml_tracker.get_training_runs(
        db, model_type=model_type, symbol=symbol, limit=limit, offset=offset
    )
    return {"runs": runs, "count": len(runs), "limit": limit, "offset": offset}


@router.get("/tracking/accuracy/{symbol}")
async def get_rolling_accuracy(
    symbol: str,
    model_type: Optional[str] = Query(None, description="Model type (default: ensemble)"),
    window: int = Query(100, ge=10, le=1000),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Get rolling prediction accuracy for a symbol."""
    from app.services.ml.tracking import ml_tracker

    symbol = symbol.upper()
    if symbol not in settings.symbols_list:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not configured")

    mt = model_type or "ensemble"
    accuracy = await ml_tracker.get_rolling_accuracy(db, mt, symbol, window=window)

    return {
        "symbol": symbol,
        "model_type": mt,
        "window": window,
        "rolling_accuracy": round(accuracy, 4),
    }


@router.get("/tracking/comparison/{symbol}")
async def get_model_comparison(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Compare all model types for a symbol by accuracy, confidence, and latency."""
    from app.services.ml.tracking import ml_tracker

    symbol = symbol.upper()
    if symbol not in settings.symbols_list:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not configured")

    comparisons = await ml_tracker.get_model_comparison(db, symbol)
    return {"symbol": symbol, "models": comparisons}


@router.post("/tracking/predictions/{prediction_id}/outcome")
async def update_prediction_outcome(
    prediction_id: int,
    actual_outcome: str = Query(..., description="Actual outcome: BUY, SELL, or HOLD"),
    outcome_pnl: Optional[float] = Query(None, description="Realized P&L"),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Update a prediction with its actual outcome for accuracy tracking."""
    from app.services.ml.tracking import ml_tracker

    actual_outcome = actual_outcome.upper()
    if actual_outcome not in ("BUY", "SELL", "HOLD"):
        raise HTTPException(status_code=400, detail="outcome must be BUY, SELL, or HOLD")

    updated = await ml_tracker.update_prediction_outcome(
        db, prediction_id, actual_outcome, outcome_pnl=outcome_pnl
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return {"prediction_id": prediction_id, "actual_outcome": actual_outcome, "updated": True}

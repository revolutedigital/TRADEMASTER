"""ML model management and explainability endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from app.config import settings
from app.dependencies import require_auth
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

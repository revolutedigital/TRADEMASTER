"""ML/AI model management and monitoring endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.dependencies import get_db

logger = get_logger(__name__)
router = APIRouter()


@router.get("/models")
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all registered ML models and their metadata."""
    try:
        from app.services.ml.model_registry import model_registry
        models = await model_registry.list_models(db)
        return {"models": models}
    except Exception as e:
        logger.warning("list_models_failed", error=str(e))
        return {"models": []}


@router.get("/feature-importance/{symbol}")
async def get_feature_importance(symbol: str):
    """Get feature importance rankings for a symbol's model."""
    try:
        from app.services.ml.explainability import model_explainer
        result = model_explainer.get_feature_importance(symbol.upper())
        return result
    except Exception as e:
        logger.warning("feature_importance_failed", symbol=symbol, error=str(e))
        return {
            "symbol": symbol.upper(),
            "features": [],
            "method": "unavailable",
            "error": str(e),
        }


@router.get("/drift/{symbol}")
async def get_model_drift(symbol: str):
    """Check for feature drift in a symbol's model."""
    try:
        from app.services.ml.pipeline import prediction_pipeline
        drift_status = prediction_pipeline.check_drift(symbol.upper())
        return drift_status
    except Exception as e:
        logger.warning("drift_check_failed", symbol=symbol, error=str(e))
        return {
            "symbol": symbol.upper(),
            "drift_detected": False,
            "status": "check_unavailable",
        }


@router.get("/performance")
async def get_model_performance():
    """Get overall ML model performance metrics."""
    try:
        from app.core.metrics import metrics
        return {
            "inference_count": metrics.signals_generated._value if hasattr(metrics.signals_generated, '_value') else 0,
            "status": "operational",
        }
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


@router.get("/automl/history")
async def get_automl_history():
    """Get AutoML evaluation history."""
    try:
        from app.services.ml.automl import auto_model_selector
        return {"history": auto_model_selector.get_history()}
    except Exception as e:
        return {"history": [], "error": str(e)}

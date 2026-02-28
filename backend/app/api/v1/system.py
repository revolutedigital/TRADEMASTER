"""System endpoints: health check, status, metrics, initialization."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.dependencies import require_auth

logger = get_logger(__name__)

router = APIRouter()

# Track initialization state
_init_status = {"seeding": False, "training": False, "error": None}


@router.get("/health")
async def health_check():
    from app.services.exchange.binance_client import binance_client
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "env": settings.app_env,
        "testnet": settings.binance_testnet,
        "binance_connected": binance_client._client is not None,
    }


@router.get("/status")
async def system_status():
    from app.services.exchange.binance_client import binance_client
    from app.services.trading_engine import trading_engine

    return {
        "trading_symbols": settings.symbols_list,
        "risk_limits": {
            "max_risk_per_trade": settings.trading_max_risk_per_trade,
            "max_portfolio_exposure": settings.trading_max_portfolio_exposure,
            "max_daily_drawdown": settings.trading_max_daily_drawdown,
        },
        "engine_running": trading_engine._running,
        "binance_connected": binance_client._client is not None,
    }


@router.post("/connect-binance")
async def connect_binance(_user: dict = Depends(require_auth)):
    """Try to (re)connect to Binance API. Requires authentication."""
    from app.services.exchange.binance_client import binance_client
    try:
        await binance_client.connect()
        return {"status": "connected", "testnet": settings.binance_testnet}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@router.post("/start-services")
async def start_all_services(_user: dict = Depends(require_auth)):
    """Start all background services. Requires authentication."""
    from app.services.exchange.binance_client import binance_client
    if binance_client._client is None:
        return {"status": "error", "detail": "Binance not connected. Call /connect-binance first."}

    from app.main import _start_background_services
    try:
        await _start_background_services()
        return {"status": "started"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    return metrics.collect()


@router.post("/initialize")
async def initialize_system(
    background_tasks: BackgroundTasks,
    intervals: str = "1h,4h",
    days_back: int = 90,
    _user: dict = Depends(require_auth),
):
    """Seed historical data and train ML models.

    This runs as a background task - check /api/v1/system/init-status for progress.

    Args:
        intervals: Comma-separated intervals to seed (default: 1h,4h)
        days_back: How many days of history to download (default: 90)
    """
    if _init_status["seeding"] or _init_status["training"]:
        return {"status": "already_running", "detail": _init_status}

    _init_status["error"] = None
    background_tasks.add_task(_run_initialization, intervals, days_back)
    return {"status": "started", "intervals": intervals, "days_back": days_back}


@router.get("/init-status")
async def init_status():
    """Check initialization progress."""
    return _init_status


async def _run_initialization(intervals: str, days_back: int) -> None:
    """Background task: seed historical data then train models."""
    interval_list = [i.strip() for i in intervals.split(",")]

    # Phase 1: Seed historical data
    _init_status["seeding"] = True
    try:
        from app.models.base import async_session_factory, engine, Base
        from app.services.exchange.binance_client import binance_client
        from app.services.market.data_collector import market_data_collector

        # Ensure tables exist
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        for symbol in settings.symbols_list:
            for interval in interval_list:
                logger.info("seeding_data", symbol=symbol, interval=interval, days=days_back)
                async with async_session_factory() as db:
                    count = await market_data_collector.seed_historical(
                        db=db, symbol=symbol, interval=interval, days_back=days_back,
                    )
                    logger.info("data_seeded", symbol=symbol, interval=interval, candles=count)

        _init_status["seeding"] = False
        logger.info("seed_phase_complete")
    except Exception as e:
        _init_status["seeding"] = False
        _init_status["error"] = f"Seed failed: {e}"
        logger.error("seed_failed", error=str(e))
        return

    # Phase 2: Train models
    _init_status["training"] = True
    try:
        from app.models.base import async_session_factory
        from app.services.market.data_collector import market_data_collector
        from app.services.ml.features import feature_engineer
        from app.services.ml.models.lstm_model import LSTMTradingModel
        from app.services.ml.models.xgboost_model import XGBoostTradingModel
        from app.services.ml.preprocessor import Preprocessor

        models_dir = Path("ml_artifacts/models")
        scalers_dir = Path("ml_artifacts/scalers")
        models_dir.mkdir(parents=True, exist_ok=True)
        scalers_dir.mkdir(parents=True, exist_ok=True)

        for symbol in settings.symbols_list:
            symbol_lower = symbol.lower()
            logger.info("training_start", symbol=symbol)

            # Load data
            async with async_session_factory() as db:
                df = await market_data_collector.get_latest_candles(
                    db=db, symbol=symbol, interval="1h", limit=10000,
                )

            if df.empty or len(df) < 500:
                logger.warning("insufficient_training_data", symbol=symbol, rows=len(df))
                continue

            # Feature engineering
            df_features = feature_engineer.build_features(df)
            feature_cols = feature_engineer.get_feature_columns(df_features)

            # Create target
            preprocessor = Preprocessor(threshold=0.005)
            df_features = preprocessor.create_target(df_features, horizon=5)

            # Train XGBoost
            tabular_data = preprocessor.prepare_tabular(df_features, feature_cols)
            xgb_model = XGBoostTradingModel()
            xgb_result = xgb_model.train(
                X_train=tabular_data.X_train, y_train=tabular_data.y_train,
                X_val=tabular_data.X_val, y_val=tabular_data.y_val,
                feature_names=tabular_data.feature_names,
            )
            xgb_model.save(models_dir / f"xgboost_{symbol_lower}.json")
            logger.info("xgboost_trained", symbol=symbol, val_acc=round(xgb_result.val_accuracy, 4))

            # Train LSTM
            seq_data = preprocessor.prepare_sequences(df_features, feature_cols, seq_length=60)
            lstm_model = LSTMTradingModel()
            lstm_result = lstm_model.train(
                X_train=seq_data.X_train, y_train=seq_data.y_train,
                X_val=seq_data.X_val, y_val=seq_data.y_val,
                epochs=50, batch_size=64, patience=10,
            )
            lstm_model.save(models_dir / f"lstm_{symbol_lower}.pt")
            logger.info("lstm_trained", symbol=symbol, val_acc=round(lstm_result.val_accuracy, 4))

            # Save scalers (separate for tabular and sequence models)
            Preprocessor.save_scaler(tabular_data.scaler, scalers_dir / f"scaler_{symbol_lower}.joblib")
            Preprocessor.save_scaler(seq_data.scaler, scalers_dir / f"seq_scaler_{symbol_lower}.joblib")

            logger.info("training_complete", symbol=symbol)

        # Reload models in pipeline
        from app.services.ml.pipeline import ml_pipeline
        for symbol in settings.symbols_list:
            await ml_pipeline.load_models(symbol)

        _init_status["training"] = False
        logger.info("training_phase_complete")
    except Exception as e:
        _init_status["training"] = False
        _init_status["error"] = f"Training failed: {e}"
        logger.error("training_failed", error=str(e))

"""CLI script to train ML models on historical data."""

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.core.logging import get_logger, setup_logging
from app.models.base import async_session_factory, engine, Base
from app.services.market.data_collector import market_data_collector
from app.services.ml.features import feature_engineer
from app.services.ml.models.lstm_model import LSTMTradingModel
from app.services.ml.models.xgboost_model import XGBoostTradingModel
from app.services.ml.preprocessor import Preprocessor

logger = get_logger(__name__)

MODELS_DIR = Path("ml_artifacts/models")
SCALERS_DIR = Path("ml_artifacts/scalers")


async def train_for_symbol(symbol: str, interval: str = "1h"):
    """Train LSTM and XGBoost models for a given symbol."""
    symbol_lower = symbol.lower()
    logger.info("training_start", symbol=symbol, interval=interval)

    # 1. Load data from database
    async with async_session_factory() as db:
        df = await market_data_collector.get_latest_candles(
            db=db, symbol=symbol, interval=interval, limit=10000
        )

    if df.empty or len(df) < 500:
        logger.error("insufficient_data", symbol=symbol, rows=len(df))
        return

    logger.info("data_loaded", symbol=symbol, rows=len(df))

    # 2. Feature engineering
    df_features = feature_engineer.build_features(df)
    feature_cols = feature_engineer.get_feature_columns(df_features)
    logger.info("features_built", n_features=len(feature_cols))

    # 3. Create target
    preprocessor = Preprocessor(threshold=0.005)
    df_features = preprocessor.create_target(df_features, horizon=5)

    # Log class distribution
    target_dist = df_features["target"].value_counts().to_dict()
    logger.info("target_distribution", distribution=target_dist)

    # 4. Train XGBoost
    logger.info("training_xgboost", symbol=symbol)
    tabular_data = preprocessor.prepare_tabular(df_features, feature_cols)

    xgb_model = XGBoostTradingModel()
    xgb_result = xgb_model.train(
        X_train=tabular_data.X_train,
        y_train=tabular_data.y_train,
        X_val=tabular_data.X_val,
        y_val=tabular_data.y_val,
        feature_names=tabular_data.feature_names,
    )
    logger.info(
        "xgboost_done",
        train_acc=round(xgb_result.accuracy, 4),
        val_acc=round(xgb_result.val_accuracy, 4),
    )

    # Save XGBoost
    xgb_path = MODELS_DIR / f"xgboost_{symbol_lower}.json"
    xgb_model.save(xgb_path)

    # Log feature importance
    importance = xgb_model.get_feature_importance(top_n=10)
    logger.info("xgboost_top_features", features=importance)

    # 5. Train LSTM
    logger.info("training_lstm", symbol=symbol)
    seq_data = preprocessor.prepare_sequences(
        df_features, feature_cols, seq_length=60
    )

    lstm_model = LSTMTradingModel()
    lstm_result = lstm_model.train(
        X_train=seq_data.X_train,
        y_train=seq_data.y_train,
        X_val=seq_data.X_val,
        y_val=seq_data.y_val,
        epochs=100,
        batch_size=64,
        patience=10,
    )
    logger.info(
        "lstm_done",
        train_acc=round(lstm_result.accuracy, 4),
        val_acc=round(lstm_result.val_accuracy, 4),
        best_epoch=lstm_result.best_epoch,
    )

    # Save LSTM
    lstm_path = MODELS_DIR / f"lstm_{symbol_lower}.pt"
    lstm_model.save(lstm_path)

    # 6. Save scaler
    scaler_path = SCALERS_DIR / f"scaler_{symbol_lower}.joblib"
    Preprocessor.save_scaler(tabular_data.scaler, scaler_path)

    # 7. Evaluate on test set
    logger.info("evaluating_test_set")

    # XGBoost test
    xgb_test_pred = np.array([
        xgb_model.predict(tabular_data.X_test[i]).action
        for i in range(len(tabular_data.X_test))
    ])
    xgb_test_acc = float(np.mean(xgb_test_pred == tabular_data.y_test))

    # LSTM test
    lstm_test_pred = np.array([
        lstm_model.predict(seq_data.X_test[i]).action
        for i in range(len(seq_data.X_test))
    ])
    lstm_test_acc = float(np.mean(lstm_test_pred == seq_data.y_test))

    logger.info(
        "test_results",
        symbol=symbol,
        xgboost_test_acc=round(xgb_test_acc, 4),
        lstm_test_acc=round(lstm_test_acc, 4),
    )

    logger.info("training_complete", symbol=symbol)


async def main():
    setup_logging()
    logger.info("train_models_start", symbols=settings.symbols_list)

    for symbol in settings.symbols_list:
        await train_for_symbol(symbol, interval="1h")

    logger.info("train_models_complete")


if __name__ == "__main__":
    asyncio.run(main())

"""ML inference pipeline orchestrator: features -> models -> signal."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.events import Event, EventType, event_bus
from app.core.logging import get_logger
from app.services.ml.features import feature_engineer
from app.services.ml.models.base import ModelPrediction
from app.services.ml.models.ensemble import EnsembleModel
from app.services.ml.models.lstm_model import LSTMTradingModel
from app.services.ml.models.xgboost_model import XGBoostTradingModel
from app.services.ml.models.transformer_model import TransformerTradingModel
from app.services.ml.preprocessor import Preprocessor

logger = get_logger(__name__)

MODELS_DIR = Path("ml_artifacts/models")
SCALERS_DIR = Path("ml_artifacts/scalers")


class MLPipeline:
    """Orchestrates the full ML inference pipeline:
    Raw OHLCV -> Features -> Preprocessing -> Model Inference -> Signal
    """

    def __init__(self) -> None:
        self._lstm = LSTMTradingModel()
        self._xgboost = XGBoostTradingModel()
        self._transformer = TransformerTradingModel()
        self._ensemble: EnsembleModel | None = None
        self._preprocessor = Preprocessor()
        self._tabular_scaler = None  # For XGBoost
        self._sequence_scaler = None  # For LSTM/Transformer
        self._feature_cols: list[str] = []

    async def load_models(self, symbol: str = "BTCUSDT") -> None:
        """Load trained models and scalers from disk."""
        symbol_lower = symbol.lower()

        lstm_path = MODELS_DIR / f"lstm_{symbol_lower}.pt"
        xgb_path = MODELS_DIR / f"xgboost_{symbol_lower}.json"
        transformer_path = MODELS_DIR / f"transformer_{symbol_lower}.pt"
        tabular_scaler_path = SCALERS_DIR / f"scaler_{symbol_lower}.joblib"
        sequence_scaler_path = SCALERS_DIR / f"seq_scaler_{symbol_lower}.joblib"

        models = {}
        weights = {}

        if lstm_path.exists():
            self._lstm.load(lstm_path)
            models["lstm"] = self._lstm
            weights["lstm"] = 0.35
            logger.info("lstm_loaded", symbol=symbol)

        if xgb_path.exists():
            self._xgboost.load(xgb_path)
            models["xgboost"] = self._xgboost
            weights["xgboost"] = 0.35
            logger.info("xgboost_loaded", symbol=symbol)

        if transformer_path.exists():
            self._transformer.load(transformer_path)
            models["transformer"] = self._transformer
            weights["transformer"] = 0.30
            logger.info("transformer_loaded", symbol=symbol)

        if models:
            self._ensemble = EnsembleModel(models=models, weights=weights)

        # Load scalers (separate for tabular and sequence models)
        if tabular_scaler_path.exists():
            self._tabular_scaler = Preprocessor.load_scaler(tabular_scaler_path)

        if sequence_scaler_path.exists():
            self._sequence_scaler = Preprocessor.load_scaler(sequence_scaler_path)
        elif tabular_scaler_path.exists():
            # Fallback: use tabular scaler for sequences (backward compat)
            self._sequence_scaler = Preprocessor.load_scaler(tabular_scaler_path)
            logger.warning("using_tabular_scaler_for_sequences", symbol=symbol)

        logger.info("ml_pipeline_loaded", models=list(models.keys()))

    async def predict(self, df: pd.DataFrame, symbol: str) -> ModelPrediction | None:
        """Run full inference pipeline on latest market data.

        Args:
            df: OHLCV DataFrame with at least 200 rows.
            symbol: Trading symbol (e.g., "BTCUSDT").

        Returns:
            ModelPrediction or None if insufficient data/models.
        """
        if not self._ensemble:
            logger.warning("no_models_loaded")
            return None

        if len(df) < 200:
            logger.warning("insufficient_data", rows=len(df))
            return None

        # 1. Feature engineering
        df_features = feature_engineer.build_features(df)

        if df_features.empty:
            return None

        feature_cols = feature_engineer.get_feature_columns(df_features)

        # 2. Prepare inputs for each model type
        features_dict = {}

        # XGBoost: single row, tabular scaler
        if self._xgboost.is_loaded:
            latest = df_features.iloc[-1:]
            feature_values = latest[feature_cols].values
            feature_values = np.nan_to_num(feature_values, nan=0.0)
            if self._tabular_scaler:
                feature_values = self._tabular_scaler.transform(feature_values)
            features_dict["xgboost"] = feature_values[0]  # (n_features,)

        # Sequence models (LSTM, Transformer): use sequence scaler
        seq_len = 60
        if len(df_features) >= seq_len and (self._lstm.is_loaded or self._transformer.is_loaded):
            seq_data = df_features[feature_cols].iloc[-seq_len:].values
            seq_data = np.nan_to_num(seq_data, nan=0.0)
            if self._sequence_scaler:
                seq_data = self._sequence_scaler.transform(seq_data)

            if self._lstm.is_loaded:
                features_dict["lstm"] = seq_data

            if self._transformer.is_loaded:
                features_dict["transformer"] = seq_data

        # 3. Ensemble prediction
        prediction = self._ensemble.predict(features_dict)

        # 4. Publish signal event
        await event_bus.publish(Event(
            type=EventType.SIGNAL_GENERATED,
            data={
                "symbol": symbol,
                "action": prediction.action_label,
                "strength": prediction.signal_strength,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities.tolist(),
                "model": "ensemble",
            },
        ))

        logger.info(
            "prediction_generated",
            symbol=symbol,
            action=prediction.action_label,
            strength=round(prediction.signal_strength, 4),
            confidence=round(prediction.confidence, 4),
        )

        return prediction


ml_pipeline = MLPipeline()

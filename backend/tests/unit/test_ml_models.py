"""Tests for ML models: LSTM, XGBoost, Transformer, Ensemble, DQN."""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.features import FeatureEngineer
from app.services.ml.models.base import ModelPrediction
from app.services.ml.models.ensemble import EnsembleModel
from app.services.ml.models.lstm_model import LSTMTradingModel
from app.services.ml.models.xgboost_model import XGBoostTradingModel
from app.services.ml.models.transformer_model import TransformerTradingModel
from app.services.ml.preprocessor import Preprocessor
from app.services.ml.rl.environment import TradingEnvironment
from app.services.ml.rl.dqn_agent import DQNAgent
from app.services.ml.sentiment.analyzer import SentimentAnalyzer


@pytest.fixture
def synthetic_data():
    """Generate synthetic OHLCV data with enough rows for feature engineering."""
    np.random.seed(42)
    n = 500
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000) + 100

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def tabular_data():
    """Small synthetic tabular dataset for quick model tests."""
    np.random.seed(42)
    n_features = 20
    X_train = np.random.randn(200, n_features)
    y_train = np.random.randint(0, 3, 200)
    X_val = np.random.randn(50, n_features)
    y_val = np.random.randint(0, 3, 50)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def sequence_data():
    """Small synthetic sequence dataset for LSTM tests."""
    np.random.seed(42)
    seq_len, n_features = 60, 20
    X_train = np.random.randn(100, seq_len, n_features).astype(np.float32)
    y_train = np.random.randint(0, 3, 100)
    X_val = np.random.randn(30, seq_len, n_features).astype(np.float32)
    y_val = np.random.randint(0, 3, 30)
    return X_train, y_train, X_val, y_val


# --- Feature Engineer ---

def test_feature_engineer_builds_features(synthetic_data):
    fe = FeatureEngineer()
    result = fe.build_features(synthetic_data)
    assert len(result.columns) > 50
    feature_cols = fe.get_feature_columns(result)
    assert len(feature_cols) > 50


# --- Preprocessor ---

def test_preprocessor_creates_target(synthetic_data):
    fe = FeatureEngineer()
    df = fe.build_features(synthetic_data)
    prep = Preprocessor(threshold=0.005)
    df_with_target = prep.create_target(df, horizon=5)
    assert "target" in df_with_target.columns
    assert set(df_with_target["target"].unique()).issubset({0, 1, 2})


def test_preprocessor_tabular_split(synthetic_data):
    fe = FeatureEngineer()
    df = fe.build_features(synthetic_data)
    prep = Preprocessor()
    df = prep.create_target(df, horizon=5)
    feature_cols = fe.get_feature_columns(df)

    split = prep.prepare_tabular(df, feature_cols)
    assert split.X_train.shape[1] == len(feature_cols)
    assert len(split.X_train) > 0
    assert len(split.X_val) > 0
    assert len(split.X_test) > 0


# --- XGBoost ---

def test_xgboost_train_and_predict(tabular_data):
    X_train, y_train, X_val, y_val = tabular_data
    model = XGBoostTradingModel()

    result = model.train(X_train, y_train, X_val, y_val, n_estimators=10)
    assert result.accuracy > 0
    assert model.is_loaded

    pred = model.predict(X_val[0])
    assert isinstance(pred, ModelPrediction)
    assert pred.action in (0, 1, 2)
    assert -1.0 <= pred.signal_strength <= 1.0
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == 3


def test_xgboost_save_load(tabular_data, tmp_path):
    X_train, y_train, X_val, y_val = tabular_data
    model = XGBoostTradingModel()
    model.train(X_train, y_train, X_val, y_val, n_estimators=10)

    path = tmp_path / "xgb_test.json"
    model.save(path)
    assert path.exists()

    model2 = XGBoostTradingModel()
    model2.load(path)
    assert model2.is_loaded

    pred1 = model.predict(X_val[0])
    pred2 = model2.predict(X_val[0])
    assert pred1.action == pred2.action


# --- LSTM ---

def test_lstm_train_and_predict(sequence_data):
    X_train, y_train, X_val, y_val = sequence_data
    model = LSTMTradingModel()

    result = model.train(X_train, y_train, X_val, y_val, epochs=5, patience=3)
    assert result.epochs_trained > 0
    assert model.is_loaded

    pred = model.predict(X_val[0])
    assert isinstance(pred, ModelPrediction)
    assert pred.action in (0, 1, 2)
    assert -1.0 <= pred.signal_strength <= 1.0


def test_lstm_save_load(sequence_data, tmp_path):
    X_train, y_train, X_val, y_val = sequence_data
    model = LSTMTradingModel()
    model.train(X_train, y_train, X_val, y_val, epochs=3, patience=2)

    path = tmp_path / "lstm_test.pt"
    model.save(path)
    assert path.exists()

    model2 = LSTMTradingModel()
    model2.load(path)
    assert model2.is_loaded

    pred = model2.predict(X_val[0])
    assert pred.action in (0, 1, 2)


# --- Ensemble ---

def test_ensemble_combines_models(tabular_data, sequence_data):
    X_tab_train, y_tab_train, X_tab_val, y_tab_val = tabular_data
    X_seq_train, y_seq_train, X_seq_val, y_seq_val = sequence_data

    xgb = XGBoostTradingModel()
    xgb.train(X_tab_train, y_tab_train, X_tab_val, y_tab_val, n_estimators=10)

    lstm = LSTMTradingModel()
    lstm.train(X_seq_train, y_seq_train, X_seq_val, y_seq_val, epochs=3, patience=2)

    ensemble = EnsembleModel(
        models={"xgboost": xgb, "lstm": lstm},
        weights={"xgboost": 0.5, "lstm": 0.5},
    )

    pred = ensemble.predict({
        "xgboost": X_tab_val[0],
        "lstm": X_seq_val[0],
    })

    assert isinstance(pred, ModelPrediction)
    assert pred.action in (0, 1, 2)
    assert -1.0 <= pred.signal_strength <= 1.0
    assert abs(sum(pred.probabilities) - 1.0) < 0.01


def test_ensemble_signal_to_action():
    assert EnsembleModel.signal_to_action(0.5) == "BUY"
    assert EnsembleModel.signal_to_action(-0.5) == "SELL"
    assert EnsembleModel.signal_to_action(0.0) == "HOLD"
    assert EnsembleModel.signal_to_action(0.29) == "HOLD"


def test_model_prediction_label():
    pred = ModelPrediction(
        action=2,
        probabilities=np.array([0.1, 0.2, 0.7]),
        confidence=0.7,
        signal_strength=0.6,
    )
    assert pred.action_label == "BUY"


# --- Transformer ---

def test_transformer_train_and_predict(sequence_data):
    X_train, y_train, X_val, y_val = sequence_data
    model = TransformerTradingModel()

    result = model.train(X_train, y_train, X_val, y_val, epochs=5, patience=3)
    assert result.accuracy > 0
    assert model.is_loaded

    pred = model.predict(X_val[0])
    assert isinstance(pred, ModelPrediction)
    assert pred.action in (0, 1, 2)
    assert -1.0 <= pred.signal_strength <= 1.0


def test_transformer_save_load(sequence_data, tmp_path):
    X_train, y_train, X_val, y_val = sequence_data
    model = TransformerTradingModel()
    model.train(X_train, y_train, X_val, y_val, epochs=3, patience=2)

    path = tmp_path / "transformer_test.pt"
    model.save(path)
    assert path.exists()

    model2 = TransformerTradingModel()
    model2.load(path)
    assert model2.is_loaded

    pred = model2.predict(X_val[0])
    assert pred.action in (0, 1, 2)


# --- DQN Agent ---

def test_dqn_environment_step():
    n = 100
    prices = 50000 + np.cumsum(np.random.randn(n) * 50)
    features = np.random.randn(n, 10)

    env = TradingEnvironment(prices=prices, features=features, initial_capital=10000)
    state = env.reset()

    assert state.position == 0
    assert state.equity_ratio == 1.0

    # Open long
    result = env.step(1)
    assert not result.done
    assert result.state.position == 1

    # Close position
    result = env.step(3)
    assert result.state.position == 0


def test_dqn_agent_select_action():
    n = 100
    prices = 50000 + np.cumsum(np.random.randn(n) * 50)
    features = np.random.randn(n, 10)

    env = TradingEnvironment(prices=prices, features=features)
    state = env.reset()

    agent = DQNAgent(state_dim=env.state_dim, n_actions=4)
    action = agent.select_action(state)
    assert action in range(4)


def test_dqn_agent_training():
    np.random.seed(42)
    n = 200
    prices = 50000 + np.cumsum(np.random.randn(n) * 50)
    features = np.random.randn(n, 10).astype(np.float32)

    env = TradingEnvironment(prices=prices, features=features)
    agent = DQNAgent(state_dim=env.state_dim, n_actions=4, batch_size=32)

    result = agent.train_on_env(env, episodes=5)
    assert "mean_reward" in result
    assert "best_reward" in result


# --- Sentiment Analyzer ---

def test_sentiment_analyzer_fallback():
    """Test that analyzer returns neutral when model is not loaded."""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Bitcoin surges to new all-time high")
    assert result.label == "neutral"
    assert result.compound == 0.0


def test_sentiment_aggregate_empty():
    analyzer = SentimentAnalyzer()
    score = analyzer.aggregate_sentiment([])
    assert score == 0.0

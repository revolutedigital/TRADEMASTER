"""Unit tests for transformer predictor and synthetic data."""
import numpy as np
import pytest

from app.services.ml.models.transformer_model import TemporalFusionPredictor, NumpyAttention
from app.services.ml.synthetic_data import SyntheticMarketGenerator
from app.services.ml.sentiment import SentimentAnalyzer


class TestNumpyAttention:
    def test_output_shape(self):
        attn = NumpyAttention(d_model=32, n_heads=4)
        x = np.random.randn(10, 32).astype(np.float32)
        output, weights = attn.forward(x)
        assert output.shape == (10, 32)
        assert weights.shape == (10, 10)

    def test_attention_weights_sum_to_one(self):
        attn = NumpyAttention(d_model=16, n_heads=2)
        x = np.random.randn(5, 16).astype(np.float32)
        _, weights = attn.forward(x)
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestTransformerPredictor:
    def test_predict_output_structure(self):
        pred = TemporalFusionPredictor(d_model=32, horizons=[1, 5, 10])
        features = np.random.randn(20, 8).astype(np.float32)
        result = pred.predict(features)
        assert len(result.predictions) == 3
        assert len(result.horizons) == 3
        assert 0 <= result.confidence <= 1
        assert len(result.attention_weights) == 20

    def test_predictions_bounded(self):
        pred = TemporalFusionPredictor(d_model=32)
        features = np.random.randn(30, 10).astype(np.float32)
        result = pred.predict(features)
        for p in result.predictions:
            assert 0 <= p <= 1  # Sigmoid output


class TestSyntheticGenerator:
    def test_flash_crash_shape(self):
        gen = SyntheticMarketGenerator(seed=42)
        scenario = gen.generate_flash_crash(candles=50)
        assert scenario.data.shape == (50, 5)
        assert scenario.name == "flash_crash"

    def test_all_scenarios(self):
        gen = SyntheticMarketGenerator(seed=42)
        scenarios = gen.generate_all_scenarios()
        assert len(scenarios) == 5
        for s in scenarios:
            assert s.data.shape[1] == 5
            assert s.candles == s.data.shape[0]

    def test_ohlcv_validity(self):
        gen = SyntheticMarketGenerator(seed=42)
        scenario = gen.generate_sideways(candles=100)
        for i in range(100):
            o, h, l, c, v = scenario.data[i]
            assert h >= l, f"high < low at candle {i}"
            assert v > 0, f"volume <= 0 at candle {i}"


class TestSentimentAnalyzer:
    def test_basic_sentiment(self):
        analyzer = SentimentAnalyzer()
        prices = np.linspace(100, 110, 30)  # Uptrend
        volumes = np.ones(30) * 1000
        result = analyzer.analyze_from_market_data(prices, volumes)
        assert -1 <= result.overall <= 1
        assert result.interpretation in ("extreme_fear", "fear", "neutral", "greed", "extreme_greed")

    def test_downtrend_is_fearful(self):
        analyzer = SentimentAnalyzer()
        prices = np.linspace(110, 90, 30)  # Downtrend
        volumes = np.ones(30) * 1000
        result = analyzer.analyze_from_market_data(prices, volumes)
        assert result.overall < 0  # Should be bearish/fearful

    def test_uptrend_is_greedy(self):
        analyzer = SentimentAnalyzer()
        prices = np.linspace(90, 120, 30)  # Strong uptrend
        volumes = np.ones(30) * 1000
        result = analyzer.analyze_from_market_data(prices, volumes)
        assert result.overall > 0  # Should be bullish/greedy

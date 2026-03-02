"""Performance regression benchmarks.

Run with: pytest tests/performance/ --benchmark
Tracks execution time for critical operations to detect regressions.
"""

import time
import pytest
import numpy as np
import pandas as pd


class TestFeatureEngineeringPerformance:
    """Benchmark feature engineering pipeline."""

    def _create_sample_data(self, n: int) -> pd.DataFrame:
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(n)) + 50000
        return pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(n)) * 100,
            'low': prices - np.abs(np.random.randn(n)) * 100,
            'close': prices + np.random.randn(n) * 50,
            'volume': np.abs(np.random.randn(n)) * 1000 + 500,
        })

    def test_feature_engineering_1k_rows(self):
        """Feature engineering for 1k rows should complete in < 2s."""
        from app.services.ml.features import feature_engineer
        df = self._create_sample_data(1000)
        
        start = time.perf_counter()
        result = feature_engineer.build_features(df)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0, f"Feature engineering took {elapsed:.2f}s (limit: 2.0s)"
        assert len(result) > 0

    def test_feature_engineering_10k_rows(self):
        """Feature engineering for 10k rows should complete in < 10s."""
        from app.services.ml.features import feature_engineer
        df = self._create_sample_data(10000)
        
        start = time.perf_counter()
        result = feature_engineer.build_features(df)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 10.0, f"Feature engineering took {elapsed:.2f}s (limit: 10.0s)"
        assert len(result) > 0


class TestRiskCalculationPerformance:
    """Benchmark risk calculation operations."""

    def test_var_calculation_speed(self):
        """VaR calculation for 1000 returns should complete in < 100ms."""
        returns = np.random.randn(1000) * 0.02
        
        start = time.perf_counter()
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"VaR calculation took {elapsed:.4f}s"
        assert var_95 < 0
        assert var_99 < var_95

    def test_monte_carlo_simulation_speed(self):
        """Monte Carlo with 10k simulations should complete in < 5s."""
        from app.services.risk.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator()
        returns = list(np.random.randn(252) * 0.02)
        
        start = time.perf_counter()
        result = sim.simulate(portfolio_value=10000.0, returns=returns, n_simulations=10000, horizon_days=30)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0, f"Monte Carlo took {elapsed:.2f}s (limit: 5.0s)"
        assert "median_outcome" in result


class TestDataProcessingPerformance:
    """Benchmark data processing operations."""

    def test_candle_batch_processing(self):
        """Processing 10k candles should complete in < 1s."""
        from tests.factories.factories import OHLCVFactory
        
        start = time.perf_counter()
        candles = OHLCVFactory.create_batch(10000)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"Batch creation took {elapsed:.2f}s"
        assert len(candles) == 10000

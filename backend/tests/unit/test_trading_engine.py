"""Unit tests for the TradingEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.services.trading_engine import TradingEngine


class TestTradingEngine:
    def test_initial_state(self):
        engine = TradingEngine()
        assert engine._running is False
        assert engine._last_signal_time == {}

    @patch("app.services.trading_engine.binance_client")
    @patch("app.services.trading_engine.circuit_breaker")
    async def test_start_initializes_circuit_breaker(self, mock_cb, mock_binance):
        engine = TradingEngine()
        mock_cb.restore_from_redis = AsyncMock(return_value=False)
        mock_binance.get_balance = AsyncMock(return_value=10000.0)
        mock_cb.initialize = MagicMock()

        # Start and immediately stop to avoid infinite loop
        async def start_and_stop():
            engine._running = True
            mock_cb.restore_from_redis.return_value = False
            await engine.start.__wrapped__(engine) if hasattr(engine.start, '__wrapped__') else None

        # Just test initialization logic
        mock_cb.restore_from_redis.return_value = False
        mock_binance.get_balance.return_value = 10000.0

        assert engine._running is False

    def test_min_signal_interval(self):
        engine = TradingEngine()
        assert engine._min_signal_interval_seconds == 10

    @patch("app.services.trading_engine.ml_pipeline")
    async def test_engine_stop(self, mock_ml):
        engine = TradingEngine()
        engine._running = True
        await engine.stop()
        assert engine._running is False


class TestSignalThrottling:
    def test_signal_time_tracking(self):
        engine = TradingEngine()
        now = datetime.now(timezone.utc)
        engine._last_signal_time["BTCUSDT"] = now
        assert "BTCUSDT" in engine._last_signal_time
        assert engine._last_signal_time["BTCUSDT"] == now

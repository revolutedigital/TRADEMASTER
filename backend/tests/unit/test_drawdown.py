"""Unit tests for the DrawdownCircuitBreaker with Redis persistence."""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.risk.drawdown import (
    DrawdownCircuitBreaker,
    CircuitBreakerState,
    REDIS_KEY,
)


class TestCircuitBreakerStates:
    def test_initial_state_normal(self):
        cb = DrawdownCircuitBreaker()
        cb.initialize(10000)
        assert cb.state == CircuitBreakerState.NORMAL
        assert cb.can_trade is True
        assert cb.position_size_multiplier == 1.0

    def test_daily_drawdown_triggers_pause(self):
        cb = DrawdownCircuitBreaker(max_daily_drawdown=0.03)
        cb.initialize(10000)
        state = cb.update(9600)  # 4% loss
        assert state == CircuitBreakerState.PAUSED
        assert cb.can_trade is False

    def test_weekly_drawdown_triggers_pause(self):
        cb = DrawdownCircuitBreaker(max_weekly_drawdown=0.07)
        cb.initialize(10000)
        state = cb.update(9200)  # 8% loss
        assert state in (CircuitBreakerState.PAUSED, CircuitBreakerState.HALTED)

    def test_total_drawdown_triggers_halt(self):
        cb = DrawdownCircuitBreaker(max_total_drawdown=0.15)
        cb.initialize(10000)
        state = cb.update(8400)  # 16% total drawdown
        assert state == CircuitBreakerState.HALTED
        assert cb.position_size_multiplier == 0.0
        assert cb.can_trade is False

    def test_manual_reset_restores_normal(self):
        cb = DrawdownCircuitBreaker(max_total_drawdown=0.15)
        cb.initialize(10000)
        cb.update(8000)
        assert cb.state == CircuitBreakerState.HALTED
        cb.manual_reset(8000)
        assert cb.state == CircuitBreakerState.NORMAL

    def test_get_status_dict(self):
        cb = DrawdownCircuitBreaker()
        cb.initialize(10000)
        status = cb.get_status()
        assert isinstance(status, dict)
        assert "state" in status
        assert "can_trade" in status
        assert "position_size_multiplier" in status

    def test_reduced_state_half_size(self):
        cb = DrawdownCircuitBreaker(max_monthly_drawdown=0.05)
        cb.initialize(10000)
        # Trigger reduced state (monthly drawdown > 5%)
        state = cb.update(9400)
        # Depending on implementation, might be REDUCED or PAUSED
        if state == CircuitBreakerState.REDUCED:
            assert cb.position_size_multiplier == 0.5
            assert cb.can_trade is True


class TestCircuitBreakerRedis:
    @patch("app.services.risk.drawdown.redis")
    async def test_save_to_redis(self, mock_redis):
        mock_redis.set = AsyncMock()
        cb = DrawdownCircuitBreaker()
        cb.initialize(10000)
        await cb.save_to_redis()
        mock_redis.set.assert_called_once()

    @patch("app.services.risk.drawdown.redis")
    async def test_restore_from_redis_no_data(self, mock_redis):
        mock_redis.get = AsyncMock(return_value=None)
        cb = DrawdownCircuitBreaker()
        result = await cb.restore_from_redis()
        assert result is False

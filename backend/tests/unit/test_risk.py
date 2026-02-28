"""Tests for risk management: position sizing, stop loss, circuit breaker, risk manager."""

import pytest
import numpy as np

from app.services.risk.position_sizer import PositionSizer, PositionSize
from app.services.risk.stop_loss import StopLossCalculator
from app.services.risk.drawdown import DrawdownCircuitBreaker, CircuitBreakerState
from app.services.risk.manager import RiskManager, TradeProposal
from app.core.exceptions import RiskLimitExceededError, DrawdownCircuitBreakerError


# ---- Position Sizer ----

class TestPositionSizer:
    def setup_method(self):
        self.sizer = PositionSizer(
            max_risk_per_trade=0.02,
            max_single_asset_exposure=0.30,
            kelly_fraction=0.15,
        )

    def test_fixed_fraction_basic(self):
        result = self.sizer.fixed_fraction(
            equity=10000, price=50000, stop_distance_pct=0.02
        )
        assert result.quantity > 0
        assert result.risk_pct == 0.02
        assert result.risk_amount == 200  # 2% of 10k
        assert result.method == "fixed_fraction"

    def test_fixed_fraction_respects_max_exposure(self):
        result = self.sizer.fixed_fraction(
            equity=10000, price=100, stop_distance_pct=0.001
        )
        # Very tight stop would give huge position, should be capped
        assert result.notional_value <= 10000 * 0.30  # max 30%

    def test_fractional_kelly(self):
        result = self.sizer.fractional_kelly(
            equity=10000, win_rate=0.55, avg_win=200, avg_loss=100,
            price=50000, stop_distance_pct=0.02,
        )
        assert result.quantity > 0
        assert result.risk_pct <= 0.02
        assert result.method == "fractional_kelly"

    def test_kelly_negative_expectancy_returns_zero(self):
        result = self.sizer.fractional_kelly(
            equity=10000, win_rate=0.30, avg_win=100, avg_loss=200,
            price=50000, stop_distance_pct=0.02,
        )
        assert result.quantity == 0

    def test_volatility_scaled(self):
        result = self.sizer.volatility_scaled(
            equity=10000, price=50000, atr=1000, atr_multiplier=2.0
        )
        assert result.quantity > 0

    def test_zero_equity_returns_zero(self):
        result = self.sizer.fixed_fraction(equity=0, price=50000, stop_distance_pct=0.02)
        assert result.quantity == 0


# ---- Stop Loss ----

class TestStopLoss:
    def setup_method(self):
        self.calc = StopLossCalculator(
            atr_multiplier=2.0,
            trailing_activation_pct=0.015,
            trailing_distance_pct=0.01,
            risk_reward_ratio=2.0,
        )

    def test_atr_based_long(self):
        sl = self.calc.atr_based(entry_price=50000, atr=500, side="LONG")
        assert sl.stop_price == 49000  # 50000 - 2*500
        assert sl.take_profit_price == 52000  # 50000 + 2*500*2

    def test_atr_based_short(self):
        sl = self.calc.atr_based(entry_price=50000, atr=500, side="SHORT")
        assert sl.stop_price == 51000
        assert sl.take_profit_price == 48000

    def test_trailing_stop_activates(self):
        # Entry at 50000, 1.5% profit = 50750
        new_stop = self.calc.update_trailing_stop(
            entry_price=50000, current_price=51000,
            current_stop=49000, side="LONG",
        )
        # Should trail at 1% from current: 51000 * 0.99 = 50490
        assert new_stop > 49000
        assert new_stop == pytest.approx(50490, rel=0.01)

    def test_trailing_stop_doesnt_activate_too_early(self):
        new_stop = self.calc.update_trailing_stop(
            entry_price=50000, current_price=50500,
            current_stop=49000, side="LONG",
        )
        # Only 1% profit, threshold is 1.5% — should NOT activate
        assert new_stop == 49000

    def test_is_stop_hit_long(self):
        assert self.calc.is_stop_hit(48000, 49000, "LONG") is True
        assert self.calc.is_stop_hit(50000, 49000, "LONG") is False

    def test_is_stop_hit_short(self):
        assert self.calc.is_stop_hit(52000, 51000, "SHORT") is True
        assert self.calc.is_stop_hit(50000, 51000, "SHORT") is False

    def test_is_take_profit_hit(self):
        assert self.calc.is_take_profit_hit(53000, 52000, "LONG") is True
        assert self.calc.is_take_profit_hit(47000, 48000, "SHORT") is True


# ---- Drawdown Circuit Breaker ----

class TestDrawdownCircuitBreaker:
    def test_initial_state_is_normal(self):
        cb = DrawdownCircuitBreaker()
        cb.initialize(10000)
        assert cb.state == CircuitBreakerState.NORMAL
        assert cb.can_trade is True
        assert cb.position_size_multiplier == 1.0

    def test_daily_drawdown_pauses_trading(self):
        cb = DrawdownCircuitBreaker(max_daily_drawdown=0.03)
        cb.initialize(10000)
        # Simulate 4% daily loss
        state = cb.update(9600)
        assert state == CircuitBreakerState.PAUSED
        assert cb.can_trade is False

    def test_total_drawdown_halts(self):
        cb = DrawdownCircuitBreaker(max_total_drawdown=0.15)
        cb.initialize(10000)
        state = cb.update(8400)  # 16% drawdown from peak
        assert state == CircuitBreakerState.HALTED
        assert cb.position_size_multiplier == 0.0

    def test_recovery_to_normal(self):
        cb = DrawdownCircuitBreaker(max_daily_drawdown=0.03)
        cb.initialize(10000)
        cb.update(9600)  # Trigger pause
        assert cb.state == CircuitBreakerState.PAUSED
        # Recovery (within same session, no daily loss)
        cb._pnl_history.clear()
        state = cb.update(10000)
        assert state == CircuitBreakerState.NORMAL

    def test_manual_reset(self):
        cb = DrawdownCircuitBreaker(max_total_drawdown=0.15)
        cb.initialize(10000)
        cb.update(8000)
        assert cb.state == CircuitBreakerState.HALTED
        cb.manual_reset(8000)
        assert cb.state == CircuitBreakerState.NORMAL

    def test_get_status(self):
        cb = DrawdownCircuitBreaker()
        cb.initialize(10000)
        status = cb.get_status()
        assert "state" in status
        assert "can_trade" in status
        assert status["can_trade"] is True


# ---- Risk Manager ----

class TestRiskManager:
    def test_valid_trade_passes(self):
        rm = RiskManager()
        circuit_breaker_module = __import__(
            "app.services.risk.drawdown", fromlist=["circuit_breaker"]
        )
        circuit_breaker_module.circuit_breaker.initialize(10000)

        proposal = TradeProposal(
            symbol="BTCUSDT",
            side="BUY",
            signal_strength=0.6,
            entry_price=50000,
            atr=500,
            current_equity=10000,
            current_exposure=0,
            symbol_exposure=0,
        )

        approved = rm.validate_trade(proposal)
        assert approved.symbol == "BTCUSDT"
        assert approved.side == "BUY"
        assert approved.quantity > 0
        assert approved.stop_loss.stop_price < 50000
        assert len(approved.risk_checks_passed) >= 5

    def test_rejects_when_exposure_exceeded(self):
        rm = RiskManager()
        from app.services.risk.drawdown import circuit_breaker
        circuit_breaker.initialize(10000)

        proposal = TradeProposal(
            symbol="BTCUSDT",
            side="BUY",
            signal_strength=0.6,
            entry_price=50000,
            atr=500,
            current_equity=10000,
            current_exposure=5500,  # Already 55% exposure
            symbol_exposure=0,
        )

        with pytest.raises(RiskLimitExceededError):
            rm.validate_trade(proposal)

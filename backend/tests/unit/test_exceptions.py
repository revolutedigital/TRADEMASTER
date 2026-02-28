"""Tests for custom exception hierarchy."""

from app.core.exceptions import (
    TradeMasterError,
    ExchangeConnectionError,
    ExchangeRateLimitError,
    OrderExecutionError,
    RiskLimitExceededError,
    DrawdownCircuitBreakerError,
    ModelNotLoadedError,
)


def test_base_exception():
    err = TradeMasterError("something went wrong", "TEST_ERROR")
    assert str(err) == "something went wrong"
    assert err.code == "TEST_ERROR"


def test_exchange_exceptions_are_trademaster_errors():
    assert issubclass(ExchangeConnectionError, TradeMasterError)
    assert issubclass(ExchangeRateLimitError, TradeMasterError)
    assert issubclass(OrderExecutionError, TradeMasterError)


def test_trading_exceptions():
    err = RiskLimitExceededError("max exposure reached")
    assert err.code == "RISK_LIMIT_EXCEEDED"
    assert "max exposure" in str(err)


def test_circuit_breaker_exception():
    err = DrawdownCircuitBreakerError()
    assert err.code == "DRAWDOWN_CIRCUIT_BREAKER"


def test_ml_exception():
    err = ModelNotLoadedError()
    assert err.code == "MODEL_NOT_LOADED"

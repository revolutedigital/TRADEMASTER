class TradeMasterError(Exception):
    """Base exception for all TradeMaster errors."""

    def __init__(self, message: str = "", code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


# --- Exchange Errors ---


class ExchangeError(TradeMasterError):
    """Base exception for exchange-related errors."""

    def __init__(self, message: str = "", code: str = "EXCHANGE_ERROR"):
        super().__init__(message, code)


class ExchangeConnectionError(ExchangeError):
    def __init__(self, message: str = "Failed to connect to exchange"):
        super().__init__(message, "EXCHANGE_CONNECTION_ERROR")


class ExchangeRateLimitError(ExchangeError):
    def __init__(self, message: str = "Exchange rate limit exceeded"):
        super().__init__(message, "EXCHANGE_RATE_LIMIT")


class OrderExecutionError(ExchangeError):
    def __init__(self, message: str = "Order execution failed"):
        super().__init__(message, "ORDER_EXECUTION_ERROR")


class InsufficientBalanceError(ExchangeError):
    def __init__(self, message: str = "Insufficient balance"):
        super().__init__(message, "INSUFFICIENT_BALANCE")


# --- Trading Errors ---


class TradingError(TradeMasterError):
    """Base exception for trading logic errors."""

    def __init__(self, message: str = "", code: str = "TRADING_ERROR"):
        super().__init__(message, code)


class RiskLimitExceededError(TradingError):
    def __init__(self, message: str = "Risk limit exceeded"):
        super().__init__(message, "RISK_LIMIT_EXCEEDED")


class DrawdownCircuitBreakerError(TradingError):
    def __init__(self, message: str = "Drawdown circuit breaker triggered"):
        super().__init__(message, "DRAWDOWN_CIRCUIT_BREAKER")


class InvalidSignalError(TradingError):
    def __init__(self, message: str = "Invalid trading signal"):
        super().__init__(message, "INVALID_SIGNAL")


# --- ML Errors ---


class MLError(TradeMasterError):
    """Base exception for ML pipeline errors."""

    def __init__(self, message: str = "", code: str = "ML_ERROR"):
        super().__init__(message, code)


class ModelNotLoadedError(MLError):
    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, "MODEL_NOT_LOADED")


class PredictionError(MLError):
    def __init__(self, message: str = "Prediction failed"):
        super().__init__(message, "PREDICTION_ERROR")


# --- Data Errors ---


class DataError(TradeMasterError):
    """Base exception for data pipeline errors."""

    def __init__(self, message: str = "", code: str = "DATA_ERROR"):
        super().__init__(message, code)


class DataNotFoundError(DataError):
    def __init__(self, message: str = "Data not found"):
        super().__init__(message, "DATA_NOT_FOUND")

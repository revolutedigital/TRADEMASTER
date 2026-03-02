"""TradeMaster exception hierarchy.

Organized by domain with support for retryable errors, context info,
and proper HTTP status mapping.
"""


class TradeMasterError(Exception):
    """Base exception for all TradeMaster errors."""

    def __init__(self, message: str = "", code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}" if self.message else self.code


class RetriableError(TradeMasterError):
    """Base for errors that should trigger retry logic."""

    def __init__(self, message: str = "", code: str = "RETRIABLE_ERROR", max_retries: int = 3):
        super().__init__(message, code)
        self.max_retries = max_retries


# --- Validation Errors ---


class ValidationError(TradeMasterError):
    """Business rule validation failures."""

    def __init__(self, message: str = "Validation failed", code: str = "VALIDATION_ERROR", field: str | None = None):
        super().__init__(message, code)
        self.field = field


class NotFoundError(TradeMasterError):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found", code: str = "NOT_FOUND", resource: str | None = None):
        super().__init__(message, code)
        self.resource = resource


class ConfigurationError(TradeMasterError):
    """Configuration/setup errors that prevent normal operation."""

    def __init__(self, message: str = "Configuration error"):
        super().__init__(message, "CONFIGURATION_ERROR")


# --- Exchange Errors ---


class ExchangeError(TradeMasterError):
    """Base exception for exchange-related errors."""

    def __init__(
        self,
        message: str = "",
        code: str = "EXCHANGE_ERROR",
        exchange_response: dict | None = None,
        retry_after: int | None = None,
    ):
        super().__init__(message, code)
        self.exchange_response = exchange_response
        self.retry_after = retry_after


class ExchangeConnectionError(ExchangeError, RetriableError):
    def __init__(self, message: str = "Failed to connect to exchange"):
        TradeMasterError.__init__(self, message, "EXCHANGE_CONNECTION_ERROR")
        self.max_retries = 5
        self.exchange_response = None
        self.retry_after = None


class ExchangeRateLimitError(ExchangeError, RetriableError):
    def __init__(self, message: str = "Exchange rate limit exceeded", retry_after: int | None = None):
        TradeMasterError.__init__(self, message, "EXCHANGE_RATE_LIMIT")
        self.retry_after = retry_after
        self.exchange_response = None
        self.max_retries = 3


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


# --- Data / Repository Errors ---


class DataError(TradeMasterError):
    """Base exception for data pipeline errors."""

    def __init__(self, message: str = "", code: str = "DATA_ERROR"):
        super().__init__(message, code)


class DataNotFoundError(DataError):
    def __init__(self, message: str = "Data not found"):
        super().__init__(message, "DATA_NOT_FOUND")


class RepositoryError(DataError):
    """Database/repository layer errors."""

    def __init__(self, message: str = "Repository operation failed"):
        super().__init__(message, "REPOSITORY_ERROR")


class DataIntegrityError(DataError):
    """Data integrity constraint violation."""

    def __init__(self, message: str = "Data integrity error"):
        super().__init__(message, "DATA_INTEGRITY_ERROR")


# --- HTTP status mapping for global exception handler ---

EXCEPTION_STATUS_MAP: dict[type[TradeMasterError], int] = {
    ValidationError: 422,
    NotFoundError: 404,
    ConfigurationError: 500,
    ExchangeRateLimitError: 429,
    InsufficientBalanceError: 400,
    OrderExecutionError: 502,
    ExchangeConnectionError: 503,
    RiskLimitExceededError: 400,
    DrawdownCircuitBreakerError: 403,
    InvalidSignalError: 400,
    ModelNotLoadedError: 503,
    PredictionError: 500,
    DataNotFoundError: 404,
    RepositoryError: 500,
    DataIntegrityError: 409,
}

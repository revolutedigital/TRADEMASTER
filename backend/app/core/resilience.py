"""Resilience patterns: retry, circuit breaker, rate limiter."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (Exception,)


def retry_async(config: RetryConfig | None = None):
    """Decorator for async retry with exponential backoff."""
    cfg = config or RetryConfig()

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(cfg.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_error = e
                    if attempt < cfg.max_retries:
                        delay = min(
                            cfg.base_delay * (cfg.exponential_base ** attempt),
                            cfg.max_delay,
                        )
                        logger.warning(
                            "retry_attempt",
                            func=func.__name__,
                            attempt=attempt + 1,
                            max_retries=cfg.max_retries,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)

            logger.error(
                "retry_exhausted",
                func=func.__name__,
                max_retries=cfg.max_retries,
                error=str(last_error),
            )
            raise last_error  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


@dataclass
class ServiceCircuitBreaker:
    """Circuit breaker for external service calls.

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

    _failure_count: int = field(default=0, init=False)
    _state: str = field(default="CLOSED", init=False)
    _last_failure_time: datetime | None = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> str:
        if self._state == "OPEN" and self._last_failure_time:
            elapsed = (
                datetime.now(timezone.utc) - self._last_failure_time
            ).total_seconds()
            if elapsed >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
        return self._state

    @property
    def is_available(self) -> bool:
        state = self.state
        if state == "CLOSED":
            return True
        if state == "HALF_OPEN":
            return self._half_open_calls < self.half_open_max_calls
        return False

    def record_success(self) -> None:
        if self._state == "HALF_OPEN":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = "CLOSED"
                self._failure_count = 0
                logger.info("circuit_breaker_closed")
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            logger.warning("circuit_breaker_reopened")
        elif self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                "circuit_breaker_opened",
                failures=self._failure_count,
            )

    def reset(self) -> None:
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = None


class Reconciler:
    """Reconciles local state with exchange state.

    Periodically checks that local positions and orders
    match what the exchange reports, fixing discrepancies.
    """

    def __init__(self):
        self._last_reconciliation: datetime | None = None

    async def reconcile_orders(self, local_orders: list, exchange_orders: list) -> dict:
        """Compare local orders with exchange orders and report discrepancies."""
        local_by_id = {o.get("exchange_order_id"): o for o in local_orders if o.get("exchange_order_id")}
        exchange_by_id = {o.get("orderId"): o for o in exchange_orders}

        missing_local = []
        missing_exchange = []
        status_mismatch = []

        for oid, eo in exchange_by_id.items():
            if str(oid) not in local_by_id:
                missing_local.append(eo)

        for oid, lo in local_by_id.items():
            if oid not in exchange_by_id:
                missing_exchange.append(lo)
            else:
                eo = exchange_by_id[oid]
                if lo.get("status") != eo.get("status"):
                    status_mismatch.append({
                        "order_id": oid,
                        "local_status": lo.get("status"),
                        "exchange_status": eo.get("status"),
                    })

        result = {
            "missing_local": len(missing_local),
            "missing_exchange": len(missing_exchange),
            "status_mismatch": len(status_mismatch),
            "details": status_mismatch[:10],
        }

        if missing_local or status_mismatch:
            logger.warning("reconciliation_discrepancies", **result)
        else:
            logger.info("reconciliation_clean")

        self._last_reconciliation = datetime.now(timezone.utc)
        return result


reconciler = Reconciler()

"""Rate limiter for Binance API and internal endpoints."""

import time
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

# Binance rate limits
BINANCE_WEIGHT_LIMIT_PER_MINUTE = 2400
BINANCE_ORDER_LIMIT_PER_SECOND = 10
BINANCE_ORDER_LIMIT_PER_DAY = 200000
THROTTLE_THRESHOLD = 0.80  # Start throttling at 80% usage


@dataclass
class RateLimitWindow:
    """Sliding window rate limit tracker."""

    max_requests: int
    window_seconds: float
    requests: list[float] = field(default_factory=list)

    def _cleanup(self) -> None:
        cutoff = time.monotonic() - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff]

    def can_proceed(self) -> bool:
        self._cleanup()
        return len(self.requests) < self.max_requests

    def record(self) -> None:
        self.requests.append(time.monotonic())

    @property
    def usage_ratio(self) -> float:
        self._cleanup()
        return len(self.requests) / self.max_requests if self.max_requests > 0 else 0.0

    @property
    def remaining(self) -> int:
        self._cleanup()
        return max(0, self.max_requests - len(self.requests))


class BinanceRateLimiter:
    """Tracks Binance API rate limits from response headers."""

    def __init__(self) -> None:
        self._weight_used: int = 0
        self._weight_limit: int = BINANCE_WEIGHT_LIMIT_PER_MINUTE
        self._last_reset: float = time.monotonic()
        self._order_window = RateLimitWindow(
            max_requests=BINANCE_ORDER_LIMIT_PER_SECOND, window_seconds=1.0
        )

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """Update rate limit state from Binance response headers."""
        if "X-MBX-USED-WEIGHT-1M" in headers:
            self._weight_used = int(headers["X-MBX-USED-WEIGHT-1M"])

        now = time.monotonic()
        if now - self._last_reset > 60:
            self._weight_used = 0
            self._last_reset = now

    def can_make_request(self, weight: int = 1) -> bool:
        """Check if we can make a request without exceeding rate limits."""
        threshold = int(self._weight_limit * THROTTLE_THRESHOLD)
        if self._weight_used + weight > threshold:
            logger.warning(
                "binance_rate_limit_throttle",
                used=self._weight_used,
                threshold=threshold,
                limit=self._weight_limit,
            )
            return False
        return True

    def can_place_order(self) -> bool:
        return self._order_window.can_proceed()

    def record_order(self) -> None:
        self._order_window.record()

    @property
    def weight_usage_ratio(self) -> float:
        return self._weight_used / self._weight_limit if self._weight_limit > 0 else 0.0

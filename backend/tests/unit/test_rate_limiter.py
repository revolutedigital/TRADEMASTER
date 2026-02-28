"""Tests for rate limiter."""

from app.core.rate_limiter import BinanceRateLimiter, RateLimitWindow


def test_rate_limit_window_allows_requests():
    window = RateLimitWindow(max_requests=10, window_seconds=60)
    assert window.can_proceed() is True
    assert window.remaining == 10


def test_rate_limit_window_blocks_when_full():
    window = RateLimitWindow(max_requests=3, window_seconds=60)
    window.record()
    window.record()
    window.record()
    assert window.can_proceed() is False
    assert window.remaining == 0


def test_rate_limit_window_usage_ratio():
    window = RateLimitWindow(max_requests=10, window_seconds=60)
    window.record()
    window.record()
    assert window.usage_ratio == 0.2


def test_binance_rate_limiter_allows_by_default():
    limiter = BinanceRateLimiter()
    assert limiter.can_make_request(weight=1) is True


def test_binance_rate_limiter_throttles_at_threshold():
    limiter = BinanceRateLimiter()
    # Simulate high usage from headers
    limiter.update_from_headers({"X-MBX-USED-WEIGHT-1M": "2000"})
    assert limiter.can_make_request(weight=1) is False


def test_binance_rate_limiter_weight_usage():
    limiter = BinanceRateLimiter()
    limiter.update_from_headers({"X-MBX-USED-WEIGHT-1M": "1200"})
    assert limiter.weight_usage_ratio == 0.5

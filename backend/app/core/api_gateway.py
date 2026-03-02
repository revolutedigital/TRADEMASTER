"""API Gateway: centralized rate limiting, auth, caching, and analytics.

Provides Kong/Traefik-like gateway features at the application level.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouteConfig:
    path_prefix: str
    service: str
    rate_limit: int = 100  # requests per minute
    cache_ttl: int = 0  # seconds, 0 = no cache
    auth_required: bool = True
    cors_enabled: bool = True


@dataclass
class RequestAnalytics:
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    requests_by_path: dict = field(default_factory=lambda: defaultdict(int))
    requests_by_status: dict = field(default_factory=lambda: defaultdict(int))


class APIGateway:
    """Application-level API gateway with centralized features."""

    def __init__(self):
        self._routes: dict[str, RouteConfig] = {}
        self._rate_counters: dict[str, list[float]] = defaultdict(list)
        self._cache: dict[str, tuple[float, dict]] = {}
        self._analytics = RequestAnalytics()
        self._latencies: list[float] = []

    def register_route(self, config: RouteConfig) -> None:
        """Register a route with gateway configuration."""
        self._routes[config.path_prefix] = config
        logger.info("gateway_route_registered", path=config.path_prefix, service=config.service)

    def check_rate_limit(self, client_id: str, path: str) -> bool:
        """Check if request is within rate limit."""
        route = self._find_route(path)
        if not route:
            return True

        now = time.time()
        window = 60.0  # 1 minute window
        key = f"{client_id}:{route.path_prefix}"

        # Clean old entries
        self._rate_counters[key] = [t for t in self._rate_counters[key] if now - t < window]

        if len(self._rate_counters[key]) >= route.rate_limit:
            logger.warning("rate_limit_exceeded", client=client_id, path=path)
            return False

        self._rate_counters[key].append(now)
        return True

    def get_cached(self, path: str) -> dict | None:
        """Get cached response if available."""
        if path in self._cache:
            expires, data = self._cache[path]
            if time.time() < expires:
                return data
            del self._cache[path]
        return None

    def set_cache(self, path: str, data: dict, ttl: int | None = None) -> None:
        """Cache a response."""
        route = self._find_route(path)
        cache_ttl = ttl or (route.cache_ttl if route else 0)
        if cache_ttl > 0:
            self._cache[path] = (time.time() + cache_ttl, data)

    def record_request(self, path: str, status_code: int, latency_ms: float) -> None:
        """Record request analytics."""
        self._analytics.total_requests += 1
        self._analytics.requests_by_path[path] += 1
        self._analytics.requests_by_status[str(status_code)] += 1

        if 200 <= status_code < 400:
            self._analytics.successful += 1
        else:
            self._analytics.failed += 1

        self._latencies.append(latency_ms)
        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-5000:]

        n = len(self._latencies)
        self._analytics.avg_latency_ms = sum(self._latencies) / n
        sorted_latencies = sorted(self._latencies)
        self._analytics.p95_latency_ms = sorted_latencies[int(n * 0.95)] if n > 0 else 0

    def _find_route(self, path: str) -> RouteConfig | None:
        for prefix, config in self._routes.items():
            if path.startswith(prefix):
                return config
        return None

    def get_analytics(self) -> dict:
        return {
            "total_requests": self._analytics.total_requests,
            "successful": self._analytics.successful,
            "failed": self._analytics.failed,
            "error_rate": round(self._analytics.failed / max(self._analytics.total_requests, 1) * 100, 2),
            "avg_latency_ms": round(self._analytics.avg_latency_ms, 1),
            "p95_latency_ms": round(self._analytics.p95_latency_ms, 1),
            "top_paths": dict(sorted(self._analytics.requests_by_path.items(), key=lambda x: x[1], reverse=True)[:10]),
            "status_codes": dict(self._analytics.requests_by_status),
            "cache_entries": len(self._cache),
            "routes_configured": len(self._routes),
        }

    def initialize_default_routes(self):
        """Register default API routes with gateway config."""
        routes = [
            RouteConfig("/api/v1/market", "market-service", rate_limit=200, cache_ttl=5),
            RouteConfig("/api/v1/portfolio", "portfolio-service", rate_limit=100, cache_ttl=10),
            RouteConfig("/api/v1/trading", "trading-service", rate_limit=50),
            RouteConfig("/api/v1/risk", "risk-service", rate_limit=100, cache_ttl=15),
            RouteConfig("/api/v1/signals", "ml-service", rate_limit=100, cache_ttl=30),
            RouteConfig("/api/v1/backtest", "backtest-service", rate_limit=20),
            RouteConfig("/api/v1/system", "system-service", rate_limit=200, cache_ttl=60, auth_required=False),
            RouteConfig("/api/v2", "v2-service", rate_limit=150),
        ]
        for route in routes:
            self.register_route(route)


api_gateway = APIGateway()

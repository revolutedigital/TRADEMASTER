"""Idempotency key middleware for safe request retries.

Supports pluggable storage backends:
- ``InMemoryIdempotencyStore`` – for development / single-instance deployments.
- ``RedisIdempotencyStore`` – for production / multi-instance deployments.

Use :func:`get_idempotency_store` to obtain the appropriate store.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 86_400  # 24 hours


# ---------------------------------------------------------------------------
# Store interface
# ---------------------------------------------------------------------------

class IdempotencyStore(ABC):
    """Abstract interface for idempotency response caching."""

    @abstractmethod
    async def get(self, key: str) -> dict | None:
        """Return the cached response dict for *key*, or ``None``."""

    @abstractmethod
    async def set(self, key: str, value: dict, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        """Persist *value* under *key* with a TTL in seconds."""


# ---------------------------------------------------------------------------
# In-memory store (development / tests)
# ---------------------------------------------------------------------------

class InMemoryIdempotencyStore(IdempotencyStore):
    """Simple dict-backed store.  Not suitable for multi-process deployments."""

    def __init__(self, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        self._cache: dict[str, dict] = {}
        self._ttl = ttl

    async def get(self, key: str) -> dict | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if (datetime.now(timezone.utc) - cached_at).total_seconds() > self._ttl:
            del self._cache[key]
            return None
        return entry

    async def set(self, key: str, value: dict, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        self._cache[key] = value
        self._cleanup()

    # -- internal --

    def _cleanup(self) -> None:
        """Remove expired entries from the in-memory cache."""
        now = datetime.now(timezone.utc)
        expired = [
            k
            for k, v in self._cache.items()
            if (now - datetime.fromisoformat(v["cached_at"])).total_seconds() > self._ttl
        ]
        for k in expired:
            del self._cache[k]


# ---------------------------------------------------------------------------
# Redis store (production)
# ---------------------------------------------------------------------------

class RedisIdempotencyStore(IdempotencyStore):
    """Redis-backed store using ``SETEX`` for automatic expiry.

    Args:
        redis_url: Redis connection string.  Defaults to ``settings.redis_url``.
        ttl: Default time-to-live in seconds for cached responses.
        key_prefix: Namespace prefix for Redis keys.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl: int = _DEFAULT_TTL_SECONDS,
        key_prefix: str = "idempotency:",
    ) -> None:
        self._redis_url = redis_url or settings.redis_url
        self._ttl = ttl
        self._prefix = key_prefix
        self._pool: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        """Lazy-initialise the async Redis connection pool."""
        if self._pool is None:
            self._pool = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
            )
            logger.info("redis_idempotency_store_connected", url=self._redis_url)
        return self._pool

    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> dict | None:
        client = await self._get_client()
        raw = await client.get(self._make_key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("redis_idempotency_deserialize_error", key=key)
            return None

    async def set(self, key: str, value: dict, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        client = await self._get_client()
        effective_ttl = ttl or self._ttl
        await client.setex(
            self._make_key(key),
            effective_ttl,
            json.dumps(value, default=str),
        )
        logger.debug("redis_idempotency_cached", key=key, ttl=effective_ttl)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_idempotency_store(use_redis: bool = True) -> IdempotencyStore:
    """Return the appropriate idempotency store backend.

    Args:
        use_redis: When ``True`` (the default), return a
            :class:`RedisIdempotencyStore`; otherwise fall back to
            :class:`InMemoryIdempotencyStore`.

    Returns:
        An :class:`IdempotencyStore` instance.
    """
    if use_redis:
        logger.info("idempotency_store_backend", backend="redis")
        return RedisIdempotencyStore()
    logger.info("idempotency_store_backend", backend="in-memory")
    return InMemoryIdempotencyStore()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Return cached responses for duplicate POST requests sharing the same
    ``X-Idempotency-Key`` header.

    Args:
        app: The ASGI application.
        store: An :class:`IdempotencyStore` backend.  When ``None``, an
            :class:`InMemoryIdempotencyStore` is used.
    """

    def __init__(self, app: Any, store: IdempotencyStore | None = None) -> None:
        super().__init__(app)
        self.store = store or InMemoryIdempotencyStore()

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method != "POST":
            return await call_next(request)

        key = request.headers.get("X-Idempotency-Key")
        if not key:
            return await call_next(request)

        # Check cache
        cached = await self.store.get(key)
        if cached:
            logger.info("idempotency_cache_hit", key=key[:16])
            return JSONResponse(
                status_code=cached["status_code"],
                content=cached["body"],
                headers={"X-Idempotency-Cached": "true"},
            )

        # Execute request
        response = await call_next(request)

        # Only cache successful responses
        if 200 <= response.status_code < 300:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            try:
                body_json = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body_json = {"raw": body.decode("utf-8", errors="replace")}

            entry = {
                "status_code": response.status_code,
                "body": body_json,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            await self.store.set(key, entry)

            return JSONResponse(status_code=response.status_code, content=body_json)

        return response

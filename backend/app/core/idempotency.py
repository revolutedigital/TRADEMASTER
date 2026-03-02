"""Idempotency key middleware for safe request retries."""
import hashlib
import json
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

# In-memory cache (use Redis in production for multi-instance)
_idempotency_cache: dict[str, dict] = {}
_CACHE_TTL_SECONDS = 86400  # 24 hours


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Return cached responses for duplicate POST requests with same idempotency key."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method != "POST":
            return await call_next(request)

        key = request.headers.get("X-Idempotency-Key")
        if not key:
            return await call_next(request)

        # Check cache
        cached = _idempotency_cache.get(key)
        if cached:
            logger.info("idempotency_cache_hit", key=key[:16])
            return JSONResponse(
                status_code=cached["status_code"],
                content=cached["body"],
                headers={"X-Idempotency-Cached": "true"},
            )

        # Execute and cache
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

            _idempotency_cache[key] = {
                "status_code": response.status_code,
                "body": body_json,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }

            # Cleanup old entries
            _cleanup_cache()

            return JSONResponse(status_code=response.status_code, content=body_json)

        return response


def _cleanup_cache():
    """Remove expired entries from the cache."""
    now = datetime.now(timezone.utc)
    expired = []
    for key, val in _idempotency_cache.items():
        cached_at = datetime.fromisoformat(val["cached_at"])
        if (now - cached_at).total_seconds() > _CACHE_TTL_SECONDS:
            expired.append(key)
    for key in expired:
        del _idempotency_cache[key]

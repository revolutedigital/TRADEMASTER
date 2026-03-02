"""RASP: Runtime Application Self-Protection.

Detects and blocks common attack patterns at the application layer.
"""

import re
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)

# Common SQL injection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r"(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC)\b.*\b(FROM|INTO|TABLE|WHERE)\b)", re.IGNORECASE),
    re.compile(r"(--|;|\/\*|\*\/|xp_|sp_)", re.IGNORECASE),
    re.compile(r"('\s*(OR|AND)\s*'?\s*\d+\s*=\s*\d+)", re.IGNORECASE),
    re.compile(r"(SLEEP\s*\(\s*\d+\s*\)|BENCHMARK\s*\()", re.IGNORECASE),
]

# Common XSS patterns
XSS_PATTERNS = [
    re.compile(r"<script[^>]*>", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"on(load|error|click|mouse|focus|blur)\s*=", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\./"),
    re.compile(r"\.\.\\"),
    re.compile(r"%2e%2e", re.IGNORECASE),
]


def _check_patterns(value: str, patterns: list[re.Pattern], attack_type: str) -> str | None:
    """Check a value against attack patterns. Returns the attack type if matched."""
    for pattern in patterns:
        if pattern.search(value):
            return attack_type
    return None


def _scan_request_data(data: str) -> str | None:
    """Scan request data for attack patterns."""
    result = _check_patterns(data, SQL_INJECTION_PATTERNS, "sql_injection")
    if result:
        return result
    result = _check_patterns(data, XSS_PATTERNS, "xss")
    if result:
        return result
    result = _check_patterns(data, PATH_TRAVERSAL_PATTERNS, "path_traversal")
    if result:
        return result
    return None


class RASPMiddleware(BaseHTTPMiddleware):
    """Runtime Application Self-Protection middleware.

    Scans incoming requests for common attack patterns (SQLi, XSS, path traversal)
    and blocks them before they reach application code.
    """

    EXEMPT_PATHS = {"/docs", "/redoc", "/openapi.json", "/api/v1/system/health"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Check URL path
        attack = _scan_request_data(str(request.url))
        if attack:
            logger.warning(
                "rasp_blocked",
                attack_type=attack,
                path=request.url.path,
                ip=request.client.host if request.client else "unknown",
            )
            return JSONResponse(status_code=403, content={"error": "request_blocked", "message": "Suspicious request detected"})

        # Check query parameters
        for key, value in request.query_params.items():
            attack = _scan_request_data(f"{key}={value}")
            if attack:
                logger.warning(
                    "rasp_blocked",
                    attack_type=attack,
                    param=key,
                    path=request.url.path,
                    ip=request.client.host if request.client else "unknown",
                )
                return JSONResponse(status_code=403, content={"error": "request_blocked", "message": "Suspicious request detected"})

        # Check headers for injection
        for header_name in ("referer", "user-agent", "x-forwarded-for"):
            header_value = request.headers.get(header_name, "")
            if header_value:
                attack = _scan_request_data(header_value)
                if attack:
                    logger.warning(
                        "rasp_blocked",
                        attack_type=attack,
                        header=header_name,
                        path=request.url.path,
                        ip=request.client.host if request.client else "unknown",
                    )
                    return JSONResponse(status_code=403, content={"error": "request_blocked", "message": "Suspicious request detected"})

        return await call_next(request)

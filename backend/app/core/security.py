"""Security utilities: JWT tokens, password hashing, API key encryption, middleware."""

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from cryptography.fernet import Fernet
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings


# --- JWT ---

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode["exp"] = expire
    to_encode["type"] = "access"
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days)
    to_encode["exp"] = expire
    to_encode["type"] = "refresh"
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str, token_type: str = "access") -> dict | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type", "access") != token_type:
            return None
        return payload
    except JWTError:
        return None


# --- Password ---

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# --- API Key Encryption ---

class APIKeyEncryptor:
    """Encrypt/decrypt API keys at rest using Fernet symmetric encryption."""

    def __init__(self, key: bytes | None = None):
        self._fernet = Fernet(key or Fernet.generate_key())

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode()).decode()


# --- CSRF ---

def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def validate_csrf_token(request_token: str | None, cookie_token: str | None) -> bool:
    if not request_token or not cookie_token:
        return False
    return secrets.compare_digest(request_token, cookie_token)


# --- CSRF Middleware ---

# Paths exempt from CSRF validation (login sets the cookie, docs are read-only)
CSRF_EXEMPT_PATHS: set[str] = {
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/system/health",
    "/api/v1/system/health/detailed",
}


class CSRFMiddleware(BaseHTTPMiddleware):
    """Validate CSRF token on state-changing requests (POST/PUT/DELETE/PATCH).

    Compares the ``csrf_token`` cookie against the ``X-CSRF-Token`` header
    using constant-time comparison to prevent timing attacks.
    GET/HEAD/OPTIONS and explicitly exempted paths are allowed through.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            # Allow exempt paths (auth endpoints, docs, health)
            if request.url.path not in CSRF_EXEMPT_PATHS:
                csrf_cookie = request.cookies.get("csrf_token")
                csrf_header = request.headers.get("X-CSRF-Token")
                if not validate_csrf_token(csrf_header, csrf_cookie):
                    from starlette.responses import JSONResponse
                    return JSONResponse(
                        status_code=403,
                        content={"error": "csrf_validation_failed", "message": "CSRF token missing or invalid"},
                    )
        return await call_next(request)


# --- Security Headers Middleware ---

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' wss: ws: https:; "
            "font-src 'self' data:; "
            "frame-ancestors 'none'"
        )
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


# --- WebSocket Auth Helper ---

async def authenticate_websocket(websocket) -> dict | None:
    """Extract and verify JWT from WebSocket connection.
    Checks query param 'token' or Authorization header.
    """
    token = websocket.query_params.get("token")
    if not token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        return None
    return verify_token(token)


# --- Rate Limiter ---

class AppRateLimiter:
    """In-memory rate limiter for API endpoints."""

    def __init__(self):
        self._requests: dict[str, list[float]] = {}

    def _cleanup(self, key: str, window_seconds: int) -> None:
        now = datetime.now(timezone.utc).timestamp()
        if key in self._requests:
            self._requests[key] = [
                t for t in self._requests[key]
                if now - t < window_seconds
            ]

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        self._cleanup(key, window_seconds)
        if key not in self._requests:
            self._requests[key] = []
        if len(self._requests[key]) >= max_requests:
            return False
        self._requests[key].append(datetime.now(timezone.utc).timestamp())
        return True


app_rate_limiter = AppRateLimiter()

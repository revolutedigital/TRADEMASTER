"""Authentication endpoints: login, token refresh, and logout."""

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request, Response, status

from app.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    generate_csrf_token,
    verify_token,
    app_rate_limiter,
    brute_force,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)
    totp_code: str | None = Field(default=None, min_length=6, max_length=6, description="6-digit TOTP code (required when 2FA is enabled)")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request, response: Response):
    """Authenticate and return a JWT token. Also sets httpOnly cookie."""
    # Rate limit: 5 attempts per minute per IP
    client_ip = request.client.host if request.client else "unknown"
    if not await app_rate_limiter.is_allowed_async(f"login:{client_ip}", max_requests=5, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again in 1 minute.",
        )

    # Brute-force protection: progressive delays + lockout
    bf_allowed, bf_wait = brute_force.check(client_ip)
    if not bf_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Account locked. Try again in {bf_wait} seconds.",
        )
    if bf_wait > 0:
        import asyncio
        await asyncio.sleep(bf_wait)

    if req.username != settings.admin_username or req.password != settings.admin_password:
        brute_force.record_failure(client_ip)
        from app.core.audit import audit_logger
        await audit_logger.log_login(req.username, success=False, ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # --- Optional TOTP 2FA check ---
    if settings.totp_enabled and settings.totp_secret:
        if not req.totp_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="TOTP code required (2FA is enabled)",
            )
        from app.core.totp import TOTPManager
        if not TOTPManager.verify_totp(settings.totp_secret, req.totp_code):
            brute_force.record_failure(client_ip)
            from app.core.audit import audit_logger
            await audit_logger.log_login(req.username, success=False, ip=client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid TOTP code",
            )

    brute_force.record_success(client_ip)
    token_data = {"sub": req.username, "role": "admin"}
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)
    csrf_token = generate_csrf_token()

    # Cross-origin cookies require SameSite=None + Secure because
    # up.railway.app is on the Public Suffix List, making frontend/backend
    # subdomains different "sites" in the browser's cookie model.
    _ck = {"secure": True, "samesite": "none"}

    response.set_cookie(
        key="access_token", value=access_token,
        httponly=True, max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/", **_ck,
    )
    response.set_cookie(
        key="refresh_token", value=refresh_token,
        httponly=True, max_age=settings.jwt_refresh_token_expire_days * 86400,
        path="/", **_ck,
    )
    response.set_cookie(
        key="csrf_token", value=csrf_token,
        httponly=False, max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/", **_ck,
    )

    from app.core.audit import audit_logger
    await audit_logger.log_login(req.username, success=True, ip=client_ip)

    # Also return token in body for backward compatibility
    return TokenResponse(access_token=access_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: Request, response: Response):
    """Refresh access token using refresh token from cookie."""
    refresh = request.cookies.get("refresh_token")
    if not refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No refresh token",
        )

    payload = verify_token(refresh, token_type="refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    token_data = {"sub": payload["sub"], "role": payload.get("role", "admin")}
    new_access = create_access_token(data=token_data)
    new_csrf = generate_csrf_token()

    _ck = {"secure": True, "samesite": "none"}
    response.set_cookie(
        key="access_token", value=new_access,
        httponly=True, max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/", **_ck,
    )
    response.set_cookie(
        key="csrf_token", value=new_csrf,
        httponly=False, max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/", **_ck,
    )

    return TokenResponse(access_token=new_access)


@router.post("/logout")
async def logout(response: Response):
    """Clear authentication cookies."""
    _ck = {"secure": True, "samesite": "none"}
    response.delete_cookie("access_token", path="/", **_ck)
    response.delete_cookie("refresh_token", path="/", **_ck)
    response.delete_cookie("csrf_token", path="/", **_ck)
    return {"status": "logged_out"}

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
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request, response: Response):
    """Authenticate and return a JWT token. Also sets httpOnly cookie."""
    # Rate limit: 5 attempts per minute per IP
    client_ip = request.client.host if request.client else "unknown"
    if not app_rate_limiter.is_allowed(f"login:{client_ip}", max_requests=5, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again in 1 minute.",
        )

    if req.username != settings.admin_username or req.password != settings.admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token_data = {"sub": req.username, "role": "admin"}
    access_token = create_access_token(data=token_data)
    refresh_token = create_refresh_token(data=token_data)
    csrf_token = generate_csrf_token()

    # Set httpOnly cookie for access token
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/",
    )
    # Set refresh token in httpOnly cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        max_age=settings.jwt_refresh_token_expire_days * 86400,
        path="/api/v1/auth",
    )
    # CSRF token readable by JavaScript
    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        httponly=False,
        secure=settings.is_production,
        samesite="lax",
        max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/",
    )

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

    response.set_cookie(
        key="access_token",
        value=new_access,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/",
    )
    response.set_cookie(
        key="csrf_token",
        value=new_csrf,
        httponly=False,
        secure=settings.is_production,
        samesite="lax",
        max_age=settings.jwt_access_token_expire_minutes * 60,
        path="/",
    )

    return TokenResponse(access_token=new_access)


@router.post("/logout")
async def logout(response: Response):
    """Clear authentication cookies."""
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/api/v1/auth")
    response.delete_cookie("csrf_token", path="/")
    return {"status": "logged_out"}

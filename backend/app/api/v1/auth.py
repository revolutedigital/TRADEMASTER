"""Authentication endpoints: login and token refresh."""

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.core.security import create_access_token, verify_password, hash_password

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticate and return a JWT token."""
    if req.username != settings.admin_username or req.password != settings.admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token(data={"sub": req.username, "role": "admin"})
    return TokenResponse(access_token=token)

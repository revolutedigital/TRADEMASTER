"""API Key management endpoints."""

import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_auth
from app.models.api_key import APIKey

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class CreateKeyRequest(BaseModel):
    name: str
    scopes: str = "read"  # comma-separated: read,trade,admin


class APIKeyResponse(BaseModel):
    id: int
    name: str
    key_prefix: str
    scopes: str
    is_active: bool
    last_used_at: str | None
    created_at: str


@router.post("")
async def create_api_key(
    body: CreateKeyRequest,
    db: AsyncSession = Depends(get_db),
    auth: dict = Depends(require_auth),
):
    raw_key = f"tm_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = APIKey(
        user_id=auth.get("sub", "admin"),
        key_hash=key_hash,
        key_prefix=raw_key[:8],
        name=body.name,
        scopes=body.scopes,
    )
    db.add(api_key)
    await db.commit()

    return {"id": api_key.id, "key": raw_key, "name": body.name, "scopes": body.scopes}


@router.get("")
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    auth: dict = Depends(require_auth),
):
    result = await db.execute(
        select(APIKey)
        .where(APIKey.user_id == auth.get("sub", "admin"))
        .order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        {
            "id": k.id,
            "name": k.name,
            "key_prefix": k.key_prefix,
            "scopes": k.scopes,
            "is_active": k.is_active,
            "last_used_at": str(k.last_used_at) if k.last_used_at else None,
            "created_at": str(k.created_at),
        }
        for k in keys
    ]


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    auth: dict = Depends(require_auth),
):
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == auth.get("sub", "admin"))
    )
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    await db.commit()
    return {"status": "revoked"}

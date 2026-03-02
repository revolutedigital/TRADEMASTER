"""Security.txt endpoint per RFC 9116."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["security"])


SECURITY_TXT = """Contact: security@trademaster.app
Expires: 2027-01-01T00:00:00.000Z
Preferred-Languages: en, pt
Policy: https://trademaster.app/security-policy
Canonical: https://trademaster.app/.well-known/security.txt
"""


@router.get("/.well-known/security.txt", response_class=PlainTextResponse)
async def security_txt():
    return SECURITY_TXT

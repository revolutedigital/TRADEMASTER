"""Honeypot endpoints to detect and alert on attacker probing."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

honeypot_router = APIRouter(tags=["honeypot"], include_in_schema=False)


async def _log_attacker(request: Request, path: str) -> None:
    """Log details of attacker hitting honeypot endpoint."""
    logger.warning(
        "honeypot_triggered",
        path=path,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
        method=request.method,
        headers=dict(request.headers),
    )


@honeypot_router.get("/admin/login")
@honeypot_router.post("/admin/login")
async def fake_admin_login(request: Request):
    await _log_attacker(request, "/admin/login")
    return HTMLResponse(
        '<html><body><h1>Admin Login</h1><form method="post">'
        '<input name="username" placeholder="Username"/>'
        '<input name="password" type="password" placeholder="Password"/>'
        '<button type="submit">Login</button></form></body></html>',
        status_code=200,
    )


@honeypot_router.get("/wp-admin")
@honeypot_router.get("/wp-login.php")
async def fake_wordpress(request: Request):
    await _log_attacker(request, request.url.path)
    return HTMLResponse("<html><body><h1>WordPress</h1></body></html>", status_code=200)


@honeypot_router.get("/.env")
async def fake_env(request: Request):
    await _log_attacker(request, "/.env")
    return HTMLResponse(
        "APP_KEY=base64:FAKE_KEY_FOR_HONEYPOT\nDB_PASSWORD=honeypot_trap\n",
        status_code=200,
        media_type="text/plain",
    )


@honeypot_router.get("/phpmyadmin")
@honeypot_router.get("/phpMyAdmin")
async def fake_phpmyadmin(request: Request):
    await _log_attacker(request, request.url.path)
    return HTMLResponse("<html><body><h1>phpMyAdmin</h1></body></html>", status_code=200)


@honeypot_router.get("/.git/config")
async def fake_git(request: Request):
    await _log_attacker(request, "/.git/config")
    return HTMLResponse("[core]\n\trepositoryformatversion = 0\n", status_code=200, media_type="text/plain")


@honeypot_router.get("/actuator")
@honeypot_router.get("/actuator/health")
async def fake_actuator(request: Request):
    await _log_attacker(request, request.url.path)
    return {"status": "UP"}

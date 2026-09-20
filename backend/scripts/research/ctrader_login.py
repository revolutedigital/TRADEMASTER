"""Interactive OAuth login for the cTrader Open API: you sign in with your own account in the browser.

It opens the consent page (Google sign-in works there), catches the redirect on a local port, swaps
the one-time code for an access and a refresh token and stores them in
`~/.config/trademaster/ctrader.env` (mode 600). The client id and secret are read from that file or
asked for on the terminal (the secret is not echoed). Nothing secret is printed.

Run it in your own terminal: `cd backend && .venv/bin/python -m scripts.research.ctrader_login`.
"""

from __future__ import annotations

import getpass
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from scripts.research.ctrader_accounts import ENV_FILE, load_environment

AUTHORIZE_URL = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
TOKEN_URL = "https://openapi.ctrader.com/apps/token"
DEFAULT_REDIRECT = "http://localhost:8080/callback"
WAIT_SECONDS = 300


class LoginError(Exception):
    pass


def authorization_url(client_id: str, redirect_uri: str, scope: str = "trading") -> str:
    query = urlencode({"client_id": client_id, "redirect_uri": redirect_uri, "scope": scope, "product": "web"})
    return f"{AUTHORIZE_URL}?{query}"


def code_from_query(query: str) -> str:
    """The authorization code in a redirect's query string, or a clear error."""
    values = parse_qs(query)
    if "error" in values:
        raise LoginError(f"authorization refused: {values['error'][0]}")
    if "code" not in values:
        raise LoginError("the redirect carried no authorization code")
    return values["code"][0]


def wait_for_code(redirect_uri: str, timeout: float = WAIT_SECONDS) -> str:
    """Serve one request on the redirect's local port and return the code it carries."""
    target = urlparse(redirect_uri)
    outcome: dict[str, str] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - the http.server naming
            if urlparse(self.path).path != target.path:
                self.send_response(404)
                self.end_headers()
                return
            try:
                outcome["code"] = code_from_query(urlparse(self.path).query)
                body = "Login concluído. Pode fechar esta aba."
            except LoginError as error:
                outcome["error"] = str(error)
                body = "Login não concluído. Volte ao terminal."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
            done.set()

        def log_message(self, *args: object) -> None:  # keep the code out of any log
            return

    server = HTTPServer((target.hostname or "localhost", target.port or 80), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        if not done.wait(timeout):
            raise LoginError("no redirect arrived in time")
    finally:
        server.shutdown()
        server.server_close()
    if "error" in outcome:
        raise LoginError(outcome["error"])
    return outcome["code"]


def exchange_code(
    code: str, client_id: str, client_secret: str, redirect_uri: str, client: httpx.Client | None = None
) -> dict[str, str]:
    params = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
              "client_id": client_id, "client_secret": client_secret}
    response = (client or httpx).get(TOKEN_URL, params=params, timeout=30)
    data = response.json() if response.content else {}
    if response.status_code != 200 or data.get("errorCode") or "accessToken" not in data:
        raise LoginError(f"token exchange failed: {data.get('errorCode') or response.status_code}")
    return {"CTRADER_ACCESS_TOKEN": data["accessToken"], "CTRADER_REFRESH_TOKEN": data.get("refreshToken", "")}


def save_environment(updates: dict[str, str], path: Path = ENV_FILE) -> None:
    """Merge `updates` into the private env file, keeping other lines, readable only by you."""
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [line for line in existing if line.partition("=")[0].strip() not in updates]
    lines = [*kept, *(f"{key}={value}" for key, value in updates.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(temporary, path)


def main() -> int:
    values = load_environment()
    client_id = values.get("CTRADER_CLIENT_ID") or input("client_id: ").strip()
    client_secret = values.get("CTRADER_CLIENT_SECRET") or getpass.getpass("client_secret (não aparece): ").strip()
    redirect_uri = input(f"redirect URI cadastrada [{DEFAULT_REDIRECT}]: ").strip() or DEFAULT_REDIRECT
    url = authorization_url(client_id, redirect_uri)
    sys.stdout.write(f"Abrindo o navegador. Se não abrir, use este endereço:\n{url}\n")
    webbrowser.open(url)
    try:
        code = wait_for_code(redirect_uri)
        tokens = exchange_code(code, client_id, client_secret, redirect_uri)
    except LoginError as error:
        sys.stdout.write(f"falhou: {error}\n")
        return 1
    save_environment({"CTRADER_CLIENT_ID": client_id, "CTRADER_CLIENT_SECRET": client_secret, **tokens})
    sys.stdout.write(f"Tokens gravados em {ENV_FILE} (só você lê). Nada foi mostrado na tela.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

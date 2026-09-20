"""The OAuth helper: URL, redirect capture, code exchange and the private file."""

import stat
import threading
import time

import httpx
import pytest

from scripts.research import ctrader_login as login


def test_the_authorization_url_asks_for_trading_on_the_registered_redirect() -> None:
    url = login.authorization_url("cid_1", "http://localhost:8080/callback")

    assert url.startswith("https://id.ctrader.com/my/settings/openapi/grantingaccess/?")
    assert "client_id=cid_1" in url and "scope=trading" in url and "redirect_uri=http%3A%2F%2Flocalhost%3A8080%2Fcallback" in url


def test_the_code_or_the_refusal_is_read_from_the_redirect() -> None:
    assert login.code_from_query("code=abc123&state=x") == "abc123"
    with pytest.raises(login.LoginError, match="access_denied"):
        login.code_from_query("error=access_denied")
    with pytest.raises(login.LoginError, match="no authorization code"):
        login.code_from_query("")


def test_the_local_server_catches_the_redirect_and_returns_the_code() -> None:
    found: dict[str, str] = {}
    redirect = "http://127.0.0.1:18765/callback"
    thread = threading.Thread(target=lambda: found.update(code=login.wait_for_code(redirect, timeout=10)))
    thread.start()
    time.sleep(0.4)

    answer = httpx.get(redirect + "?code=the-code", timeout=5)
    thread.join(10)

    assert answer.status_code == 200 and found["code"] == "the-code"


def test_the_exchange_returns_both_tokens_and_a_refusal_carries_only_the_code() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        assert request.url.params["grant_type"] == "authorization_code" and request.url.params["code"] == "c1"
        return httpx.Response(200, json={"accessToken": "AT", "refreshToken": "RT", "expiresIn": 2628000})

    def refused(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errorCode": "CH_ACCESS_TOKEN_INVALID", "description": "secret S3CRET leaked here"})

    good = login.exchange_code("c1", "id", "S3CRET", "http://localhost/cb", httpx.Client(transport=httpx.MockTransport(ok)))
    assert good == {"CTRADER_ACCESS_TOKEN": "AT", "CTRADER_REFRESH_TOKEN": "RT"}
    with pytest.raises(login.LoginError) as error:
        login.exchange_code("c1", "id", "S3CRET", "http://localhost/cb", httpx.Client(transport=httpx.MockTransport(refused)))
    assert "CH_ACCESS_TOKEN_INVALID" in str(error.value) and "S3CRET" not in str(error.value)


def test_the_env_file_is_private_and_merges_instead_of_overwriting(tmp_path) -> None:
    path = tmp_path / "sub" / "ctrader.env"
    login.save_environment({"CTRADER_CLIENT_ID": "id", "CTRADER_ACCESS_TOKEN": "old"}, path)

    login.save_environment({"CTRADER_ACCESS_TOKEN": "new", "CTRADER_REFRESH_TOKEN": "rt"}, path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    lines = dict(line.split("=", 1) for line in path.read_text().splitlines())
    assert lines == {"CTRADER_CLIENT_ID": "id", "CTRADER_ACCESS_TOKEN": "new", "CTRADER_REFRESH_TOKEN": "rt"}

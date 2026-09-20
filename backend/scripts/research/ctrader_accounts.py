"""List the trading accounts an Open API access token can reach, on the demo endpoint only.

It authenticates the application, asks which accounts the token opens and prints their numeric
ids (the `CTRADER_ACCOUNT_ID` the probe needs), whether each is live or demo and its login number.
Reads CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET and CTRADER_ACCESS_TOKEN from the environment or
from `~/.config/trademaster/ctrader.env`, and never prints any of them.

Nothing here touches the trading engine, the database, or an exchange order book.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import sys
from pathlib import Path

from app.fx.runner.ctrader.client import (
    DEMO_HOST,
    PORT,
    Credentials,
    encode_frame,
    payload_type_of,
    read_frame,
    server_error,
)
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages

ENV_FILE = Path.home() / ".config" / "trademaster" / "ctrader.env"
REQUIRED = ("CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN")
TIMEOUT_SECONDS = 20.0


def load_environment(path: Path = ENV_FILE) -> dict[str, str]:
    """The process environment, completed with the KEY=VALUE lines of the private env file."""
    values = dict(os.environ)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() and not key.startswith("#"):
                values.setdefault(key.strip(), value.strip())
    return values


async def _ask(reader, writer, message, expect, credentials: Credentials):
    writer.write(encode_frame(payload_type_of(message), message, "1"))
    await writer.drain()
    while True:
        envelope = await asyncio.wait_for(read_frame(reader), TIMEOUT_SECONDS)
        if envelope.payloadType == payload_type_of(expect()):
            answer = expect()
            answer.ParseFromString(envelope.payload)
            return answer
        for candidate in (messages.ProtoOAErrorRes,):
            if envelope.payloadType == payload_type_of(candidate()):
                error = candidate()
                error.ParseFromString(envelope.payload)
                found = server_error(error)
                raise RuntimeError(credentials.redact(f"{found.code}: {found.description}"))


async def list_accounts(
    credentials: Credentials, host: str = DEMO_HOST, port: int = PORT, *, use_tls: bool = True
) -> list[dict[str, object]]:
    reader, writer = await asyncio.open_connection(
        host, port, ssl=ssl.create_default_context() if use_tls else None
    )
    try:
        await _ask(
            reader, writer,
            messages.ProtoOAApplicationAuthReq(
                clientId=credentials.client_id, clientSecret=credentials.client_secret
            ),
            messages.ProtoOAApplicationAuthRes, credentials,
        )
        answer = await _ask(
            reader, writer,
            messages.ProtoOAGetAccountListByAccessTokenReq(accessToken=credentials.access_token),
            messages.ProtoOAGetAccountListByAccessTokenRes, credentials,
        )
    finally:
        writer.close()
    return [
        {"account_id": a.ctidTraderAccountId, "live": a.isLive, "login": a.traderLogin,
         "broker": a.brokerTitleShort}
        for a in answer.ctidTraderAccount
    ]


def main() -> int:
    values = load_environment()
    missing = [name for name in REQUIRED if not values.get(name)]
    if missing:
        sys.stdout.write(f"missing: {', '.join(missing)}\n")
        return 2
    credentials = Credentials(
        values["CTRADER_CLIENT_ID"], values["CTRADER_CLIENT_SECRET"], values["CTRADER_ACCESS_TOKEN"], 0
    )
    try:
        accounts = asyncio.run(list_accounts(credentials))
    except (RuntimeError, OSError, TimeoutError) as error:
        sys.stdout.write(f"failed: {credentials.redact(str(error))}\n")
        return 1
    for account in accounts:
        sys.stdout.write(
            f"account_id={account['account_id']} {'LIVE' if account['live'] else 'demo'} "
            f"login={account['login']} broker={account['broker']}\n"
        )
    sys.stdout.write(f"{len(accounts)} account(s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

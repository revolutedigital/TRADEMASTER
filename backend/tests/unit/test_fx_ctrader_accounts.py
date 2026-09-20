"""The account-listing script against a tiny local server, and its handling of secrets."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from app.fx.runner.ctrader.client import Credentials, encode_frame, payload_type_of, read_frame
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from scripts.research import ctrader_accounts as accounts

CREDENTIALS = Credentials("cid-123_abc", "SECRET-xyz", "TOKEN-789", 0)


@asynccontextmanager
async def server(refuse_application: bool = False):
    async def handle(reader, writer):
        try:
            while True:
                envelope = await read_frame(reader)
                if envelope.payloadType == payload_type_of(messages.ProtoOAApplicationAuthReq()):
                    if refuse_application:
                        reply = messages.ProtoOAErrorRes(
                            errorCode="CH_CLIENT_AUTH_FAILURE", description="bad secret SECRET-xyz", ctidTraderAccountId=0
                        )
                    else:
                        reply = messages.ProtoOAApplicationAuthRes()
                else:
                    reply = messages.ProtoOAGetAccountListByAccessTokenRes(accessToken="TOKEN-789")
                    account = reply.ctidTraderAccount.add()
                    account.ctidTraderAccountId, account.isLive, account.traderLogin = 4242, False, 999001
                    account.brokerTitleShort = "Fusion"
                writer.write(encode_frame(payload_type_of(reply), reply, envelope.clientMsgId))
                await writer.drain()
        except asyncio.IncompleteReadError:
            writer.close()

    running = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield running.sockets[0].getsockname()[1]
    finally:
        running.close()
        await running.wait_closed()


async def test_it_lists_the_demo_account_numbers_of_the_token() -> None:
    async with server() as port:
        found = await accounts.list_accounts(CREDENTIALS, "127.0.0.1", port, use_tls=False)

    assert found == [{"account_id": 4242, "live": False, "login": 999001, "broker": "Fusion"}]


async def test_a_refused_application_raises_without_leaking_the_secret() -> None:
    async with server(refuse_application=True) as port:
        with pytest.raises(RuntimeError) as error:
            await accounts.list_accounts(CREDENTIALS, "127.0.0.1", port, use_tls=False)

    assert "CH_CLIENT_AUTH_FAILURE" in str(error.value) and "SECRET-xyz" not in str(error.value)


def test_the_private_env_file_completes_the_environment_and_the_process_wins(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ctrader.env"
    path.write_text("# comment\nCTRADER_CLIENT_ID=from-file\nCTRADER_CLIENT_SECRET=s=with=equals\n\n", encoding="utf-8")
    monkeypatch.setenv("CTRADER_CLIENT_ID", "from-process")

    values = accounts.load_environment(path)

    assert values["CTRADER_CLIENT_ID"] == "from-process"
    assert values["CTRADER_CLIENT_SECRET"] == "s=with=equals"
    assert accounts.load_environment(tmp_path / "missing.env")["CTRADER_CLIENT_ID"] == "from-process"

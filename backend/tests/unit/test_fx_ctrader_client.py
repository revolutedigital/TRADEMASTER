"""The cTrader transport: framing, correlation, auth, heartbeat, rate limit and reconnection."""

import asyncio
import logging
import struct

import pytest

from app.fx.runner.ctrader.client import (
    MAX_FRAME_BYTES,
    CTraderAuthError,
    CTraderRequestError,
    Credentials,
    FrameError,
    RateLimiter,
    encode_frame,
    payload_type_of,
    read_frame,
)
from app.fx.runner.ctrader.proto import OpenApiCommonMessages_pb2 as common
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from app.fx.runner.venue import VenueUnavailable
from tests.unit.fx_ctrader_server import FakeCTraderServer, client_for, eventually

ACCOUNT = 1001
TRADER_REQ = payload_type_of(messages.ProtoOATraderReq())
SYMBOLS_REQ = payload_type_of(messages.ProtoOASymbolsListReq())
APPLICATION_AUTH_REQ = payload_type_of(messages.ProtoOAApplicationAuthReq())


@pytest.fixture
async def server():
    async with FakeCTraderServer() as running:
        yield running


def trader_request() -> messages.ProtoOATraderReq:
    return messages.ProtoOATraderReq(ctidTraderAccountId=ACCOUNT)


# framing


def test_a_frame_is_four_big_endian_length_bytes_then_the_envelope() -> None:
    # a heartbeat is payloadType 51: field 1, varint -> 0x08 0x33; two bytes long
    assert encode_frame(51, None, None) == bytes([0, 0, 0, 2, 0x08, 0x33])
    request = messages.ProtoOAReconcileReq(ctidTraderAccountId=7)
    envelope = common.ProtoMessage(
        payloadType=2124, payload=request.SerializeToString(), clientMsgId="m1"
    )
    body = envelope.SerializeToString()
    assert encode_frame(2124, request, "m1") == struct.pack(">I", len(body)) + body


async def test_a_frame_delivered_in_small_pieces_is_reassembled() -> None:
    reader = asyncio.StreamReader()
    request = messages.ProtoOAReconcileReq(ctidTraderAccountId=7)
    wire = encode_frame(2124, request, "m1") + encode_frame(51, None, None)

    async def feed() -> None:
        for start in range(0, len(wire), 3):
            reader.feed_data(wire[start : start + 3])
            await asyncio.sleep(0)

    feeder = asyncio.create_task(feed())
    first, second = await read_frame(reader), await read_frame(reader)
    await feeder

    assert (first.payloadType, first.clientMsgId) == (2124, "m1")
    assert messages.ProtoOAReconcileReq.FromString(first.payload).ctidTraderAccountId == 7
    assert second.payloadType == 51


async def test_two_frames_that_arrive_in_one_chunk_are_split() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(encode_frame(51, None, None) * 2)

    assert [(await read_frame(reader)).payloadType for _ in range(2)] == [51, 51]


async def test_an_oversized_or_garbled_frame_is_a_frame_error() -> None:
    oversized = asyncio.StreamReader()
    oversized.feed_data(struct.pack(">I", MAX_FRAME_BYTES + 1))
    garbled = asyncio.StreamReader()
    garbled.feed_data(struct.pack(">I", 3) + b"\xff\xff\xff")

    with pytest.raises(FrameError):
        await read_frame(oversized)
    with pytest.raises(FrameError):
        await read_frame(garbled)


async def test_a_link_that_closes_inside_a_frame_is_not_a_frame() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack(">I", 10) + b"ab")
    reader.feed_eof()

    with pytest.raises(asyncio.IncompleteReadError):
        await read_frame(reader)


async def test_requests_work_when_the_server_writes_one_byte_at_a_time(server) -> None:
    server.chunk_size = 1

    async with client_for(server) as client:
        response = await client.request(trader_request(), messages.ProtoOATraderRes)

    assert response.trader.balance == 50_000


# correlation and timeouts


async def test_answers_are_matched_by_client_msg_id_even_when_they_arrive_out_of_order(
    server,
) -> None:
    server.delays[SYMBOLS_REQ] = 0.15
    async with client_for(server) as client:
        slow = asyncio.create_task(
            client.request(
                messages.ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT),
                messages.ProtoOASymbolsListRes,
            )
        )
        await asyncio.sleep(0.02)
        trader = await client.request(trader_request(), messages.ProtoOATraderRes)
        assert not slow.done()
        listing = await slow

    assert trader.trader.balance == 50_000
    assert len(listing.symbol) == 4


async def test_an_unanswered_request_is_venue_unavailable_and_leaves_nothing_pending(
    server,
) -> None:
    server.silent.add(TRADER_REQ)
    async with client_for(server, request_timeout=0.2) as client:
        with pytest.raises(VenueUnavailable, match="did not answer"):
            await client.request(trader_request(), messages.ProtoOATraderRes)

        assert client._pending == {}
        server.silent.clear()
        assert (await client.request(trader_request(), messages.ProtoOATraderRes)).trader


async def test_an_error_answer_raises_with_the_server_code(server) -> None:
    server.fail_payload(TRADER_REQ, "CH_SERVER_NOT_REACHABLE")
    async with client_for(server) as client:
        with pytest.raises(CTraderRequestError) as caught:
            await client.request(trader_request(), messages.ProtoOATraderRes)

    assert caught.value.code == "CH_SERVER_NOT_REACHABLE"
    assert isinstance(caught.value, VenueUnavailable)


async def test_a_message_the_client_does_not_know_is_ignored_and_a_broken_one_drops_the_link(
    server,
) -> None:
    async with client_for(server) as client:
        await server.send_frame(9999, b"from the future")
        assert (await client.request(trader_request(), messages.ProtoOATraderRes)).trader
        assert server.connection_count == 1

        await server.send_frame(2126, b"\xff\xff\xff")  # an ExecutionEvent that is not protobuf
        await eventually(lambda: server.connection_count == 2 and client.connected)


async def test_a_failing_listener_does_not_end_the_session(server) -> None:
    def broken(message, client_msg_id) -> None:
        raise RuntimeError("listener bug")

    async with client_for(server) as client:
        client.add_listener(broken)
        await server.push_spot("EURUSD", 1.1, 1.10008)
        await client.subscribe_spots([1])
        await asyncio.sleep(0.05)

        assert client.connected and server.connection_count == 1


# authentication


async def test_the_session_authenticates_the_application_then_the_account(server) -> None:
    async with client_for(server) as client:
        assert client.connected

    application = server.requests_of(messages.ProtoOAApplicationAuthReq)
    account = server.requests_of(messages.ProtoOAAccountAuthReq)
    assert [(a.clientId, a.clientSecret) for a in application] == [
        (server.client_id, server.client_secret)
    ]
    assert [(a.ctidTraderAccountId, a.accessToken) for a in account] == [
        (ACCOUNT, server.access_token)
    ]


async def test_refused_credentials_are_fatal_and_are_not_retried(server) -> None:
    wrong = Credentials(server.client_id, "wrong-secret-value", server.access_token, ACCOUNT)
    client = client_for(server, wrong)

    with pytest.raises(CTraderAuthError, match="CH_CLIENT_AUTH_FAILURE"):
        await client.start()
    await asyncio.sleep(0.2)

    assert server.connection_count == 1  # no reconnection loop
    assert client.fatal_error is not None
    with pytest.raises(CTraderAuthError):
        await client.request(trader_request(), messages.ProtoOATraderRes)


async def test_a_token_the_server_refuses_is_fatal_and_its_echo_is_scrubbed(server) -> None:
    server.echo_secret_in_auth_error = True
    leaky = Credentials(server.client_id, server.client_secret, "token-that-would-leak", ACCOUNT)
    client = client_for(server, leaky)

    with pytest.raises(CTraderAuthError) as caught:
        await client.start()

    assert "CH_ACCESS_TOKEN_INVALID" in str(caught.value)
    assert "token-that-would-leak" not in str(caught.value)
    assert "***" in str(caught.value)


async def test_a_link_that_dies_during_the_handshake_fails_it_at_once_instead_of_timing_out(
    server,
) -> None:
    server.drop_on_next.append(APPLICATION_AUTH_REQ)
    started = asyncio.get_running_loop().time()

    async with client_for(server, request_timeout=5.0) as client:
        assert client.connected

    assert asyncio.get_running_loop().time() - started < 2.0
    assert server.connection_count == 2


async def test_a_transient_auth_error_is_retried_until_it_works(server) -> None:
    server.fail_payload(APPLICATION_AUTH_REQ, "CH_SERVER_NOT_REACHABLE")

    async with client_for(server) as client:
        assert client.connected
    assert server.connection_count == 2


# reconnection


async def test_after_a_drop_it_reconnects_reauthenticates_and_resubscribes_the_spots(
    server,
) -> None:
    server.set_quote("EURUSD", 1.1, 1.10008)
    spots: list[messages.ProtoOASpotEvent] = []

    def collect(message, client_msg_id) -> None:
        if isinstance(message, messages.ProtoOASpotEvent):
            spots.append(message)

    async with client_for(server) as client:
        client.add_listener(collect)
        await client.subscribe_spots([1])
        await eventually(lambda: len(spots) == 1)  # the first event carries the last price

        server.drop_connections()
        await eventually(lambda: server.connection_count == 2 and client.connected)
        await eventually(lambda: len(spots) == 2)  # the resubscription got the last price again
        await server.push_spot("EURUSD", 1.1002, 1.10028)
        await eventually(lambda: len(spots) == 3)

    assert len(server.requests_of(messages.ProtoOAApplicationAuthReq)) == 2
    assert len(server.requests_of(messages.ProtoOAAccountAuthReq)) == 2
    subscriptions = server.requests_of(messages.ProtoOASubscribeSpotsReq)
    assert [list(s.symbolId) for s in subscriptions] == [[1], [1]]


async def test_calls_fail_fast_while_there_is_no_session(server) -> None:
    async with client_for(server, backoff_initial=0.5, backoff_max=0.5) as client:
        server.drop_connections()
        await eventually(lambda: not client.connected)

        with pytest.raises(VenueUnavailable, match="no authenticated"):
            await client.request(trader_request(), messages.ProtoOATraderRes)
        with pytest.raises(VenueUnavailable):
            await client.submit(trader_request(), "m1")

        await eventually(lambda: client.connected)
        assert (await client.request(trader_request(), messages.ProtoOATraderRes)).trader


async def test_the_backoff_doubles_up_to_its_cap(server, caplog) -> None:
    for _ in range(5):
        server.fail_payload(APPLICATION_AUTH_REQ, "CH_SERVER_NOT_REACHABLE")
    caplog.set_level(logging.WARNING, logger="app.fx.runner.ctrader.client")

    async with client_for(server, backoff_initial=0.01, backoff_max=0.04):
        pass

    delays = [record.args[-1] for record in caplog.records]
    assert delays == [0.01, 0.02, 0.04, 0.04, 0.04]


async def test_a_client_disconnect_event_ends_the_session_and_a_new_one_is_authenticated(
    server,
) -> None:
    async with client_for(server) as client:
        await server.send_event(messages.ProtoOAClientDisconnectEvent(reason="blocked"))
        await eventually(lambda: server.connection_count == 2 and client.connected)


async def test_an_account_disconnect_event_matters_only_for_our_account(server) -> None:
    async with client_for(server) as client:
        await server.send_event(messages.ProtoOAAccountDisconnectEvent(ctidTraderAccountId=42))
        await asyncio.sleep(0.15)
        assert server.connection_count == 1

        await server.send_event(messages.ProtoOAAccountDisconnectEvent(ctidTraderAccountId=ACCOUNT))
        await eventually(lambda: server.connection_count == 2 and client.connected)


async def test_an_invalidated_token_is_fatal_and_there_is_no_reconnection_loop(server) -> None:
    async with client_for(server) as client:
        event = messages.ProtoOAAccountsTokenInvalidatedEvent(
            ctidTraderAccountIds=[ACCOUNT], reason="revoked"
        )
        await server.send_event(event)
        await eventually(lambda: client.fatal_error is not None)
        await asyncio.sleep(0.15)

        assert server.connection_count == 1 and not client.connected
        with pytest.raises(CTraderAuthError, match="invalidated"):
            await client.request(trader_request(), messages.ProtoOATraderRes)


# heartbeat, idle link and rate limits


async def test_the_client_sends_heartbeats_and_a_server_heartbeat_is_not_a_message(server) -> None:
    received = []
    async with client_for(server, heartbeat_interval=0.03) as client:
        client.add_listener(lambda message, client_msg_id: received.append(message))
        await server.send_heartbeat()
        await eventually(lambda: server.heartbeats >= 3)

    assert received == []


async def test_a_silent_link_is_dropped_but_heartbeats_keep_a_quiet_one_alive(server) -> None:
    async with client_for(server, idle_timeout=0.15) as client:
        await eventually(lambda: server.connection_count >= 2)  # nothing arrived: presumed dead

    async with FakeCTraderServer() as lively:
        async with client_for(lively, idle_timeout=0.15) as client:

            async def beat() -> None:
                while True:
                    await lively.send_heartbeat()
                    await asyncio.sleep(0.05)

            beating = asyncio.create_task(beat())
            await asyncio.sleep(0.5)
            beating.cancel()
            assert lively.connection_count == 1 and client.connected


async def test_the_limiter_never_lets_more_than_its_limit_into_any_window() -> None:
    limiter = RateLimiter(3, 0.2)
    loop = asyncio.get_running_loop()
    stamps = []

    for _ in range(9):
        await limiter.acquire()
        stamps.append(loop.time())

    assert all(stamps[i + 3] - stamps[i] >= 0.2 - 0.005 for i in range(len(stamps) - 3))


async def test_requests_leave_the_client_at_the_configured_rate(server) -> None:
    async with client_for(server, rate_limit=(3, 0.15)) as client:
        await asyncio.gather(
            *(client.request(trader_request(), messages.ProtoOATraderRes) for _ in range(7))
        )

    arrivals = [
        at
        for at, m in zip(server.received_at, server.received, strict=True)
        if isinstance(m, messages.ProtoOATraderReq)
    ]
    assert len(arrivals) == 7
    assert all(arrivals[i + 3] - arrivals[i] >= 0.15 - 0.03 for i in range(4))


async def test_retry_after_holds_that_request_type_and_only_that_one(server) -> None:
    server.fail_payload(TRADER_REQ, "BLOCKED_PAYLOAD_TYPE", retry_after=1)
    async with client_for(server) as client:
        with pytest.raises(CTraderRequestError) as caught:
            await client.request(trader_request(), messages.ProtoOATraderRes)
        assert caught.value.code == "BLOCKED_PAYLOAD_TYPE"

        listing = messages.ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT)
        started = asyncio.get_running_loop().time()
        await client.request(listing, messages.ProtoOASymbolsListRes)  # another type: not held
        assert asyncio.get_running_loop().time() - started < 0.5

        await client.request(trader_request(), messages.ProtoOATraderRes)  # held until unlocked

    traders = [
        at
        for at, m in zip(server.received_at, server.received, strict=True)
        if isinstance(m, messages.ProtoOATraderReq)
    ]
    assert traders[1] - traders[0] >= 0.9


async def test_a_block_longer_than_the_request_timeout_fails_without_sending(server) -> None:
    server.fail_payload(TRADER_REQ, "BLOCKED_PAYLOAD_TYPE", retry_after=30)
    async with client_for(server, request_timeout=0.5) as client:
        with pytest.raises(CTraderRequestError):
            await client.request(trader_request(), messages.ProtoOATraderRes)
        with pytest.raises(VenueUnavailable, match="blocked"):
            await client.request(trader_request(), messages.ProtoOATraderRes)

    assert len(server.requests_of(messages.ProtoOATraderReq)) == 1


# secrets


async def test_the_credentials_never_show_in_repr_str_or_logs(server, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    credentials = server.credentials
    secrets = (credentials.client_id, credentials.client_secret, credentials.access_token)
    client = client_for(server, credentials, request_timeout=0.3)

    async with client:
        server.drop_connections()  # a reconnection logs a warning
        await eventually(lambda: server.connection_count == 2 and client.connected)
        shown = [repr(credentials), str(credentials), f"{credentials}", repr(client), str(client)]
        with pytest.raises(VenueUnavailable) as caught:
            server.silent.add(TRADER_REQ)
            await client.request(trader_request(), messages.ProtoOATraderRes)
        shown.append(str(caught.value))

    assert caplog.records
    for text in [*shown, caplog.text]:
        assert not any(secret in text for secret in secrets)
    assert "account_id=1001" in repr(credentials)


def test_redact_scrubs_every_secret_from_server_text() -> None:
    credentials = Credentials("the-id", "the-secret", "the-token", 1)

    assert credentials.redact("bad the-token / the-secret / the-id") == "bad *** / *** / ***"
    assert Credentials("", "", "", 1).redact("nothing to hide") == "nothing to hide"

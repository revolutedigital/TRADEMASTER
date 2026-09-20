"""Transport of the cTrader Open API: framing, request correlation, heartbeat and reconnection.

The Open API speaks protobuf over TLS (`demo.ctraderapi.com:5035`, `live.ctraderapi.com:5035`).
Every message is a `ProtoMessage{payloadType, payload, clientMsgId}` serialized and preceded by
4 bytes, big-endian, with its length. `CTraderClient.run()` keeps one authenticated session alive:
it connects, does the application auth and the account auth, resubscribes the spots that were
asked for, and reads until the link drops, then reconnects with a capped exponential backoff.
While there is no session every call raises `VenueUnavailable`. A credential the server refuses
(or a token it invalidates) is fatal: `run()` raises `CTraderAuthError` and never loops on it.

Requests (`request`) are answered by a message carrying the same `clientMsgId`. Trading requests
(`submit`) are answered by events, so the client only sends them and hands every message that is
not the answer to a pending request to the listeners (see `CTraderVenue`). Sends are spaced to at
most 5 per second, and a `BLOCKED_PAYLOAD_TYPE` answer holds the sends until its `retryAfter`.

The credentials live in a `Credentials` object whose `repr` hides them; they are never logged, and
text that comes from the server is scrubbed of them before it reaches an exception or a log line.

Unverified assumptions (no credentials yet; the demo probe `scripts/research/ctrader_probe.py`
settles each one, item numbers are those of section 4.4 of docs/forex/ctrader-paths.md): the
length prefix is big-endian (the docs only say to swap bytes on little-endian hosts, so this is
inferred) and the first authentication of item 1 shows it; the answer to a request echoes its
`clientMsgId` (item 1; trade events are matched by content instead, see venue.py); the server
sends some traffic, spots or heartbeats, at least every `idle_timeout` seconds, otherwise a quiet
market would be read as a dead link (item 7, set `idle_timeout=None` if it does not); the auth
error codes arrive in `ProtoOAErrorRes.errorCode` as the enum names and the ones listed in
`FATAL_AUTH_CODES` are the permanent ones (the probe's extra bad-token check, and item 8); the
`retryAfter` rate-limit answer cannot be provoked without breaking the limit on purpose, so it is
implemented from the `.proto` alone and has no matrix item.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import ssl
import struct
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self, TypeVar

from google.protobuf.message import DecodeError, Message

from app.fx.runner.ctrader.proto import OpenApiCommonMessages_pb2 as common
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from app.fx.runner.venue import VenueUnavailable

log = logging.getLogger(__name__)

DEMO_HOST = "demo.ctraderapi.com"
LIVE_HOST = "live.ctraderapi.com"
PORT = 5035
LENGTH_PREFIX = struct.Struct(">I")
MAX_FRAME_BYTES = 32 * 1024 * 1024
PRICE_SCALE = 100_000  # spot and relative prices are integers in 1/100000

# Errors that no reconnection can fix: the operator has to change something.
FATAL_AUTH_CODES = frozenset(
    {
        "CH_CLIENT_AUTH_FAILURE",
        "CH_OA_CLIENT_NOT_FOUND",
        "CH_ACCESS_TOKEN_INVALID",
        "OA_AUTH_TOKEN_EXPIRED",
        "CH_CTID_TRADER_ACCOUNT_NOT_FOUND",
        "RET_NO_SUCH_LOGIN",
        "RET_ACCOUNT_DISABLED",
    }
)
BLOCKED_PAYLOAD_TYPE = "BLOCKED_PAYLOAD_TYPE"

ResponseT = TypeVar("ResponseT", bound=Message)
Listener = Callable[[Message, str | None], None]


class CTraderAuthError(Exception):
    """The application or the account credentials were refused for good; fix them, do not retry."""


class CTraderRequestError(VenueUnavailable):
    """The server answered a request with an error response."""

    def __init__(self, code: str, description: str = "") -> None:
        super().__init__(f"{code}: {description}" if description else code)
        self.code = code


class SessionLost(VenueUnavailable):
    """The server ended the session; a new one has to be authenticated."""


class FrameError(Exception):
    """The bytes on the wire are not a valid frame."""


@dataclass(frozen=True, repr=False)
class Credentials:
    client_id: str
    client_secret: str
    access_token: str
    account_id: int

    def __repr__(self) -> str:
        return f"Credentials(account_id={self.account_id}, secrets=<hidden>)"

    def redact(self, text: str) -> str:
        for secret in (self.client_id, self.client_secret, self.access_token):
            if secret:
                text = text.replace(secret, "***")
        return text


@dataclass(frozen=True)
class ServerError:
    code: str
    description: str
    retry_after: int | None = None


def server_error(message: Message) -> ServerError | None:
    """The error a message carries, if it is one of the messages the server uses for errors."""
    if isinstance(message, messages.ProtoOAErrorRes):
        retry_after = message.retryAfter if message.HasField("retryAfter") else None
        return ServerError(message.errorCode, message.description, retry_after)
    if isinstance(message, (messages.ProtoOAOrderErrorEvent, common.ProtoErrorRes)):
        return ServerError(message.errorCode, message.description)
    return None


def _payload_registry() -> dict[int, type[Message]]:
    """Payload type -> message class, read from the default `payloadType` of each message."""
    registry: dict[int, type[Message]] = {}
    for module in (common, messages):
        for name, descriptor in module.DESCRIPTOR.message_types_by_name.items():
            field = descriptor.fields_by_name.get("payloadType")
            if field is not None and field.has_default_value:
                registry[field.default_value] = getattr(module, name)
    return registry


PAYLOADS = _payload_registry()
HEARTBEAT_EVENT = common.ProtoHeartbeatEvent.DESCRIPTOR.fields_by_name["payloadType"].default_value


def payload_type_of(message: Message) -> int:
    return message.DESCRIPTOR.fields_by_name["payloadType"].default_value


def encode_frame(payload_type: int, message: Message | None, client_msg_id: str | None) -> bytes:
    envelope = common.ProtoMessage(payloadType=payload_type)
    if message is not None:
        envelope.payload = message.SerializeToString()
    if client_msg_id is not None:
        envelope.clientMsgId = client_msg_id
    body = envelope.SerializeToString()
    return LENGTH_PREFIX.pack(len(body)) + body


async def read_frame(reader: asyncio.StreamReader) -> common.ProtoMessage:
    """Read one frame; a link that closes in the middle raises `IncompleteReadError`."""
    (length,) = LENGTH_PREFIX.unpack(await reader.readexactly(LENGTH_PREFIX.size))
    if length > MAX_FRAME_BYTES:
        raise FrameError(f"a frame of {length} bytes is over the {MAX_FRAME_BYTES} byte limit")
    envelope = common.ProtoMessage()
    try:
        envelope.ParseFromString(await reader.readexactly(length))
    except DecodeError as error:
        raise FrameError("a frame is not a valid ProtoMessage") from error
    return envelope


class RateLimiter:
    """At most `limit` events in any `window` seconds."""

    def __init__(self, limit: int, window: float) -> None:
        self._limit, self._window = limit, window
        self._sent: deque[float] = deque()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            while self._sent and now - self._sent[0] >= self._window:
                self._sent.popleft()
            if len(self._sent) < self._limit:
                self._sent.append(now)
                return
            await asyncio.sleep(self._window - (now - self._sent[0]))


class CTraderClient:
    def __init__(
        self,
        credentials: Credentials,
        *,
        host: str = DEMO_HOST,
        port: int = PORT,
        use_tls: bool = True,
        request_timeout: float = 10.0,
        connect_timeout: float = 10.0,
        heartbeat_interval: float = 10.0,
        idle_timeout: float | None = 60.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        rate_limit: tuple[int, float] = (5, 1.0),
    ) -> None:
        self._credentials = credentials
        self._host, self._port, self._use_tls = host, port, use_tls
        self._request_timeout, self._connect_timeout = request_timeout, connect_timeout
        self._heartbeat_interval, self._idle_timeout = heartbeat_interval, idle_timeout
        self._backoff_initial, self._backoff_max = backoff_initial, backoff_max
        self._limiter = RateLimiter(*rate_limit)
        self._ids = itertools.count(1)
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, tuple[int, asyncio.Future[Message]]] = {}
        self._listeners: list[Listener] = []
        self._spot_ids: list[int] = []
        self._blocked_until: dict[int, float] = {}  # payload type (0 = every type) -> loop time
        self._ready = asyncio.Event()
        self._fatal: CTraderAuthError | None = None
        self._task: asyncio.Task[None] | None = None
        self.sessions = 0  # how many sessions have been authenticated so far

    def __repr__(self) -> str:
        return (
            f"CTraderClient(host={self._host!r}, port={self._port}, "
            f"account_id={self.account_id}, connected={self.connected})"
        )

    @property
    def account_id(self) -> int:
        return self._credentials.account_id

    @property
    def connected(self) -> bool:
        return self._ready.is_set()

    @property
    def fatal_error(self) -> CTraderAuthError | None:
        return self._fatal

    def redact(self, text: str) -> str:
        return self._credentials.redact(text)

    def next_id(self) -> str:
        return f"m{next(self._ids)}"

    def add_listener(self, listener: Listener) -> None:
        """Receive every message that is not the answer to a pending request (spots, events)."""
        self._listeners.append(listener)

    async def request(self, message: Message, expect: type[ResponseT]) -> ResponseT:
        """Send a request and return its answer; an error answer raises `CTraderRequestError`."""
        self._require_session()
        response = await self._call(message)
        if not isinstance(response, expect):
            raise VenueUnavailable(
                f"{type(message).__name__} was answered with {type(response).__name__}"
            )
        return response

    async def submit(self, message: Message, client_msg_id: str) -> None:
        """Send a request whose outcome arrives as events (an order); do not wait for anything."""
        self._require_session()
        await self._send(message, client_msg_id)

    async def subscribe_spots(self, symbol_ids: Sequence[int]) -> None:
        """Subscribe now, and again after every reconnection."""
        self._spot_ids = sorted({*self._spot_ids, *symbol_ids})
        await self.request(self._subscribe_message(symbol_ids), messages.ProtoOASubscribeSpotsRes)

    def _subscribe_message(self, symbol_ids: Sequence[int]) -> messages.ProtoOASubscribeSpotsReq:
        return messages.ProtoOASubscribeSpotsReq(
            ctidTraderAccountId=self.account_id,
            symbolId=list(symbol_ids),
            subscribeToSpotTimestamp=True,
        )

    def _require_session(self) -> None:
        if self._fatal is not None:
            raise self._fatal
        if not self._ready.is_set():
            raise VenueUnavailable("there is no authenticated cTrader session right now")

    async def _call(self, message: Message) -> Message:
        payload_type = payload_type_of(message)
        client_msg_id = self.next_id()
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending[client_msg_id] = (payload_type, future)
        try:
            await self._send(message, client_msg_id)
            return await asyncio.wait_for(future, self._request_timeout)
        except TimeoutError as error:
            raise VenueUnavailable(
                f"cTrader did not answer {type(message).__name__} in {self._request_timeout}s"
            ) from error
        finally:
            del self._pending[client_msg_id]
            if future.done() and not future.cancelled():
                future.exception()  # mark it retrieved when the send failed before the wait

    async def _send(
        self, message: Message, client_msg_id: str | None = None, *, throttle: bool = True
    ) -> None:
        payload_type = payload_type_of(message)
        if throttle:
            await self._wait_for_turn(payload_type)
        writer = self._writer
        if writer is None:
            raise VenueUnavailable("the cTrader connection is down")
        writer.write(encode_frame(payload_type, message, client_msg_id))
        try:
            await writer.drain()
        except OSError as error:
            raise VenueUnavailable("the cTrader connection dropped while sending") from error

    async def _wait_for_turn(self, payload_type: int) -> None:
        loop = asyncio.get_running_loop()
        until = max(self._blocked_until.get(payload_type, 0.0), self._blocked_until.get(0, 0.0))
        held = until - loop.time()
        if held > self._request_timeout:
            raise VenueUnavailable(f"cTrader blocked this request type for {held:.0f} more seconds")
        if held > 0:
            await asyncio.sleep(held)
        await self._limiter.acquire()

    async def start(self, timeout: float = 30.0) -> None:
        """Run the session in the background and return once it is authenticated."""
        self._task = asyncio.create_task(self.run())
        ready = asyncio.create_task(self._ready.wait())
        try:
            done, _ = await asyncio.wait(
                {self._task, ready}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            ready.cancel()
        if self._task in done:
            self._task.result()  # raises the fatal error that ended it
        if not done:
            await self.stop()
            raise VenueUnavailable(f"cTrader did not authenticate within {timeout}s")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    def drop_connection(self) -> None:
        """Cut the link as a network failure would; `run()` reconnects."""
        if self._writer is not None:
            self._writer.transport.abort()

    async def run(self) -> None:
        """Keep an authenticated session up until cancelled; only a fatal auth error ends it."""
        loop = asyncio.get_running_loop()
        delay = self._backoff_initial
        while True:
            started = loop.time()
            try:
                await self._session()
            except CTraderAuthError as error:
                self._fatal = error
                raise
            except (OSError, TimeoutError, EOFError, FrameError, VenueUnavailable) as error:
                if loop.time() - started > self._backoff_max:
                    delay = self._backoff_initial  # a long healthy session earns a fresh start
                log.warning(
                    "cTrader session lost (%s); reconnecting in %.1fs",
                    self.redact(f"{type(error).__name__}: {error}"),
                    delay,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._backoff_max)

    async def _session(self) -> None:
        context = ssl.create_default_context() if self._use_tls else None
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port, ssl=context), self._connect_timeout
        )
        self._writer = writer
        workers = [
            asyncio.create_task(self._read_loop(reader)),
            asyncio.create_task(self._heartbeat_loop()),
        ]
        try:
            await self._authenticate()
            self.sessions += 1
            self._ready.set()
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
            for worker in done:
                worker.result()
            raise VenueUnavailable("the cTrader session ended")
        finally:
            self._ready.clear()
            self._writer = None
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            self._fail_pending(VenueUnavailable("the cTrader connection is down"))

    async def _authenticate(self) -> None:
        credentials = self._credentials
        await self._auth_call(
            messages.ProtoOAApplicationAuthReq(
                clientId=credentials.client_id, clientSecret=credentials.client_secret
            )
        )
        await self._auth_call(
            messages.ProtoOAAccountAuthReq(
                ctidTraderAccountId=credentials.account_id, accessToken=credentials.access_token
            )
        )
        if self._spot_ids:
            await self._auth_call(self._subscribe_message(self._spot_ids))

    async def _auth_call(self, message: Message) -> None:
        try:
            await self._call(message)
        except CTraderRequestError as error:
            if error.code in FATAL_AUTH_CODES:
                raise CTraderAuthError(f"cTrader refused the credentials: {error}") from error
            raise

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            await self._send(common.ProtoHeartbeatEvent(), throttle=False)

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(read_frame(reader), self._idle_timeout)
                except TimeoutError as error:
                    raise VenueUnavailable(f"nothing received for {self._idle_timeout}s") from error
                self._dispatch(frame)
        except BaseException as error:
            fatal = isinstance(error, CTraderAuthError)
            self._fail_pending(error if fatal else VenueUnavailable("the cTrader link dropped"))
            raise

    def _fail_pending(self, error: Exception) -> None:
        for _, future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    def _dispatch(self, frame: common.ProtoMessage) -> None:
        payload_type = frame.payloadType
        message_class = PAYLOADS.get(payload_type)
        if payload_type == HEARTBEAT_EVENT or message_class is None:
            return  # traffic proves the link is alive; an unknown message is not ours to read
        message = message_class()
        try:
            message.ParseFromString(frame.payload)
        except DecodeError as error:
            raise FrameError(f"payload type {payload_type} is not a valid message") from error
        client_msg_id = frame.clientMsgId if frame.HasField("clientMsgId") else None
        self._check_session_events(message)
        error_found = server_error(message)
        pending = self._pending.get(client_msg_id) if client_msg_id else None
        if error_found is not None and error_found.code == BLOCKED_PAYLOAD_TYPE:
            self._hold_sends(pending[0] if pending else 0, error_found.retry_after or 0)
        if pending is not None:
            self._answer(pending[1], message, error_found)
            return
        for listener in self._listeners:
            try:
                listener(message, client_msg_id)
            except Exception:
                log.exception("a cTrader listener failed on %s", type(message).__name__)

    def _check_session_events(self, message: Message) -> None:
        if isinstance(message, messages.ProtoOAClientDisconnectEvent):
            raise SessionLost("the server ended the application session")
        if isinstance(message, messages.ProtoOAAccountDisconnectEvent):
            if message.ctidTraderAccountId == self.account_id:
                raise SessionLost("the server dropped the account session")
        if isinstance(message, messages.ProtoOAAccountsTokenInvalidatedEvent):
            ids = message.ctidTraderAccountIds
            if not ids or self.account_id in ids:
                raise CTraderAuthError("the access token was invalidated (expired or revoked)")

    def _hold_sends(self, payload_type: int, seconds: float) -> None:
        self._blocked_until[payload_type] = asyncio.get_running_loop().time() + seconds

    def _answer(
        self, future: asyncio.Future[Message], message: Message, error: ServerError | None
    ) -> None:
        if future.done():
            return
        if error is None:
            future.set_result(message)
        else:
            future.set_exception(
                CTraderRequestError(error.code, self.redact(error.description)),
            )

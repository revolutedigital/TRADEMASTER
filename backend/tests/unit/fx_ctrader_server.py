"""A cTrader Open API server for tests: real protobuf frames in clear text on 127.0.0.1.

The client under test runs its production code path against it; only TLS is switched off. It
implements what the `.proto` says (the application and account auth, symbols, trader, spot
subscription, market orders that refuse an absolute stop, reconcile, amend, close) and, where the
`.proto` is silent, an invented behavior that is marked as such; the real broker is checked by
`scripts/research/ctrader_probe.py`, not by these tests.

Faults are attributes and methods a test flips: `silent`, `delays`, `fail_next`,
`reject_next_order`, `drop_connections`, `drop_after_order`, `emit_events`, and so on.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

from google.protobuf.message import Message

from app.fx.runner.ctrader.client import (
    HEARTBEAT_EVENT,
    LENGTH_PREFIX,
    PAYLOADS,
    PRICE_SCALE,
    Credentials,
    CTraderClient,
    encode_frame,
    payload_type_of,
    read_frame,
)
from app.fx.runner.ctrader.proto import OpenApiCommonMessages_pb2 as common
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from app.fx.runner.ctrader.proto import OpenApiModelMessages_pb2 as model

# name -> (symbol id, digits, pip position)
SYMBOLS = {
    "EURUSD": (1, 5, 4),
    "GBPUSD": (2, 5, 4),
    "USDJPY": (3, 3, 2),
    "EURJPY": (4, 3, 2),
}
MIN_VOLUME = 100_000  # 1000 units = 0.01 lot, in protocol cents
STEP_VOLUME = 100_000


async def eventually(check: Callable[[], bool], timeout: float = 3.0) -> None:
    """Wait until `check()` is true; the assertion failure names what never happened."""
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not become true in time")
        await asyncio.sleep(0.01)


def client_for(
    server: FakeCTraderServer, credentials: Credentials | None = None, **overrides: object
) -> CTraderClient:
    """A client wired to `server` with timers short enough for tests."""
    settings: dict[str, object] = {
        "host": "127.0.0.1",
        "port": server.port,
        "use_tls": False,
        "request_timeout": 1.0,
        "heartbeat_interval": 5.0,
        "idle_timeout": None,
        "backoff_initial": 0.02,
        "backoff_max": 0.1,
        "rate_limit": (1000, 1.0),
    }
    settings.update(overrides)
    return CTraderClient(credentials or server.credentials, **settings)


@dataclass
class ServerPosition:
    position_id: int
    symbol_id: int
    side: int  # model.BUY or model.SELL
    volume: int
    price: float
    stop: float | None
    target: float | None
    label: str
    client_order_id: str


@dataclass
class _Connection:
    writer: asyncio.StreamWriter
    application_ok: bool = False
    account_ok: bool = False
    spots: set[int] = field(default_factory=set)


class FakeCTraderServer:
    def __init__(
        self,
        *,
        account_id: int = 1001,
        client_id: str = "test-client-id",
        client_secret: str = "test-client-secret",
        access_token: str = "test-access-token",
        balance: float = 500.0,
        money_digits: int = 2,
        currency: str = "USD",
        account_type: int = model.HEDGED,
        slash_names: bool = False,
    ) -> None:
        self.account_id, self.client_id = account_id, client_id
        self.client_secret, self.access_token = client_secret, access_token
        self.balance, self.money_digits, self.currency = balance, money_digits, currency
        self.account_type, self.slash_names = account_type, slash_names
        self.pip_positions = {name: pip for name, (_, _, pip) in SYMBOLS.items()}
        self.quotes: dict[int, tuple[float, float]] = {}
        self.positions: dict[int, ServerPosition] = {}
        self.connections: list[_Connection] = []
        self.connection_count = 0
        # what the server saw
        self.received: list[Message] = []
        self.received_at: list[float] = []
        self.orders_received: list[messages.ProtoOANewOrderReq] = []
        self.heartbeats = 0
        # faults and switches
        self.silent: set[int] = set()  # payload types that are never answered
        self.drop_on_next: list[
            int
        ] = []  # the link dies when the next request of this type arrives
        self.delays: dict[int, float] = {}  # payload type -> seconds before it is handled
        self.fail_next: dict[int, list[tuple[str, int | None]]] = {}
        self.order_rejections: list[tuple[str, str]] = []  # (code, style)
        self.amend_errors: list[str] = []
        self.close_errors: list[str] = []
        self.leak_token_in_rejections = False  # a description that echoes the token
        self.reject_application_auth = False
        self.echo_secret_in_auth_error = False
        self.emit_events = True
        self.echo_ids_on_events = True
        self.drop_after_order = False  # the order fills, then the link dies before any event
        self.drop_on_close = False  # the position closes, then the link dies before any event
        self.emit_close_event = True
        self.ignore_relative_sltp = False  # a broker that drops the relative stop and target
        self.ignore_amend = False  # an amendment that is accepted and not applied
        self.strip_labels = False  # a reconcile that forgets the label
        self.fill_slippage = 0.0  # added to a buy fill, taken from a sell fill
        self.close_commission = -0.08
        self.chunk_size: int | None = None
        self._ids = 900_000
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    # lifecycle

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    @property
    def credentials(self) -> Credentials:
        return Credentials(self.client_id, self.client_secret, self.access_token, self.account_id)

    async def stop(self) -> None:
        self.drop_connections()
        for task in list(self._tasks):
            task.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

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

    # what tests do to the market and the link

    def symbol_name(self, symbol_id: int) -> str:
        name = next(n for n, (i, _, _) in SYMBOLS.items() if i == symbol_id)
        return f"{name[:3]}/{name[3:]}" if self.slash_names else name

    def set_quote(self, symbol: str, bid: float, ask: float) -> None:
        self.quotes[SYMBOLS[symbol][0]] = (bid, ask)

    async def push_spot(
        self,
        symbol: str,
        bid: float | None = None,
        ask: float | None = None,
        *,
        at_ms: int | None = None,
    ) -> None:
        """Move the market and tell whoever is subscribed; a missing side is not sent."""
        symbol_id = SYMBOLS[symbol][0]
        old_bid, old_ask = self.quotes.get(symbol_id, (bid, ask))
        self.quotes[symbol_id] = (old_bid if bid is None else bid, old_ask if ask is None else ask)
        event = messages.ProtoOASpotEvent(
            ctidTraderAccountId=self.account_id,
            symbolId=symbol_id,
            timestamp=at_ms if at_ms is not None else int(time.time() * 1000),
        )
        if bid is not None:
            event.bid = round(bid * PRICE_SCALE)
        if ask is not None:
            event.ask = round(ask * PRICE_SCALE)
        for conn in self.connections:
            if symbol_id in conn.spots:
                await self._send(conn, event)

    def drop_connections(self) -> None:
        for conn in list(self.connections):
            conn.writer.transport.abort()

    async def send_event(self, event: Message) -> None:
        for conn in list(self.connections):
            if conn.account_ok:
                await self._send(conn, event)

    async def send_frame(self, payload_type: int, payload: bytes) -> None:
        """Write a frame with these exact payload bytes, valid or not."""
        body = common.ProtoMessage(payloadType=payload_type, payload=payload).SerializeToString()
        for conn in list(self.connections):
            conn.writer.write(LENGTH_PREFIX.pack(len(body)) + body)
            await conn.writer.drain()

    async def send_heartbeat(self) -> None:
        for conn in list(self.connections):
            await self._send(conn, common.ProtoHeartbeatEvent())

    def fail_payload(self, payload_type: int, code: str, retry_after: int | None = None) -> None:
        self.fail_next.setdefault(payload_type, []).append((code, retry_after))

    def reject_next_order(self, code: str, style: str = "order_error") -> None:
        """style: `order_error` (ProtoOAOrderErrorEvent), `execution` or `error_res`."""
        self.order_rejections.append((code, style))

    def add_position(
        self, symbol: str, side: int, units: int, price: float, *, label: str = "manual"
    ) -> ServerPosition:
        """A position that exists without an order, for valuation and reconcile tests."""
        self._ids += 1
        position = ServerPosition(
            self._ids, SYMBOLS[symbol][0], side, units * 100, price, None, None, label, label
        )
        self.positions[position.position_id] = position
        return position

    def requests_of(self, message_class: type[Message]) -> list:
        return [m for m in self.received if isinstance(m, message_class)]

    # connection handling

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = _Connection(writer)
        self.connections.append(conn)
        self.connection_count += 1
        try:
            while True:
                frame = await read_frame(reader)
                task = asyncio.create_task(self._handle(conn, frame))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.connections.remove(conn)
            writer.close()

    async def _send(
        self, conn: _Connection, message: Message, client_msg_id: str | None = None
    ) -> None:
        data = encode_frame(payload_type_of(message), message, client_msg_id)
        try:
            if self.chunk_size:
                for start in range(0, len(data), self.chunk_size):
                    conn.writer.write(data[start : start + self.chunk_size])
                    await conn.writer.drain()
                    await asyncio.sleep(0)
            else:
                conn.writer.write(data)
                await conn.writer.drain()
        except ConnectionError:
            pass

    async def _event(self, conn: _Connection, message: Message, msg_id: str | None) -> None:
        await self._send(conn, message, msg_id if self.echo_ids_on_events else None)

    async def _error(
        self,
        conn: _Connection,
        msg_id: str | None,
        code: str,
        description: str = "",
        retry_after: int | None = None,
    ) -> None:
        error = messages.ProtoOAErrorRes(
            ctidTraderAccountId=self.account_id, errorCode=code, description=description
        )
        if retry_after is not None:
            error.retryAfter = retry_after
        await self._send(conn, error, msg_id)

    async def _handle(self, conn: _Connection, frame: common.ProtoMessage) -> None:
        payload_type = frame.payloadType
        if payload_type == HEARTBEAT_EVENT:
            self.heartbeats += 1
            return
        request = PAYLOADS[payload_type]()
        request.ParseFromString(frame.payload)
        msg_id = frame.clientMsgId if frame.HasField("clientMsgId") else None
        self.received.append(request)
        self.received_at.append(time.monotonic())
        if payload_type in self.silent:
            return
        if payload_type in self.drop_on_next:
            self.drop_on_next.remove(payload_type)
            conn.writer.transport.abort()
            return
        if payload_type in self.delays:
            await asyncio.sleep(self.delays[payload_type])
        queued = self.fail_next.get(payload_type)
        if queued:
            code, retry_after = queued.pop(0)
            await self._error(conn, msg_id, code, "injected by the test", retry_after)
            return
        await self._dispatch(conn, request, msg_id)

    async def _dispatch(self, conn: _Connection, request: Message, msg_id: str | None) -> None:
        if isinstance(request, messages.ProtoOAApplicationAuthReq):
            await self._application_auth(conn, request, msg_id)
        elif isinstance(request, messages.ProtoOAAccountAuthReq):
            await self._account_auth(conn, request, msg_id)
        elif not conn.account_ok:
            await self._error(conn, msg_id, "ACCOUNT_NOT_AUTHORIZED")
        else:
            handlers: dict[type[Message], Callable[..., Awaitable[None]]] = {
                messages.ProtoOASymbolsListReq: self._symbols_list,
                messages.ProtoOASymbolByIdReq: self._symbol_by_id,
                messages.ProtoOAAssetListReq: self._asset_list,
                messages.ProtoOATraderReq: self._trader,
                messages.ProtoOASubscribeSpotsReq: self._subscribe,
                messages.ProtoOAReconcileReq: self._reconcile,
                messages.ProtoOANewOrderReq: self._new_order,
                messages.ProtoOAAmendPositionSLTPReq: self._amend,
                messages.ProtoOAClosePositionReq: self._close,
            }
            handler = handlers.get(type(request))
            if handler is None:
                await self._error(conn, msg_id, "UNSUPPORTED_MESSAGE")
            else:
                await handler(conn, request, msg_id)

    # auth and reference data

    async def _application_auth(self, conn, request, msg_id) -> None:
        if (
            self.reject_application_auth
            or request.clientId != self.client_id
            or request.clientSecret != self.client_secret
        ):
            await self._error(
                conn,
                msg_id,
                "CH_CLIENT_AUTH_FAILURE",
                "Open API client is not activated or wrong credentials",
            )
            return
        conn.application_ok = True
        await self._send(conn, messages.ProtoOAApplicationAuthRes(), msg_id)

    async def _account_auth(self, conn, request, msg_id) -> None:
        if not conn.application_ok:
            await self._error(conn, msg_id, "CH_CLIENT_NOT_AUTHENTICATED")
        elif (
            request.accessToken != self.access_token
            or request.ctidTraderAccountId != self.account_id
        ):
            note = (
                f"token {request.accessToken} is not valid"
                if self.echo_secret_in_auth_error
                else "invalid"
            )
            await self._error(conn, msg_id, "CH_ACCESS_TOKEN_INVALID", note)
        else:
            conn.account_ok = True
            response = messages.ProtoOAAccountAuthRes(ctidTraderAccountId=self.account_id)
            await self._send(conn, response, msg_id)

    async def _symbols_list(self, conn, request, msg_id) -> None:
        response = messages.ProtoOASymbolsListRes(ctidTraderAccountId=self.account_id)
        for _, (symbol_id, _, _) in SYMBOLS.items():
            response.symbol.add(
                symbolId=symbol_id, symbolName=self.symbol_name(symbol_id), enabled=True
            )
        await self._send(conn, response, msg_id)

    async def _symbol_by_id(self, conn, request, msg_id) -> None:
        response = messages.ProtoOASymbolByIdRes(ctidTraderAccountId=self.account_id)
        for name, (symbol_id, digits, _) in SYMBOLS.items():
            if symbol_id in request.symbolId:
                response.symbol.add(
                    symbolId=symbol_id,
                    digits=digits,
                    pipPosition=self.pip_positions[name],
                    minVolume=MIN_VOLUME,
                    stepVolume=STEP_VOLUME,
                    maxVolume=10_000_000_000,
                    lotSize=10_000_000,
                )
        await self._send(conn, response, msg_id)

    async def _asset_list(self, conn, request, msg_id) -> None:
        response = messages.ProtoOAAssetListRes(ctidTraderAccountId=self.account_id)
        response.asset.add(assetId=1, name=self.currency)
        await self._send(conn, response, msg_id)

    async def _trader(self, conn, request, msg_id) -> None:
        trader = model.ProtoOATrader(
            ctidTraderAccountId=self.account_id,
            balance=round(self.balance * 10**self.money_digits),
            depositAssetId=1,
            moneyDigits=self.money_digits,
            accountType=self.account_type,
        )
        response = messages.ProtoOATraderRes(ctidTraderAccountId=self.account_id, trader=trader)
        await self._send(conn, response, msg_id)

    async def _subscribe(self, conn, request, msg_id) -> None:
        conn.spots.update(request.symbolId)
        await self._send(
            conn, messages.ProtoOASubscribeSpotsRes(ctidTraderAccountId=self.account_id), msg_id
        )
        for symbol_id in request.symbolId:  # the first event carries the last price
            if symbol_id in self.quotes:
                bid, ask = self.quotes[symbol_id]
                first = messages.ProtoOASpotEvent(
                    ctidTraderAccountId=self.account_id,
                    symbolId=symbol_id,
                    bid=round(bid * PRICE_SCALE),
                    ask=round(ask * PRICE_SCALE),
                    timestamp=int(time.time() * 1000),
                )
                await self._send(conn, first)

    # trading

    def _position_proto(self, position: ServerPosition) -> model.ProtoOAPosition:
        proto = model.ProtoOAPosition(
            positionId=position.position_id,
            tradeData=self._trade_data(position),
            positionStatus=model.POSITION_STATUS_OPEN,
            swap=0,
            price=position.price,
            moneyDigits=self.money_digits,
        )
        if self.strip_labels:
            proto.tradeData.ClearField("label")
        if position.stop is not None:
            proto.stopLoss = position.stop
        if position.target is not None:
            proto.takeProfit = position.target
        return proto

    def _trade_data(self, position: ServerPosition) -> model.ProtoOATradeData:
        return model.ProtoOATradeData(
            symbolId=position.symbol_id,
            volume=position.volume,
            tradeSide=position.side,
            label=position.label,
        )

    def _order_proto(
        self, position: ServerPosition, order_type: int, status: int
    ) -> model.ProtoOAOrder:
        return model.ProtoOAOrder(
            orderId=position.position_id + 5_000,
            tradeData=self._trade_data(position),
            orderType=order_type,
            orderStatus=status,
            clientOrderId=position.client_order_id,
            executionPrice=position.price,
            positionId=position.position_id,
        )

    async def _reconcile(self, conn, request, msg_id) -> None:
        response = messages.ProtoOAReconcileRes(ctidTraderAccountId=self.account_id)
        response.position.extend(self._position_proto(p) for p in self.positions.values())
        await self._send(conn, response, msg_id)

    async def _reject(self, conn, request, msg_id, code: str, style: str) -> None:
        note = "rejected by the test server"
        if self.leak_token_in_rejections:
            note += f" for token {self.access_token}"
        if style == "error_res":
            await self._error(conn, msg_id, code, note)
        elif style == "execution":
            ghost = ServerPosition(
                0,
                request.symbolId,
                request.tradeSide,
                request.volume,
                0.0,
                None,
                None,
                request.label,
                request.clientOrderId,
            )
            event = messages.ProtoOAExecutionEvent(
                ctidTraderAccountId=self.account_id,
                executionType=model.ORDER_REJECTED,
                order=self._order_proto(ghost, request.orderType, model.ORDER_STATUS_REJECTED),
                errorCode=code,
            )
            await self._event(conn, event, msg_id)
        else:
            event = messages.ProtoOAOrderErrorEvent(
                ctidTraderAccountId=self.account_id, errorCode=code, description=note
            )
            await self._event(conn, event, msg_id)

    async def _new_order(self, conn, request: messages.ProtoOANewOrderReq, msg_id) -> None:
        self.orders_received.append(request)
        if self.order_rejections:
            code, style = self.order_rejections.pop(0)
            await self._reject(conn, request, msg_id, code, style)
            return
        market = request.orderType == model.MARKET
        if market and (request.HasField("stopLoss") or request.HasField("takeProfit")):
            # the .proto: stopLoss and takeProfit are "Not supported for MARKET orders"
            await self._reject(conn, request, msg_id, "TRADING_BAD_STOPS", "order_error")
            return
        bad_volume = request.volume < MIN_VOLUME or request.volume % STEP_VOLUME != 0
        if bad_volume:
            await self._reject(conn, request, msg_id, "TRADING_BAD_VOLUME", "order_error")
            return
        if request.symbolId not in self.quotes:
            await self._reject(conn, request, msg_id, "NO_QUOTES", "order_error")
            return
        position = self._fill(request)
        if self.drop_after_order:
            conn.writer.transport.abort()
            return
        if self.emit_events:
            accepted = messages.ProtoOAExecutionEvent(
                ctidTraderAccountId=self.account_id,
                executionType=model.ORDER_ACCEPTED,
                order=self._order_proto(position, request.orderType, model.ORDER_STATUS_ACCEPTED),
            )
            await self._event(conn, accepted, msg_id)
            filled = messages.ProtoOAExecutionEvent(
                ctidTraderAccountId=self.account_id,
                executionType=model.ORDER_FILLED,
                order=self._order_proto(position, request.orderType, model.ORDER_STATUS_FILLED),
                position=self._position_proto(position),
            )
            await self._event(conn, filled, msg_id)

    def _fill(self, request: messages.ProtoOANewOrderReq) -> ServerPosition:
        digits = next(d for _, (i, d, _) in SYMBOLS.items() if i == request.symbolId)
        bid, ask = self.quotes[request.symbolId]
        buy = request.tradeSide == model.BUY
        price = round(ask + self.fill_slippage if buy else bid - self.fill_slippage, digits)
        direction = 1 if buy else -1
        stop = target = None
        if request.HasField("relativeStopLoss") and not self.ignore_relative_sltp:
            stop = round(price - direction * request.relativeStopLoss / PRICE_SCALE, digits)
        if request.HasField("relativeTakeProfit") and not self.ignore_relative_sltp:
            target = round(price + direction * request.relativeTakeProfit / PRICE_SCALE, digits)
        if request.orderType == model.MARKET_RANGE:  # invented: absolute levels are honored here
            stop = request.stopLoss if request.HasField("stopLoss") else stop
            target = request.takeProfit if request.HasField("takeProfit") else target
        self._ids += 1
        position = ServerPosition(
            self._ids,
            request.symbolId,
            request.tradeSide,
            request.volume,
            price,
            stop,
            target,
            request.label,
            request.clientOrderId,
        )
        self.positions[position.position_id] = position
        return position

    async def _amend(self, conn, request: messages.ProtoOAAmendPositionSLTPReq, msg_id) -> None:
        position = self.positions.get(request.positionId)
        code = None
        if position is None:
            code = "POSITION_NOT_FOUND"
        elif self.amend_errors:
            code = self.amend_errors.pop(0)
        elif self._too_close(position, request):
            code = "PROTECTION_IS_TOO_CLOSE_TO_MARKET"
        if code is not None:
            error = messages.ProtoOAOrderErrorEvent(
                ctidTraderAccountId=self.account_id, errorCode=code, positionId=request.positionId
            )
            await self._event(conn, error, msg_id)
            return
        if not self.ignore_amend:
            position.stop = request.stopLoss if request.HasField("stopLoss") else None
            position.target = request.takeProfit if request.HasField("takeProfit") else None
        replaced = messages.ProtoOAExecutionEvent(
            ctidTraderAccountId=self.account_id,
            executionType=model.ORDER_REPLACED,
            position=self._position_proto(position),
        )
        await self._event(conn, replaced, msg_id)

    def _too_close(self, position: ServerPosition, request) -> bool:
        bid, ask = self.quotes[position.symbol_id]
        if not request.HasField("stopLoss"):
            return False
        return request.stopLoss >= bid if position.side == model.BUY else request.stopLoss <= ask

    async def _close(self, conn, request: messages.ProtoOAClosePositionReq, msg_id) -> None:
        position = self.positions.get(request.positionId)
        if position is None or self.close_errors:
            code = "POSITION_NOT_FOUND" if position is None else self.close_errors.pop(0)
            error = messages.ProtoOAOrderErrorEvent(
                ctidTraderAccountId=self.account_id, errorCode=code, positionId=request.positionId
            )
            await self._event(conn, error, msg_id)
            return
        bid, ask = self.quotes[position.symbol_id]
        buy = position.side == model.BUY
        exit_price = bid if buy else ask
        units = position.volume / 100
        gross = ((exit_price - position.price) if buy else (position.price - exit_price)) * units
        self.balance += gross + self.close_commission
        del self.positions[position.position_id]
        if self.drop_on_close:
            conn.writer.transport.abort()
            return
        if not self.emit_close_event:
            return
        scale = 10**self.money_digits
        detail = model.ProtoOAClosePositionDetail(
            entryPrice=position.price,
            grossProfit=round(gross * scale),
            swap=0,
            commission=round(self.close_commission * scale),
            balance=round(self.balance * scale),
            moneyDigits=self.money_digits,
            closedVolume=position.volume,
        )
        deal = model.ProtoOADeal(
            dealId=position.position_id + 9_000,
            orderId=position.position_id + 7_000,
            positionId=position.position_id,
            volume=position.volume,
            filledVolume=position.volume,
            symbolId=position.symbol_id,
            createTimestamp=int(time.time() * 1000),
            executionTimestamp=int(time.time() * 1000),
            executionPrice=exit_price,
            tradeSide=model.SELL if buy else model.BUY,
            dealStatus=model.FILLED,
            closePositionDetail=detail,
        )
        event = messages.ProtoOAExecutionEvent(
            ctidTraderAccountId=self.account_id,
            executionType=model.ORDER_FILLED,
            position=self._position_proto(position),
            deal=deal,
        )
        await self._event(conn, event, msg_id)

"""`Venue` on the cTrader Open API: what the runner asks of a broker, done over `CTraderClient`.

The broker's own state is the only truth. A market order carries the runner's `client_order_id` in
its `label` (and in `clientOrderId` when it fits in 50 characters); the position that comes back
from `ProtoOAReconcileReq` carries it again, so the id survives a restart and a repeated
`market_order` finds the position it already opened instead of sending a second order. Trading
events (`ProtoOAExecutionEvent`, `ProtoOAOrderErrorEvent`, error answers) are only a signal that
an outcome is known: after each order, amendment or close the adapter asks the broker again what
is open, and reports what it finds. Trades run one at a time; that is also what lets an error that
names no order be attributed to the one in flight.

A market order cannot carry an absolute stop, so it goes out with `relativeStopLoss` and
`relativeTakeProfit` measured from the current quote, and as soon as the position exists it is
amended to the exact absolute levels (skipped when the fill already landed on them). If that
amendment is refused (here and in `amend_protection`, where the runner's callers do not expect an
exception) the position keeps the protection it has and is returned as it stands, the refusal
code goes to the log; if it has no stop at all `Position.stop_price` is `None` and the caller
closes it.

`account()` is the balance plus the unrealized P&L of the open positions, valued with the cached
spot prices and `app.fx.instruments` (gross P&L: accrued swap and the commission of open positions
are left out, they reach the balance when the position closes). A missing price is
`VenueUnavailable`, never a guess. `spots()` yields the `Quote`s for the feed.

Unverified assumptions (no credentials yet; the probe `scripts/research/ctrader_probe.py` settles
each one, item numbers are those of section 4.4 of docs/forex/ctrader-paths.md, "preflight" is a
check the probe runs before item 1):

* Relative stop and target on a MARKET order are applied to the fill (item 1), and
  `AmendPositionSLTP` right after the fill sets the exact absolute levels, which the reconcile
  then shows (item 2). Amending with the target left out may keep or clear the old target; only
  what the reconcile shows is trusted.
* `ProtoOAPosition.tradeData.label` returns the label given to the order (item 5). It is the one
  thing idempotency depends on: if a fill is seen and the label is not found the adapter halts
  (`CTraderProtocolError`) instead of retrying into a second position.
* A reconcile sent after an order sees that order's fill (item 5).
* The shape of the trading events (which ones arrive, what they carry, whether an error names the
  request) is not relied on beyond `label`/`positionId` matching and `errorCode`; they only end
  the wait early (items 1 and 8).
* Closing deal: `closePositionDetail` has the realized amounts in `moneyDigits` decimals and the
  net result is `grossProfit + swap - |commission| - |pnlConversionFee|`; the sign convention of
  the costs is not documented, so they are always taken as costs (item 9).
* Symbol names are `EURUSD` (a `/` is stripped), volumes are in 1/100 of a unit with
  `minVolume`/`stepVolume` in the same unit (preflight and item 8), `pipPosition` gives the pip
  the runner assumes, and the deposit asset name is the currency code (preflight).
* The account is a hedging account (`accountType`, preflight): with netting the orders of one
  symbol merge into one position and the labels are lost. What Fusion actually uses is
  unverified (docs/forex/ctrader-paths.md, section 8).
* A spot event may carry only one side of the price, and its `timestamp` is Unix milliseconds
  (items 7 and 9); a timestamp outside a 30-day window around the local clock is ignored.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

from google.protobuf.message import Message

from app.fx import strategy as fx
from app.fx.instruments import (
    ConversionRates,
    Instrument,
    MissingRateError,
    UnknownInstrumentError,
    profit,
)
from app.fx.runner.ctrader.client import PRICE_SCALE, CTraderClient, ServerError, server_error
from app.fx.runner.ctrader.proto import OpenApiCommonMessages_pb2 as common
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from app.fx.runner.ctrader.proto import OpenApiModelMessages_pb2 as model
from app.fx.runner.venue import Account, OrderRejected, Position, Quote, VenueUnavailable

log = logging.getLogger(__name__)

UNITS_TO_PROTOCOL = 100  # protocol volumes are in 1/100 of a unit
MAX_LABEL = 100
MAX_CLIENT_ORDER_ID = 50
SPOT_TIME_WINDOW_SECONDS = 30 * 86_400
QUOTE_QUEUE_SIZE = 50_000

# Refusals that say "not now" rather than "no": the order was not taken and may be sent again.
TRANSIENT_CODES = frozenset(
    {
        "BLOCKED_PAYLOAD_TYPE",
        "REQUEST_FREQUENCY_EXCEEDED",
        "SERVER_IS_UNDER_MAINTENANCE",
        "CH_SERVER_NOT_REACHABLE",
        "CANT_ROUTE_REQUEST",
        "TIMEOUT_ERROR",
        "CONCURRENT_MODIFICATION",
        "PENDING_EXECUTION",
        "POSITION_LOCKED",
    }
)


class CTraderConfigError(Exception):
    """The broker account is not what the runner assumes (symbol, pip, currency, account type)."""


class CTraderProtocolError(Exception):
    """The broker behaves in a way the adapter cannot make safe; stop trading until understood."""


@dataclass(frozen=True)
class SymbolSpec:
    symbol_id: int
    digits: int
    min_volume: int | None
    step_volume: int | None
    max_volume: int | None


@dataclass
class _TradeWatch:
    """The events that can be the outcome of the one trading request in flight."""

    msg_id: str
    label: str | None = None
    position_id: int | None = None
    events: asyncio.Queue[Message] = field(default_factory=asyncio.Queue)

    def wants(self, message: Message, client_msg_id: str | None) -> bool:
        if isinstance(message, messages.ProtoOAExecutionEvent):
            return self._is_about_us(message)
        if client_msg_id and client_msg_id != self.msg_id:
            return False  # an answer to some other request
        if isinstance(message, messages.ProtoOAOrderErrorEvent) and self.position_id is not None:
            return not message.HasField("positionId") or message.positionId == self.position_id
        return True

    def _is_about_us(self, event: messages.ProtoOAExecutionEvent) -> bool:
        if self.label and self.label in (
            event.order.tradeData.label,
            event.position.tradeData.label,
        ):
            return True
        ids = (event.position.positionId, event.deal.positionId, event.order.positionId)
        return self.position_id is not None and self.position_id in ids


_TRADE_MESSAGES = (
    messages.ProtoOAExecutionEvent,
    messages.ProtoOAOrderErrorEvent,
    messages.ProtoOAErrorRes,
    common.ProtoErrorRes,
)


def _normalize(symbol_name: str) -> str:
    return symbol_name.replace("/", "").upper()


def _level(raw: model.ProtoOAPosition, name: str) -> float | None:
    return getattr(raw, name) if raw.HasField(name) and getattr(raw, name) > 0 else None


def _same_price(first: float | None, second: float | None, digits: int) -> bool:
    if first is None or second is None:
        return first is second
    return abs(first - second) < 0.5 * 10.0**-digits


def _levels_match(
    stop_now: float | None,
    target_now: float | None,
    stop: float,
    target: float | None,
    digits: int,
) -> bool:
    return _same_price(stop_now, stop, digits) and _same_price(target_now, target, digits)


def _error_of(event: Message) -> ServerError | None:
    if isinstance(event, messages.ProtoOAExecutionEvent):
        if event.executionType == model.ORDER_REJECTED or event.HasField("errorCode"):
            return ServerError(event.errorCode or "ORDER_REJECTED", "")
        return None
    return server_error(event)


def _is_fill(event: Message) -> bool:
    return isinstance(event, messages.ProtoOAExecutionEvent) and event.executionType in (
        model.ORDER_FILLED,
        model.ORDER_PARTIAL_FILL,
    )


def _is_closing_deal(event: Message) -> bool:
    return (
        isinstance(event, messages.ProtoOAExecutionEvent)
        and event.HasField("deal")
        and event.deal.HasField("closePositionDetail")
    )


def _has_levels(event: Message, stop: float, target: float | None, digits: int) -> bool:
    if not isinstance(event, messages.ProtoOAExecutionEvent) or not event.HasField("position"):
        return False
    return _levels_match(
        _level(event.position, "stopLoss"),
        _level(event.position, "takeProfit"),
        stop,
        target,
        digits,
    )


def _realized_result(detail: model.ProtoOAClosePositionDetail, money_digits: int) -> float:
    """Net result of a closing deal in the account currency; costs always count against."""
    digits = detail.moneyDigits if detail.HasField("moneyDigits") else money_digits
    costs = abs(detail.commission)
    if detail.HasField("pnlConversionFee"):
        costs += abs(detail.pnlConversionFee)
    return (detail.grossProfit + detail.swap - costs) / 10**digits


class CTraderVenue:
    def __init__(
        self,
        client: CTraderClient,
        *,
        symbols: Sequence[str],
        account_currency: str = "USD",
        trade_timeout: float = 5.0,
        confirm_pause: float = 0.5,
        max_quote_age: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        for symbol in symbols:
            Instrument.from_symbol(symbol)  # refuse what the runner cannot value, up front
        self._client = client
        self._symbols = tuple(symbols)
        self._currency = account_currency
        self._trade_timeout, self._confirm_pause = trade_timeout, confirm_pause
        self._max_quote_age, self._clock = max_quote_age, clock
        self._names_by_id: dict[int, str] = {}
        self._specs: dict[str, SymbolSpec] = {}
        self._sides: dict[str, list[float | None]] = {}
        self._quotes: dict[str, Quote] = {}
        self._quote_queue: asyncio.Queue[Quote] = asyncio.Queue(QUOTE_QUEUE_SIZE)
        self.dropped_quotes = 0
        self._money_digits: int | None = None
        self._trade_lock = asyncio.Lock()
        self._watch: _TradeWatch | None = None
        self._closing: dict[str, float] = {}  # position id -> balance before an unconfirmed close
        self._halted: str | None = None
        client.add_listener(self._on_message)

    async def start(self) -> None:
        """Check the account, load the symbols, subscribe to prices. Run once the client is up."""
        account = self._client.account_id
        listing = await self._client.request(
            messages.ProtoOASymbolsListReq(ctidTraderAccountId=account),
            messages.ProtoOASymbolsListRes,
        )
        self._names_by_id = {s.symbolId: _normalize(s.symbolName) for s in listing.symbol}
        ids_by_name = {name: symbol_id for symbol_id, name in self._names_by_id.items()}
        missing = [symbol for symbol in self._symbols if symbol not in ids_by_name]
        if missing:
            raise CTraderConfigError(f"the broker has no symbol {', '.join(missing)}")
        detail = await self._client.request(
            messages.ProtoOASymbolByIdReq(
                ctidTraderAccountId=account, symbolId=[ids_by_name[s] for s in self._symbols]
            ),
            messages.ProtoOASymbolByIdRes,
        )
        for symbol in detail.symbol:
            name = self._names_by_id[symbol.symbolId]
            self._specs[name] = self._spec_of(name, symbol)
        trader = await self._trader()
        if trader.accountType != model.HEDGED:
            raise CTraderConfigError("only hedging accounts are supported: netting merges orders")
        await self._check_currency(trader)
        await self._client.subscribe_spots([ids_by_name[s] for s in self._symbols])

    @staticmethod
    def _spec_of(name: str, symbol: model.ProtoOASymbol) -> SymbolSpec:
        pip_size = Instrument.from_symbol(name).pip_size
        if abs(10.0**-symbol.pipPosition - pip_size) > 1e-12:
            raise CTraderConfigError(
                f"{name}: the broker's pip is 10^-{symbol.pipPosition}, the runner uses {pip_size}"
            )

        def optional(field_name: str) -> int | None:
            return getattr(symbol, field_name) if symbol.HasField(field_name) else None

        return SymbolSpec(
            symbol.symbolId,
            symbol.digits,
            optional("minVolume"),
            optional("stepVolume"),
            optional("maxVolume"),
        )

    async def _check_currency(self, trader: model.ProtoOATrader) -> None:
        assets = await self._client.request(
            messages.ProtoOAAssetListReq(ctidTraderAccountId=self._client.account_id),
            messages.ProtoOAAssetListRes,
        )
        names = {asset.assetId: asset.name for asset in assets.asset}
        currency = names.get(trader.depositAssetId)
        if currency != self._currency:
            raise CTraderConfigError(
                f"the account is in {currency}, the runner values money in {self._currency}"
            )

    async def spots(self) -> AsyncIterator[Quote]:
        """Every price update, in arrival order. One consumer: the feed."""
        while True:
            yield await self._quote_queue.get()

    def conversion_rates(self) -> ConversionRates:
        return ConversionRates({s: (q.bid + q.ask) / 2 for s, q in self._quotes.items()})

    async def quote(self, symbol: str) -> Quote:
        quote = self._quotes.get(symbol)
        if quote is None:
            raise VenueUnavailable(f"no price for {symbol} has arrived yet")
        age = self._clock() - quote.time
        if age > self._max_quote_age:
            raise VenueUnavailable(f"the last price of {symbol} is {age:.0f}s old")
        return quote

    async def account(self) -> Account:
        balance = await self._balance()
        floating = self._floating_pnl(await self.positions())
        return Account(balance, balance + floating, self._currency)

    async def positions(self) -> list[Position]:
        return [self._to_position(raw) for raw in await self._reconcile()]

    async def market_order(
        self,
        *,
        symbol: str,
        side: int,
        units: int,
        stop_price: float,
        target_price: float | None,
        client_order_id: str,
    ) -> Position:
        if side not in (fx.LONG, fx.SHORT) or units <= 0:
            raise ValueError("side must be LONG or SHORT and units positive")
        if not client_order_id or len(client_order_id) > MAX_LABEL:
            raise ValueError(f"client_order_id must have 1 to {MAX_LABEL} characters")
        if self._halted:
            raise CTraderProtocolError(self._halted)
        spec = self.spec_for(symbol)
        self._check_volume(spec, units)
        async with self._trade_lock:
            position = await self._find_by_label(client_order_id)
            if position is None:
                position = await self._open(
                    spec, symbol, side, units, stop_price, target_price, client_order_id
                )
            return await self._amend_or_keep(position, stop_price, target_price)

    async def amend_protection(
        self, position_id: str, *, stop_price: float, target_price: float | None
    ) -> Position:
        async with self._trade_lock:
            position = await self._find_position(position_id)
            if position is None:
                raise OrderRejected(f"POSITION_NOT_FOUND: position {position_id} is not open")
            return await self._amend_or_keep(position, stop_price, target_price)

    async def close(self, position_id: str) -> float:
        async with self._trade_lock:
            raw = await self._find_raw(position_id)
            before = self._closing.get(position_id)
            if raw is None:
                if before is None:
                    raise OrderRejected(f"POSITION_NOT_FOUND: position {position_id} is not open")
                return await self._settle_close(position_id, before)  # an earlier attempt closed it
            if before is None:
                before = self._closing[position_id] = await self._balance()
            request = messages.ProtoOAClosePositionReq(
                ctidTraderAccountId=self._client.account_id,
                positionId=int(position_id),
                volume=raw.tradeData.volume,
            )
            event = await self._trade(request, position_id=int(position_id), done=_is_closing_deal)
            if event is not None and _is_closing_deal(event):
                self._closing.pop(position_id, None)
                return _realized_result(event.deal.closePositionDetail, self._digits())
            if await self._find_raw(position_id) is not None:
                error = None if event is None else _error_of(event)
                if error is None:
                    raise VenueUnavailable(f"closing position {position_id} was not confirmed")
                del self._closing[position_id]
                raise self._refusal(error)
            return await self._settle_close(position_id, before)

    async def _settle_close(self, position_id: str, balance_before: float) -> float:
        balance_after = await self._balance()
        self._closing.pop(position_id, None)
        return balance_after - balance_before

    async def _open(
        self,
        spec: SymbolSpec,
        symbol: str,
        side: int,
        units: int,
        stop_price: float,
        target_price: float | None,
        label: str,
    ) -> Position:
        quote = await self.quote(symbol)
        entry = quote.ask if side == fx.LONG else quote.bid
        request = messages.ProtoOANewOrderReq(
            ctidTraderAccountId=self._client.account_id,
            symbolId=spec.symbol_id,
            orderType=model.MARKET,
            tradeSide=model.BUY if side == fx.LONG else model.SELL,
            volume=units * UNITS_TO_PROTOCOL,
            label=label,
            relativeStopLoss=self._relative(spec, (entry - stop_price) * side, "stop"),
        )
        if len(label) <= MAX_CLIENT_ORDER_ID:
            request.clientOrderId = label
        if target_price is not None:
            request.relativeTakeProfit = self._relative(
                spec, (target_price - entry) * side, "target"
            )
        event = await self._trade(request, label=label, done=_is_fill)
        error = None if event is None else _error_of(event)
        filled = event is not None and error is None
        for attempt in range(2 if filled else 1):
            if attempt:
                await asyncio.sleep(self._confirm_pause)
            position = await self._find_by_label(label)
            if position is not None:
                return position
        if filled:
            self._halted = "the broker filled an order but its label is not in the reconcile"
            raise CTraderProtocolError(f"{self._halted} (order {label}); idempotency is broken")
        if error is not None:
            raise self._refusal(error)
        raise VenueUnavailable(f"the outcome of order {label} is unknown; check before resending")

    async def _amend_or_keep(
        self, position: Position, stop_price: float, target_price: float | None
    ) -> Position:
        """Amend; a refusal is reported as the position stands, not raised: the callers (executor,
        reconcile) judge `stop_price` and close a position that has none, and catch nothing."""
        try:
            return await self._amend(position, stop_price, target_price)
        except OrderRejected as refusal:
            log.warning("cTrader refused new protection for position %s: %s", position.id, refusal)
            current = await self._find_position(position.id)
            if current is None:
                raise
            return current

    async def _amend(
        self, position: Position, stop_price: float, target_price: float | None
    ) -> Position:
        """Bring the server-side stop and target to these levels and report what is there."""
        spec = self.spec_for(position.symbol)
        digits = spec.digits
        if _levels_match(
            position.stop_price, position.target_price, stop_price, target_price, digits
        ):
            return position
        request = messages.ProtoOAAmendPositionSLTPReq(
            ctidTraderAccountId=self._client.account_id,
            positionId=int(position.id),
            stopLoss=round(stop_price, digits),
        )
        if target_price is not None:
            request.takeProfit = round(target_price, digits)
        event = await self._trade(
            request,
            position_id=int(position.id),
            done=lambda e: _has_levels(e, stop_price, target_price, digits),
        )
        current = await self._find_position(position.id)
        if current is None:
            raise OrderRejected(f"POSITION_NOT_FOUND: {position.id} closed while being amended")
        if _levels_match(
            current.stop_price, current.target_price, stop_price, target_price, digits
        ):
            return current
        error = None if event is None else _error_of(event)
        if error is not None:
            raise self._refusal(error)
        return current  # not applied, or not applied yet: the caller judges the protection it has

    async def _trade(
        self,
        request: Message,
        *,
        label: str | None = None,
        position_id: int | None = None,
        done: Callable[[Message], bool],
    ) -> Message | None:
        """Send a trading request; the first event that settles it, or `None` if none arrives."""
        watch = _TradeWatch(self._client.next_id(), label, position_id)
        self._watch = watch
        try:
            await self._client.submit(request, watch.msg_id)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._trade_timeout
            while (remaining := deadline - loop.time()) > 0:
                try:
                    event = await asyncio.wait_for(watch.events.get(), remaining)
                except TimeoutError:
                    return None
                if _error_of(event) is not None or done(event):
                    return event
            return None
        finally:
            self._watch = None

    def _refusal(self, error: ServerError) -> Exception:
        text = self._client.redact(
            f"{error.code}: {error.description}" if error.description else error.code
        )
        if error.code in TRANSIENT_CODES:
            return VenueUnavailable(text)
        return OrderRejected(text)

    async def _reconcile(self) -> list[model.ProtoOAPosition]:
        response = await self._client.request(
            messages.ProtoOAReconcileReq(ctidTraderAccountId=self._client.account_id),
            messages.ProtoOAReconcileRes,
        )
        return [p for p in response.position if p.positionStatus == model.POSITION_STATUS_OPEN]

    async def _find_raw(self, position_id: str) -> model.ProtoOAPosition | None:
        return next((p for p in await self._reconcile() if str(p.positionId) == position_id), None)

    async def _find_position(self, position_id: str) -> Position | None:
        raw = await self._find_raw(position_id)
        return None if raw is None else self._to_position(raw)

    async def _find_by_label(self, label: str) -> Position | None:
        raw = next((p for p in await self._reconcile() if p.tradeData.label == label), None)
        return None if raw is None else self._to_position(raw)

    def _to_position(self, raw: model.ProtoOAPosition) -> Position:
        if not raw.HasField("price"):
            raise VenueUnavailable(f"position {raw.positionId} has no entry price")
        symbol_id = raw.tradeData.symbolId
        return Position(
            id=str(raw.positionId),
            symbol=self._names_by_id.get(symbol_id, f"#{symbol_id}"),
            side=fx.LONG if raw.tradeData.tradeSide == model.BUY else fx.SHORT,
            units=raw.tradeData.volume // UNITS_TO_PROTOCOL,
            entry_price=raw.price,
            stop_price=_level(raw, "stopLoss"),
            target_price=_level(raw, "takeProfit"),
            client_order_id=raw.tradeData.label,
        )

    async def _trader(self) -> model.ProtoOATrader:
        response = await self._client.request(
            messages.ProtoOATraderReq(ctidTraderAccountId=self._client.account_id),
            messages.ProtoOATraderRes,
        )
        if not response.trader.HasField("moneyDigits"):
            raise VenueUnavailable("cTrader did not say how many decimals the balance has")
        self._money_digits = response.trader.moneyDigits
        return response.trader

    async def _balance(self) -> float:
        trader = await self._trader()
        return trader.balance / 10**trader.moneyDigits

    def _digits(self) -> int:
        if self._money_digits is None:
            raise VenueUnavailable("the decimals of the balance are not known yet")
        return self._money_digits

    def _floating_pnl(self, positions: Sequence[Position]) -> float:
        rates = self.conversion_rates()
        total = 0.0
        for position in positions:
            quote = self._quotes.get(position.symbol)
            if quote is None:
                raise VenueUnavailable(f"no price for {position.symbol} to value the open position")
            long = position.side == fx.LONG
            try:
                total += profit(
                    Instrument.from_symbol(position.symbol),
                    side="LONG" if long else "SHORT",
                    units=position.units,
                    entry_price=position.entry_price,
                    exit_price=quote.bid if long else quote.ask,
                    rates=rates,
                    account_currency=self._currency,
                )
            except (UnknownInstrumentError, MissingRateError) as error:
                raise VenueUnavailable(
                    f"cannot value the position on {position.symbol}: {error}"
                ) from error
        return total

    def spec_for(self, symbol: str) -> SymbolSpec:
        spec = self._specs.get(symbol)
        if spec is None:
            raise OrderRejected(f"SYMBOL_NOT_FOUND: {symbol} is not a symbol this venue trades")
        return spec

    @staticmethod
    def _check_volume(spec: SymbolSpec, units: int) -> None:
        volume = units * UNITS_TO_PROTOCOL
        too_small = spec.min_volume is not None and volume < spec.min_volume
        too_large = spec.max_volume is not None and volume > spec.max_volume
        off_step = spec.step_volume is not None and volume % spec.step_volume != 0
        if too_small or too_large or off_step:
            raise OrderRejected(
                f"TRADING_BAD_VOLUME: {units} units do not fit the broker's volume rules"
            )

    @staticmethod
    def _relative(spec: SymbolSpec, distance: float, what: str) -> int:
        relative = round(round(distance, spec.digits) * PRICE_SCALE)
        if relative <= 0:
            raise OrderRejected(f"TRADING_BAD_STOPS: the {what} is not beyond the current price")
        return relative

    def _on_message(self, message: Message, client_msg_id: str | None) -> None:
        if isinstance(message, messages.ProtoOASpotEvent):
            self._on_spot(message)
            return
        watch = self._watch
        if watch is not None and isinstance(message, _TRADE_MESSAGES):
            if watch.wants(message, client_msg_id):
                watch.events.put_nowait(message)

    def _on_spot(self, event: messages.ProtoOASpotEvent) -> None:
        symbol = self._names_by_id.get(event.symbolId)
        if symbol is None:
            return
        sides = self._sides.setdefault(symbol, [None, None])
        if event.HasField("bid") and event.bid > 0:
            sides[0] = event.bid / PRICE_SCALE
        if event.HasField("ask") and event.ask > 0:
            sides[1] = event.ask / PRICE_SCALE
        bid, ask = sides
        if bid is None or ask is None or ask < bid:
            return  # one side not seen yet, or a transient cross between two half updates
        quote = Quote(symbol, bid, ask, self._spot_time(event))
        self._quotes[symbol] = quote
        if self._quote_queue.full():
            self._quote_queue.get_nowait()
            self.dropped_quotes += 1
        self._quote_queue.put_nowait(quote)

    def _spot_time(self, event: messages.ProtoOASpotEvent) -> float:
        now = self._clock()
        if event.HasField("timestamp"):
            stamped = event.timestamp / 1000.0
            if abs(now - stamped) <= SPOT_TIME_WINDOW_SECONDS:
                return stamped
        return now

"""Demo-only probe of the cTrader Open API: the test matrix of docs/forex/ctrader-paths.md, 4.4.

Run it by hand once the demo credentials exist, from `backend/`:

    CTRADER_CLIENT_ID=... CTRADER_CLIENT_SECRET=... CTRADER_ACCESS_TOKEN=... \\
    CTRADER_ACCOUNT_ID=... .venv/bin/python -m scripts.research.ctrader_probe [--items 1,2,3]

It talks to `demo.ctraderapi.com` only and refuses any other host. It opens positions of the
minimum size with a stop a few pips away, closes them, and prints one line per item: PASS, FAIL,
OBSERVATION (what the broker did, to be copied into the doc) or SKIPPED, then a summary. Values
of the credentials are never printed. The exit code is 1 if any item failed.

Items follow the matrix: 1 relative stop and target on a market order, 2 market order then amend
(and the unprotected window), 3 MARKET_RANGE with an absolute stop, 4 kill -9 with a position open
(off by default: it leaves a position on the demo on purpose; add `--items 4`, and `--kill` to make
the script end itself with SIGKILL), 5 reconnect and reconcile without a duplicate (after item 4,
run `--items 5 --label <the label item 4 printed>`), 6 token refresh (not covered here), 7
heartbeat, forced drop and resubscription of the spots, 8 broker errors, 9 cost and slippage. Two
extra checks come first: `pre` (account, symbol and volume rules) and `A` (a wrong token must be a
fatal error, not a reconnection loop).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from google.protobuf.message import Message

from app.fx import strategy as fx
from app.fx.instruments import Instrument
from app.fx.runner.ctrader.client import (
    DEMO_HOST,
    PORT,
    PRICE_SCALE,
    Credentials,
    CTraderAuthError,
    CTraderClient,
)
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from app.fx.runner.ctrader.proto import OpenApiModelMessages_pb2 as model
from app.fx.runner.ctrader.venue import CTraderVenue
from app.fx.runner.venue import Position

ENV_NAMES = (
    "CTRADER_CLIENT_ID",
    "CTRADER_CLIENT_SECRET",
    "CTRADER_ACCESS_TOKEN",
    "CTRADER_ACCOUNT_ID",
)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})  # the fake server of the tests
DEFAULT_ITEMS = ("pre", "A", "1", "2", "3", "5", "6", "7", "8", "9")
ITEM_TIMEOUT_SECONDS = 180.0
PASS, FAIL, OBSERVATION, SKIPPED = "PASS", "FAIL", "OBSERVATION", "SKIPPED"


class ProbeConfigError(Exception):
    """The probe was started wrongly; the message says how, and never quotes a credential."""


@dataclass(frozen=True)
class Outcome:
    item: str
    title: str
    status: str
    detail: str

    def line(self) -> str:
        return f"[{self.status}] {self.item}: {self.title} - {self.detail}"


def credentials_from_env(env: Mapping[str, str]) -> Credentials:
    missing = [name for name in ENV_NAMES if not env.get(name)]
    if missing:
        raise ProbeConfigError(f"missing environment variables: {', '.join(missing)}")
    try:
        account_id = int(env["CTRADER_ACCOUNT_ID"])
    except ValueError as error:
        raise ProbeConfigError("CTRADER_ACCOUNT_ID must be the numeric account id") from error
    return Credentials(
        env["CTRADER_CLIENT_ID"],
        env["CTRADER_CLIENT_SECRET"],
        env["CTRADER_ACCESS_TOKEN"],
        account_id,
    )


def check_host(host: str) -> None:
    """Only the demo endpoint (or this machine, for the tests) may be probed."""
    if host != DEMO_HOST and host not in LOOPBACK_HOSTS:
        raise ProbeConfigError(f"refusing to probe {host!r}: this probe runs on {DEMO_HOST} only")


class EventTap:
    """Remembers what the broker sends, with local arrival times, so an item can look back."""

    def __init__(self, client: CTraderClient) -> None:
        self.events: list[tuple[float, Message]] = []
        client.add_listener(
            lambda message, client_msg_id: self.events.append((time.monotonic(), message))
        )

    def mark(self) -> int:
        return len(self.events)

    async def wait(
        self, mark: int, accept: Callable[[Message], bool], timeout: float = 8.0
    ) -> tuple[float, Message] | None:
        deadline = time.monotonic() + timeout
        seen = mark
        while time.monotonic() < deadline:
            while seen < len(self.events):
                stamp, message = self.events[seen]
                seen += 1
                if accept(message):
                    return stamp, message
            await asyncio.sleep(0.02)
        return None

    def of_type(self, mark: int, message_class: type[Message]) -> list[Message]:
        return [m for _, m in self.events[mark:] if isinstance(m, message_class)]


def error_code(message: Message) -> str | None:
    if isinstance(message, (messages.ProtoOAOrderErrorEvent, messages.ProtoOAErrorRes)):
        return message.errorCode
    if isinstance(message, messages.ProtoOAExecutionEvent):
        if message.executionType == model.ORDER_REJECTED or message.HasField("errorCode"):
            return message.errorCode or "ORDER_REJECTED"
    return None


class Probe:
    def __init__(
        self,
        client: CTraderClient,
        venue: CTraderVenue,
        make_client: Callable[[Credentials], CTraderClient],
        credentials: Credentials,
        *,
        symbol: str = "EURUSD",
        sl_pips: float = 8.0,
        kill: bool = False,
        resume_label: str | None = None,
        idle_seconds: float = 40.0,
    ) -> None:
        self.client, self.venue, self.make_client, self.credentials = (
            client,
            venue,
            make_client,
            credentials,
        )
        self.symbol, self.sl_pips, self.kill = symbol, sl_pips, kill
        self.resume_label, self.idle_seconds = resume_label, idle_seconds
        self.spec = venue.spec_for(symbol)
        self.pip = Instrument.from_symbol(symbol).pip_size
        self.units = max(1000, (self.spec.min_volume or 100_000) // 100)
        self.token = uuid.uuid4().hex[:6]
        self.tap = EventTap(client)
        self.kill_when_done = False

    # helpers

    def label(self, name: str) -> str:
        return f"probe-{self.token}-{name}"

    def relative(self, pips: float) -> int:
        return round(round(pips * self.pip, self.spec.digits) * PRICE_SCALE)

    async def order(self, label: str, **fields: object) -> None:
        """A raw order, bypassing the venue's own checks; the label doubles as the client id."""
        fields = {
            "orderType": model.MARKET,
            "tradeSide": model.BUY,
            "volume": self.units * 100,
            **fields,
        }
        request = messages.ProtoOANewOrderReq(
            ctidTraderAccountId=self.client.account_id,
            symbolId=self.spec.symbol_id,
            label=label,
            clientOrderId=label,
            **fields,
        )
        await self.client.submit(request, self.client.next_id())

    def is_fill(self, label: str) -> Callable[[Message], bool]:
        def accept(message: Message) -> bool:
            return (
                isinstance(message, messages.ProtoOAExecutionEvent)
                and message.executionType in (model.ORDER_FILLED, model.ORDER_PARTIAL_FILL)
                and label in (message.order.tradeData.label, message.position.tradeData.label)
            )

        return accept

    async def find(self, label: str, timeout: float = 6.0) -> Position | None:
        deadline = time.monotonic() + timeout
        while True:
            for position in await self.venue.positions():
                if position.client_order_id == label:
                    return position
            if time.monotonic() > deadline:
                return None
            await asyncio.sleep(0.4)

    async def wait_for_reconnection(self, sessions: int, timeout: float = 60.0) -> float | None:
        started = time.monotonic()
        while self.client.sessions <= sessions:
            if time.monotonic() - started > timeout:
                return None
            await asyncio.sleep(0.05)
        return time.monotonic() - started

    def outcome(self, item: str, title: str, status: str, detail: str) -> Outcome:
        return Outcome(item, title, status, self.client.redact(detail))

    def levels_near(self, position: Position, stop: float, target: float | None) -> bool:
        tolerance = 0.5 * self.pip
        stop_ok = position.stop_price is not None and abs(position.stop_price - stop) <= tolerance
        if target is None:
            return stop_ok
        return (
            stop_ok
            and position.target_price is not None
            and abs(position.target_price - target) <= tolerance
        )

    # items

    async def item_pre(self) -> Outcome:
        account = await self.venue.account()
        quote = await self.venue.quote(self.symbol)
        spec = self.spec
        detail = (
            f"{self.symbol} digits={spec.digits} minVolume={spec.min_volume} "
            f"stepVolume={spec.step_volume} "
            f"maxVolume={spec.max_volume}; balance {account.balance:.2f} {account.currency}; "
            f"quote {quote.bid}/{quote.ask}; hedging and currency were checked by start()"
        )
        return self.outcome("pre", "account, symbol and volume rules", PASS, detail)

    async def item_A(self) -> Outcome:
        wrong = Credentials(
            self.credentials.client_id,
            self.credentials.client_secret,
            "invalid-probe-token",
            self.credentials.account_id,
        )
        client = self.make_client(wrong)
        started = time.monotonic()
        try:
            await client.start(timeout=20.0)
        except CTraderAuthError as error:
            detail = f"fatal after {time.monotonic() - started:.1f}s, no loop: {error}"
            return self.outcome("A", "a wrong token is a fatal error", PASS, detail)
        finally:
            await client.stop()
        return self.outcome(
            "A", "a wrong token is a fatal error", FAIL, "the broker accepted a wrong token"
        )

    async def item_1(self) -> Outcome:
        title = "market order with relative stop and target"
        label = self.label("i1")
        mark = self.tap.mark()
        await self.order(
            label,
            relativeStopLoss=self.relative(self.sl_pips),
            relativeTakeProfit=self.relative(2 * self.sl_pips),
        )
        fill = await self.tap.wait(mark, self.is_fill(label))
        position = await self.find(label)
        if position is None:
            return self.outcome(
                "1",
                title,
                FAIL,
                "no position carries the label after the order: the label does not round-trip",
            )
        try:
            stop = position.entry_price - self.sl_pips * self.pip
            target = position.entry_price + 2 * self.sl_pips * self.pip
            in_event = "no fill event"
            if fill is not None:
                carried = fill[1].position
                in_event = (
                    f"fill event carried SL={carried.stopLoss or None} "
                    f"TP={carried.takeProfit or None}"
                )
            detail = (
                f"reconcile shows SL={position.stop_price} TP={position.target_price} for entry "
                f"{position.entry_price} (expected SL={stop:.5f} TP={target:.5f}); {in_event}; "
                f"label round-trip ok"
            )
            status = PASS if self.levels_near(position, stop, target) else FAIL
            return self.outcome("1", title, status, detail)
        finally:
            await self.venue.close(position.id)

    async def item_2(self) -> Outcome:
        title = "market order without stop, amended right after the fill"
        label = self.label("i2")
        mark = self.tap.mark()
        await self.order(label)
        fill = await self.tap.wait(mark, self.is_fill(label))
        if fill is None:
            return self.outcome("2", title, FAIL, "no fill event within 8s")
        position_id = fill[1].position.positionId or None
        position = await self.find(label)
        if position is None:
            return self.outcome("2", title, FAIL, "no position carries the label")
        try:
            stop = position.entry_price - self.sl_pips * self.pip
            request = messages.ProtoOAAmendPositionSLTPReq(
                ctidTraderAccountId=self.client.account_id,
                positionId=int(position.id),
                stopLoss=round(stop, self.spec.digits),
            )
            sent = time.monotonic()
            await self.client.submit(request, self.client.next_id())
            protected_at = None
            while time.monotonic() - sent < 6.0:
                current = next(
                    (p for p in await self.venue.positions() if p.id == position.id), None
                )
                if current is not None and self.levels_near(current, stop, None):
                    protected_at = time.monotonic()
                    break
                await asyncio.sleep(0.1)
            if protected_at is None:
                return self.outcome(
                    "2", title, FAIL, "the stop was not visible in the reconcile within 6s"
                )
            detail = (
                f"fill -> amend sent {1000 * (sent - fill[0]):.0f}ms (position id from the event: "
                f"{'yes' if position_id else 'no, from a reconcile'}); fill -> stop visible in the "
                f"reconcile {1000 * (protected_at - fill[0]):.0f}ms (upper bound)"
            )
            return self.outcome("2", title, PASS, detail)
        finally:
            await self.venue.close(position.id)

    async def item_3(self) -> Outcome:
        title = "MARKET_RANGE with an absolute stop"
        label = self.label("i3")
        quote = await self.venue.quote(self.symbol)
        stop = round(quote.ask - self.sl_pips * self.pip, self.spec.digits)
        mark = self.tap.mark()
        await self.order(
            label,
            orderType=model.MARKET_RANGE,
            baseSlippagePrice=quote.ask,
            slippageInPoints=20,
            stopLoss=stop,
        )
        position = await self.find(label)
        if position is None:
            codes = [c for m in self.tap.of_type(mark, Message) if (c := error_code(m))]
            return self.outcome(
                "3", title, OBSERVATION, f"no position opened; errors: {codes or 'none'}"
            )
        try:
            applied = (
                position.stop_price is not None
                and abs(position.stop_price - stop) <= 0.5 * self.pip
            )
            detail = f"accepted; requested SL={stop}, reconcile shows SL={position.stop_price}"
            return self.outcome("3", title, PASS if applied else OBSERVATION, detail)
        finally:
            await self.venue.close(position.id)

    async def item_4(self) -> Outcome:
        title = "kill -9 with a position open"
        label = self.label("kill")
        quote = await self.venue.quote(self.symbol)
        stop = quote.ask - self.sl_pips * self.pip
        position = await self.venue.market_order(
            symbol=self.symbol,
            side=fx.LONG,
            units=self.units,
            stop_price=stop,
            target_price=None,
            client_order_id=label,
        )
        self.kill_when_done = self.kill
        note = (
            "the process kills itself after the report"
            if self.kill
            else f"kill -9 {os.getpid()} yourself"
        )
        detail = (
            f"position {position.id} left open on purpose with SL={position.stop_price}, "
            f"label {label}; "
            f"{note}; then check in cTrader Web that the SL is still there and fires, and run "
            f"--items 5 --label {label}"
        )
        return self.outcome("4", title, OBSERVATION, detail)

    async def item_5(self) -> Outcome:
        title = "reconnect and reconcile without a duplicate order"
        if self.resume_label:
            return await self._resume(title)
        label = self.label("i5")
        quote = await self.venue.quote(self.symbol)
        stop = quote.ask - self.sl_pips * self.pip
        order = {
            "symbol": self.symbol,
            "side": fx.LONG,
            "units": self.units,
            "stop_price": stop,
            "target_price": None,
            "client_order_id": label,
        }
        first = await self.venue.market_order(**order)
        try:
            sessions = self.client.sessions
            self.client.drop_connection()
            reconnected = await self.wait_for_reconnection(sessions)
            if reconnected is None:
                return self.outcome("5", title, FAIL, "no reconnection within 60s")
            again = await self.venue.market_order(**order)
            same_label = [p for p in await self.venue.positions() if p.client_order_id == label]
            detail = (
                f"reconnected in {reconnected:.1f}s; replay returned position {again.id} "
                f"(first {first.id}); {len(same_label)} with the label"
            )
            return self.outcome(
                "5", title, PASS if again.id == first.id and len(same_label) == 1 else FAIL, detail
            )
        finally:
            await self.venue.close(first.id)

    async def _resume(self, title: str) -> Outcome:
        label = self.resume_label or ""
        found = [p for p in await self.venue.positions() if p.client_order_id == label]
        if len(found) != 1:
            return self.outcome(
                "5",
                title,
                OBSERVATION,
                f"{len(found)} positions carry {label}: it may have hit its stop, see the history",
            )
        position = found[0]
        if position.stop_price is None:
            return self.outcome(
                "5", title, FAIL, f"position {position.id} has NO stop after the kill"
            )
        again = await self.venue.market_order(
            symbol=position.symbol,
            side=position.side,
            units=position.units,
            stop_price=position.stop_price,
            target_price=position.target_price,
            client_order_id=label,
        )
        count = len([p for p in await self.venue.positions() if p.client_order_id == label])
        detail = (
            f"after the kill: position {position.id} still has SL={position.stop_price}; "
            f"replay returned {again.id}; {count} with the label"
        )
        return self.outcome(
            "5", title, PASS if again.id == position.id and count == 1 else FAIL, detail
        )

    async def item_6(self) -> Outcome:
        detail = (
            "the adapter has no refresh flow yet; test by hand with the Playground tokens "
            "and an atomic token store"
        )
        return self.outcome("6", "token refresh with atomic persistence", SKIPPED, detail)

    async def item_7(self) -> Outcome:
        title = "heartbeat, forced drop and resubscription of the spots"
        mark = self.tap.mark()
        first = await self.tap.wait(mark, lambda m: isinstance(m, messages.ProtoOASpotEvent), 15.0)
        if first is None:
            return self.outcome(
                "7",
                title,
                OBSERVATION,
                "no spot in 15s: the market looks closed, rerun while it is open",
            )
        sessions = self.client.sessions
        await asyncio.sleep(self.idle_seconds)
        held = self.client.sessions == sessions
        self.client.drop_connection()
        reconnected = await self.wait_for_reconnection(sessions)
        if reconnected is None:
            return self.outcome("7", title, FAIL, "no reconnection within 60s")
        after = self.tap.mark()
        spot = await self.tap.wait(after, lambda m: isinstance(m, messages.ProtoOASpotEvent), 15.0)
        detail = (
            f"link held {self.idle_seconds:.0f}s idle: {held}; reconnected in "
            f"{reconnected:.1f}s; spots "
            f"{'resumed' if spot else 'did NOT resume'} after the resubscription; the weekend "
            f"maintenance "
            f"window has to be watched by hand"
        )
        return self.outcome("7", title, PASS if held and spot else FAIL, detail)

    async def item_8(self) -> Outcome:
        title = "broker errors"
        found: list[str] = []
        mark = self.tap.mark()
        await self.order(self.label("i8a"), volume=(self.spec.min_volume or 100_000) + 1)
        codes = await self._error_codes_after(mark)
        found.append(f"volume off the step -> {codes or 'NO ERROR'} (expected TRADING_BAD_VOLUME)")
        await self._close_probe_positions()
        ok = "TRADING_BAD_VOLUME" in codes

        quote = await self.venue.quote(self.symbol)
        position = await self.venue.market_order(
            symbol=self.symbol,
            side=fx.LONG,
            units=self.units,
            stop_price=quote.ask - self.sl_pips * self.pip,
            target_price=None,
            client_order_id=self.label("i8b"),
        )
        try:
            mark = self.tap.mark()
            request = messages.ProtoOAAmendPositionSLTPReq(
                ctidTraderAccountId=self.client.account_id,
                positionId=int(position.id),
                stopLoss=quote.bid,
            )
            await self.client.submit(request, self.client.next_id())
            codes = await self._error_codes_after(mark)
            found.append(
                f"stop at the market -> {codes or 'NO ERROR'} "
                f"(expected PROTECTION_IS_TOO_CLOSE_TO_MARKET)"
            )
            ok = ok and "PROTECTION_IS_TOO_CLOSE_TO_MARKET" in codes
        finally:
            await self.venue.close(position.id)

        age = time.time() - (await self.venue.quote(self.symbol)).time
        found.append(
            "MARKET_CLOSED not testable: the market is open"
            if age < 300
            else "market looks closed: send an order to see MARKET_CLOSED"
        )
        return self.outcome("8", title, PASS if ok else OBSERVATION, "; ".join(found))

    async def _error_codes_after(self, mark: int, timeout: float = 4.0) -> list[str]:
        await self.tap.wait(mark, lambda m: error_code(m) is not None, timeout)
        return [c for _, m in self.tap.events[mark:] if (c := error_code(m))]

    async def _close_probe_positions(self) -> None:
        for position in await self.venue.positions():
            if position.client_order_id.startswith(
                f"probe-{self.token}-"
            ) and not position.client_order_id.endswith("-kill"):
                await self.venue.close(position.id)

    async def item_9(self) -> Outcome:
        title = "cost and slippage against the simulated"
        quote = await self.venue.quote(self.symbol)
        label = self.label("i9")
        mark = self.tap.mark()
        position = await self.venue.market_order(
            symbol=self.symbol,
            side=fx.LONG,
            units=self.units,
            stop_price=quote.ask - self.sl_pips * self.pip,
            target_price=None,
            client_order_id=label,
        )
        net = await self.venue.close(position.id)
        slippage = (position.entry_price - quote.ask) / self.pip
        spread = (quote.ask - quote.bid) / self.pip
        detail = (
            f"spread at send {spread:.2f} pips; entry slippage {slippage:+.2f} pips "
            f"(positive = worse); net result {net:+.4f}"
        )
        closing = [
            m.deal.closePositionDetail
            for m in self.tap.of_type(mark, messages.ProtoOAExecutionEvent)
            if m.HasField("deal") and m.deal.HasField("closePositionDetail")
        ]
        if closing:
            scale = 10 ** (closing[-1].moneyDigits if closing[-1].HasField("moneyDigits") else 2)
            detail += (
                f"; closing deal: gross {closing[-1].grossProfit / scale:+.4f}, "
                f"swap {closing[-1].swap / scale:+.4f}, "
                f"commission {closing[-1].commission / scale:+.4f} "
                f"(the sign convention is what item 9 settles)"
            )
        expected = 2 * self.units * 2.25 / 100_000
        detail += (
            f"; the simulator's round-trip commission for this size is about {expected:.4f} "
            f"in the base currency"
        )
        return self.outcome("9", title, OBSERVATION, detail)

    async def cleanup(self) -> None:
        """Close whatever this run left open, except the position item 4 keeps on purpose."""
        try:
            await self._close_probe_positions()
        except Exception as error:  # a failed cleanup must not hide the report
            emit(f"cleanup failed, check the demo account by hand: {type(error).__name__}")


ITEM_TITLES = {
    "pre": "account, symbol and volume rules",
    "A": "a wrong token is a fatal error",
    "1": "market order with relative stop and target",
    "2": "market order without stop, amended right after the fill",
    "3": "MARKET_RANGE with an absolute stop",
    "4": "kill -9 with a position open",
    "5": "reconnect and reconcile without a duplicate order",
    "6": "token refresh with atomic persistence",
    "7": "heartbeat, forced drop and resubscription of the spots",
    "8": "broker errors",
    "9": "cost and slippage against the simulated",
}


def emit(line: str = "") -> None:
    sys.stdout.write(line + "\n")


async def run_items(probe: Probe, items: Sequence[str]) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for item in items:
        run: Callable[[], Awaitable[Outcome]] = getattr(probe, f"item_{item}")
        try:
            outcome = await asyncio.wait_for(run(), ITEM_TIMEOUT_SECONDS)
        except Exception as error:  # one item failing must not stop the others
            detail = f"{type(error).__name__}: {error}"
            outcome = probe.outcome(item, ITEM_TITLES[item], FAIL, detail)
        emit(outcome.line())
        outcomes.append(outcome)
    return outcomes


def summary(outcomes: Sequence[Outcome]) -> str:
    counts = {
        status: sum(o.status == status for o in outcomes)
        for status in (PASS, FAIL, OBSERVATION, SKIPPED)
    }
    return "summary: " + ", ".join(f"{count} {status.lower()}" for status, count in counts.items())


async def run_probe(args: argparse.Namespace, credentials: Credentials) -> int:
    use_tls = args.host not in LOOPBACK_HOSTS

    def make_client(with_credentials: Credentials) -> CTraderClient:
        return CTraderClient(with_credentials, host=args.host, port=args.port, use_tls=use_tls)

    client = make_client(credentials)
    venue = CTraderVenue(client, symbols=[args.symbol])
    try:
        async with client:
            await venue.start()
            probe = Probe(
                client,
                venue,
                make_client,
                credentials,
                symbol=args.symbol,
                sl_pips=args.sl_pips,
                kill=args.kill,
                resume_label=args.label,
                idle_seconds=args.idle_seconds,
            )
            outcomes = await run_items(probe, args.items)
            await probe.cleanup()
            emit(summary(outcomes))
            if probe.kill_when_done:
                os.kill(os.getpid(), signal.SIGKILL)
    except Exception as error:  # startup problems are the report
        emit(f"[FAIL] pre: could not start: {client.redact(f'{type(error).__name__}: {error}')}")
        return 1
    return 1 if any(o.status == FAIL for o in outcomes) else 0


def parse_items(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [item for item in items if item not in ITEM_TITLES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown items: {', '.join(unknown)}")
    return items


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--items", type=parse_items, default=list(DEFAULT_ITEMS))
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--sl-pips", type=float, default=8.0)
    parser.add_argument("--kill", action="store_true", help="item 4: end the process with SIGKILL")
    parser.add_argument("--label", help="item 5 after a kill: the label item 4 printed")
    parser.add_argument("--idle-seconds", type=float, default=40.0)
    parser.add_argument(
        "--host", default=DEMO_HOST, help="the demo endpoint; anything else is refused"
    )
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    try:
        check_host(args.host)
        credentials = credentials_from_env(os.environ if env is None else env)
    except ProbeConfigError as error:
        emit(str(error))
        return 2
    return asyncio.run(run_probe(args, credentials))


if __name__ == "__main__":
    raise SystemExit(main())

"""A persistent cTrader CLI session as a broker venue, for the demo runner.

The CLI logs in once and then answers each command in tens of milliseconds, an order in about half a
second. It is driven through a pseudo-terminal because the CLI refuses piped input. This is a stop-gap
until the Open API application is approved: the same `Venue` interface, so the executor, the risk
guard and the bots do not change when the adapter in `ctrader/` replaces it.

Stop loss and take profit on a market order are converted by the CLI into pips relative to the fill
(an Open API limitation), so after the fill they are amended to the exact levels the executor asked for.
"""

from __future__ import annotations

import asyncio
import json
import os
import pty
import re
import select
import signal
import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from app.fx import strategy as fx
from app.fx.runner.venue import Account, OrderRejected, Position, Quote, VenueUnavailable

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PROMPT = re.compile(r">\s*$")


class Transport(Protocol):
    def send(self, command: str, timeout: float = 30.0) -> str: ...


class PtySession:
    """Runs `argv` (the CLI, already pointed at one account) in a pty and answers commands one at a time."""

    def __init__(self, argv: Sequence[str], login_timeout: float = 90.0) -> None:
        self._argv, self._login_timeout = list(argv), login_timeout
        self._lock = threading.Lock()
        self._pid = 0
        self._fd = -1

    def start(self) -> None:
        pid, fd = pty.fork()
        if pid == 0:  # child: become the CLI
            os.execvp(self._argv[0], self._argv)  # noqa: S606
        self._pid, self._fd = pid, fd
        self._read_until_prompt(self._login_timeout)

    def _read_until_prompt(self, timeout: float) -> str:
        buffer, deadline = b"", time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._fd], [], [], 0.05)
            if not ready:
                continue
            try:
                chunk = os.read(self._fd, 65536)
            except OSError as error:
                raise VenueUnavailable("the CLI session ended") from error
            if not chunk:
                raise VenueUnavailable("the CLI session ended")
            buffer += chunk
            text = ANSI.sub("", buffer.decode(errors="replace"))
            if PROMPT.search(text.rstrip("\r\n ")) or text.rstrip().endswith(">"):
                return text
        raise VenueUnavailable(f"the CLI did not answer within {timeout:.0f}s")

    def send(self, command: str, timeout: float = 30.0) -> str:
        with self._lock:
            try:
                os.write(self._fd, (command + "\r").encode())
            except OSError as error:
                raise VenueUnavailable("the CLI session is closed") from error
            return self._read_until_prompt(timeout)

    def close(self) -> None:
        try:
            os.write(self._fd, b"quit\r")
            time.sleep(0.5)
        except OSError:
            pass
        try:
            os.kill(self._pid, signal.SIGTERM)
            os.waitpid(self._pid, 0)
        except (OSError, ChildProcessError):
            pass


def json_of(text: str) -> dict:
    """The JSON object a CLI answer carries; raises OrderRejected with the CLI's own words if there is none."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise OrderRejected(" ".join(text.split())[-240:] or "empty answer")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise OrderRejected(" ".join(text.split())[-240:]) from error


class CliVenue:
    def __init__(self, transport: Transport, account_id: int, clock: Callable[[], float] = time.time) -> None:
        self._transport, self._account_id, self._clock = transport, account_id, clock
        self._client_ids: dict[str, str] = {}  # position id -> client order id
        self._sent: dict[str, str] = {}  # client order id -> position id

    async def _ask(self, command: str, timeout: float = 30.0) -> str:
        return await asyncio.to_thread(self._transport.send, command, timeout)

    async def account(self) -> Account:
        data = json_of(await self._ask(f"account {self._account_id}"))
        return Account(float(data["balance"]), float(data["equity"]), str(data.get("depositAsset", "USD")))

    def _position(self, raw: dict) -> Position:
        identifier = str(raw["id"])
        return Position(
            id=identifier, symbol=raw["symbolName"], side=fx.LONG if raw["tradeSide"] == "Buy" else fx.SHORT,
            units=int(raw["volume"]), entry_price=float(raw["entryPrice"]),
            stop_price=raw.get("stopLoss"), target_price=raw.get("takeProfit"),
            client_order_id=self._client_ids.get(identifier, ""),
        )

    async def positions(self) -> list[Position]:
        data = json_of(await self._ask("positions"))
        return [self._position(raw) for raw in data.get("positions", [])]

    async def quote(self, symbol: str) -> Quote:
        data = json_of(await self._ask(f"price {symbol}"))
        return Quote(symbol, float(data["bid"]), float(data["ask"]), self._clock())

    async def market_order(self, *, symbol: str, side: int, units: int, stop_price: float,
                           target_price: float | None, client_order_id: str) -> Position:
        if client_order_id in self._sent:  # a retry of an order already accepted
            for position in await self.positions():
                if position.id == self._sent[client_order_id]:
                    return position
        before = {p.id for p in await self.positions()}
        word = "buy" if side == fx.LONG else "sell"
        sl = f"{stop_price:.5f}"
        levels = sl if target_price is None else f"{sl} {target_price:.5f}"
        try:
            data = json_of(await self._ask(f"order place-market {symbol} {word} {units} {levels} yes", 60.0))
        except VenueUnavailable:
            # the answer was lost: the order may still have been accepted, so look before giving up
            fresh = [p for p in await self.positions() if p.id not in before and p.symbol == symbol and p.side == side]
            if not fresh:
                raise
            data = {"positionId": fresh[0].id}
        position_id = str(data["positionId"])
        self._client_ids[position_id] = client_order_id
        self._sent[client_order_id] = position_id
        for position in await self.positions():
            if position.id == position_id:
                return position
        raise OrderRejected(f"position {position_id} did not appear after the order")

    async def amend_protection(self, position_id: str, *, stop_price: float, target_price: float | None) -> Position:
        levels = f"{stop_price:.5f}" if target_price is None else f"{stop_price:.5f} {target_price:.5f}"
        json_of(await self._ask(f"position modify {position_id} {levels} yes"))
        for position in await self.positions():
            if position.id == position_id:
                return position
        raise OrderRejected(f"position {position_id} not found after the amendment")

    async def close(self, position_id: str) -> float:
        json_of(await self._ask(f"position close {position_id} yes", 60.0))
        deals = json_of(await self._ask("deals EURUSD 5")).get("deals", [])
        matching = [d for d in deals if str(d.get("positionId")) == position_id]
        return float(matching[-1]["netProfit"]) if matching else 0.0

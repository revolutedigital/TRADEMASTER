"""What the runner needs from a broker, and nothing more.

Every broker adapter (cTrader Open API, a fake for tests, later another venue) implements this
protocol, so risk, execution and reconciliation are written and tested once. The protective stop
lives on the broker's server: if the bot dies, the stop still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class VenueUnavailable(Exception):
    """The broker could not be reached; the request may or may not have arrived."""


class OrderRejected(Exception):
    """The broker refused the order."""


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    time: float


@dataclass(frozen=True)
class Position:
    id: str
    symbol: str
    side: int  # fx.LONG or fx.SHORT
    units: int
    entry_price: float
    stop_price: float | None
    target_price: float | None
    client_order_id: str


@dataclass(frozen=True)
class Exit:
    """How a position that is no longer open ended, as the broker recorded it."""

    result: float  # realised, in the account currency, commission and swap included
    price: float
    reason: str  # "target", "stop" or "market"
    time: float  # epoch seconds


@dataclass(frozen=True)
class Account:
    balance: float
    equity: float
    currency: str = "USD"


class Venue(Protocol):
    async def account(self) -> Account: ...

    async def positions(self) -> list[Position]: ...

    async def quote(self, symbol: str) -> Quote: ...

    async def market_order(
        self, *, symbol: str, side: int, units: int, stop_price: float,
        target_price: float | None, client_order_id: str,
    ) -> Position:
        """Open a position with its protective stop attached. Repeating a `client_order_id` returns
        the position that the first call opened instead of opening a second one."""

    async def amend_protection(
        self, position_id: str, *, stop_price: float, target_price: float | None
    ) -> Position: ...

    async def close(self, position_id: str) -> float:
        """Close at market and return the realized result in the account currency."""

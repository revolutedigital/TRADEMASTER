"""Turns a strategy's decision into a protected broker position, or refuses.

Sizing is by risk (never more than the operator allowed, and no trade if the smallest lot would
already risk more), the stop is attached on the broker's server at the moment of the order and is
checked on the way back, and a retry after an outage reuses the same client order id so it can
never open a second position.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates, Instrument, size_for_risk
from app.fx.runner.journal import Journal
from app.fx.runner.risk import RiskGuard
from app.fx.runner.venue import OrderRejected, Position, Venue, VenueUnavailable

RETRIES = 3
RETRY_PAUSE_SECONDS = 0.5


class ProtectionError(Exception):
    """The broker did not attach the protective stop; the position was closed and the bot stopped."""


class Executor:
    def __init__(self, venue: Venue, guard: RiskGuard, journal: Journal, rates: ConversionRates,
                 clock: Callable[[], float] = time.time) -> None:
        self.venue, self.guard, self.journal, self.rates, self.clock = venue, guard, journal, rates, clock

    async def _retry(self, call):
        for attempt in range(RETRIES):
            try:
                return await call()
            except VenueUnavailable:
                if attempt == RETRIES - 1:
                    raise
                await asyncio.sleep(RETRY_PAUSE_SECONDS)

    async def enter(self, *, symbol: str, side: int, stop_distance: float, target_distance: float | None,
                    client_order_id: str) -> Position | None:
        """Open a protected position, or return None (with the reason journaled) if any limit refuses."""
        if self.journal.has_order(client_order_id):
            return None
        instrument = Instrument.from_symbol(symbol)
        account = await self._retry(self.venue.account)
        self.guard.observe(self.clock(), account.equity)
        positions = await self._retry(self.venue.positions)
        quote = await self._retry(lambda: self.venue.quote(symbol))
        decision = self.guard.check_entry(
            open_positions=len(positions), spread_pips=(quote.ask - quote.bid) / instrument.pip_size
        )
        if not decision.allowed:
            return self._refuse(client_order_id, decision.reason)
        sizing = size_for_risk(
            instrument, equity=account.equity, risk_fraction=self.guard.limits.risk_fraction,
            stop_distance_pips=stop_distance / instrument.pip_size, rates=self.rates,
        )
        if not sizing.tradable:
            return self._refuse(client_order_id, sizing.reason)
        if sizing.risk_at_stop > self.guard.remaining_loss_budget(account.equity):
            return self._refuse(client_order_id, "the stop risk would exceed what is left of today's loss cap")
        entry = quote.ask if side == fx.LONG else quote.bid
        stop = entry - side * stop_distance
        target = None if target_distance is None else entry + side * target_distance
        self.journal.append("order_intent", client_order_id=client_order_id, symbol=symbol, side=side,
                            units=sizing.units, stop_price=stop, target_price=target)
        try:
            position = await self._retry(lambda: self.venue.market_order(
                symbol=symbol, side=side, units=sizing.units, stop_price=stop, target_price=target,
                client_order_id=client_order_id))
        except OrderRejected as error:
            return self._refuse(client_order_id, f"broker rejected: {error}")
        if position.stop_price is None:
            position = await self._retry(lambda: self.venue.amend_protection(
                position.id, stop_price=stop, target_price=target))
        if position.stop_price is None:
            await self._retry(lambda: self.venue.close(position.id))
            self.journal.append("closed", position_id=position.id, reason="no server-side stop")
            self.guard.kill("the broker did not attach the protective stop")
            raise ProtectionError(f"{symbol}: no protective stop on the broker, position closed")
        self.journal.append("order_filled", position_id=position.id, client_order_id=client_order_id,
                            symbol=symbol, side=side, units=position.units, entry_price=position.entry_price,
                            stop_price=position.stop_price, target_price=position.target_price)
        return position

    def _refuse(self, client_order_id: str, reason: str) -> None:
        self.journal.append("refused", client_order_id=client_order_id, reason=reason)
        return None

    async def close(self, position_id: str, reason: str) -> float:
        result = await self._retry(lambda: self.venue.close(position_id))
        self.journal.append("closed", position_id=position_id, reason=reason, result=result)
        account = await self._retry(self.venue.account)
        self.guard.observe(self.clock(), account.equity)
        return result

    async def flatten_all(self, reason: str) -> int:
        """Close every open position; the kill switch uses this."""
        positions = await self._retry(self.venue.positions)
        for position in positions:
            await self.close(position.id, reason)
        return len(positions)

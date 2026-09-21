"""An in-memory broker for tests and dry runs, with the failure modes the runner must survive."""

from __future__ import annotations

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates, Instrument, profit
from app.fx.runner.venue import Account, Exit, OrderRejected, Position, Quote, VenueUnavailable


class FakeVenue:
    def __init__(self, balance: float, rates: ConversionRates) -> None:
        self.balance = balance
        self.rates = rates
        self.quotes: dict[str, Quote] = {}
        self._positions: dict[str, Position] = {}
        self._by_client_id: dict[str, str] = {}
        self._exits: dict[str, Exit] = {}
        self._counter = 0
        self.orders_sent = 0
        self.unavailable_calls = 0  # how many upcoming calls raise VenueUnavailable
        self.reject_reason: str | None = None
        self.ignores_protection = False  # a broker that silently drops the stop, to test the check

    def set_quote(self, symbol: str, bid: float, ask: float, time: float = 0.0) -> None:
        """Move the market; a protective stop or target that the move crosses fills, as on a server."""
        self.quotes[symbol] = Quote(symbol, bid, ask, time)
        for position in list(self._positions.values()):
            if position.symbol != symbol:
                continue
            exit_price, reason = None, ""
            if position.side == fx.LONG:
                if position.stop_price is not None and bid <= position.stop_price:
                    exit_price, reason = bid, "stop"  # a gap through the stop fills at the gapped price
                elif position.target_price is not None and bid >= position.target_price:
                    exit_price, reason = position.target_price, "target"
            else:
                if position.stop_price is not None and ask >= position.stop_price:
                    exit_price, reason = ask, "stop"
                elif position.target_price is not None and ask <= position.target_price:
                    exit_price, reason = position.target_price, "target"
            if exit_price is not None:
                self._settle(position, exit_price, reason)

    def _settle(self, position: Position, exit_price: float, reason: str) -> float:
        result = profit(
            Instrument.from_symbol(position.symbol), side="LONG" if position.side == fx.LONG else "SHORT",
            units=position.units, entry_price=position.entry_price, exit_price=exit_price, rates=self.rates,
        )
        self.balance += result
        del self._positions[position.id]
        self._exits[position.id] = Exit(result, exit_price, reason, self.quotes[position.symbol].time)
        return result

    def _gate(self) -> None:
        if self.unavailable_calls > 0:
            self.unavailable_calls -= 1
            raise VenueUnavailable("simulated outage")

    def _floating(self) -> float:
        total = 0.0
        for p in self._positions.values():
            quote = self.quotes[p.symbol]
            mark = quote.bid if p.side == fx.LONG else quote.ask
            total += profit(
                Instrument.from_symbol(p.symbol), side="LONG" if p.side == fx.LONG else "SHORT",
                units=p.units, entry_price=p.entry_price, exit_price=mark, rates=self.rates,
            )
        return total

    async def account(self) -> Account:
        self._gate()
        return Account(self.balance, self.balance + self._floating())

    async def positions(self) -> list[Position]:
        self._gate()
        return list(self._positions.values())

    async def quote(self, symbol: str) -> Quote:
        self._gate()
        return self.quotes[symbol]

    async def market_order(self, *, symbol, side, units, stop_price, target_price, client_order_id) -> Position:
        self._gate()
        if client_order_id in self._by_client_id:
            return self._positions[self._by_client_id[client_order_id]]
        if self.reject_reason:
            raise OrderRejected(self.reject_reason)
        quote = self.quotes[symbol]
        self._counter += 1
        self.orders_sent += 1
        position = Position(
            id=f"P{self._counter}", symbol=symbol, side=side, units=units,
            entry_price=quote.ask if side == fx.LONG else quote.bid,
            stop_price=None if self.ignores_protection else stop_price,
            target_price=None if self.ignores_protection else target_price,
            client_order_id=client_order_id,
        )
        self._positions[position.id] = position
        self._by_client_id[client_order_id] = position.id
        return position

    async def amend_protection(self, position_id, *, stop_price, target_price) -> Position:
        self._gate()
        current = self._positions[position_id]
        if self.ignores_protection:
            return current
        amended = Position(**{**current.__dict__, "stop_price": stop_price, "target_price": target_price})
        self._positions[position_id] = amended
        return amended

    async def close(self, position_id: str) -> float:
        self._gate()
        position = self._positions[position_id]
        quote = self.quotes[position.symbol]
        return self._settle(position, quote.bid if position.side == fx.LONG else quote.ask, "market")

    async def exit_of(self, position_id: str, symbol: str) -> Exit | None:
        self._gate()
        return self._exits.get(position_id)

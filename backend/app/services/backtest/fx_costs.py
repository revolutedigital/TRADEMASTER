"""Cost primitives for FX backtests: spread, slippage, commission, pip value, P&L.

The Spot crypto engine charges a flat percentage per side. FX brokers charge in pips
(the bid/ask spread), sometimes plus a commission per lot or per traded notional, so a
flat 0.3% round trip would overstate the cost of a EURUSD trade by more than an order
of magnitude. Everything here is denominated in an account held in USD; crosses that
need a third currency conversion are rejected explicitly until the instrument and
conversion service exists.

This module has no exchange, database, or engine dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

Side = Literal["LONG", "SHORT"]
Action = Literal["ENTRY", "EXIT"]

STANDARD_LOT_UNITS = 100_000

# FX rolls value dates once a day at 17:00 New York, which is 21:00 or 22:00 UTC.
NEW_YORK = ZoneInfo("America/New_York")
ROLLOVER_TIME = time(17, 0)
WEDNESDAY = 2


class UnsupportedPairError(ValueError):
    """Raised for a pair whose pip value cannot be expressed in USD without a cross rate."""


def _split_pair(symbol: str) -> tuple[str, str]:
    if len(symbol) != 6 or not symbol.isalpha() or not symbol.isupper():
        raise UnsupportedPairError(f"{symbol!r} is not a six-letter currency pair")
    return symbol[:3], symbol[3:]


def pip_size(symbol: str) -> float:
    """One pip in price terms: 0.01 for yen quotes, 0.0001 otherwise."""
    _, quote = _split_pair(symbol)
    return 0.01 if quote == "JPY" else 0.0001


def pip_value_usd(symbol: str, price: float, units: float = STANDARD_LOT_UNITS) -> float:
    """USD value of one pip for a position of `units` of the base currency."""
    _validate_positive(price=price, units=units)
    base, quote = _split_pair(symbol)
    one_pip = pip_size(symbol)
    if quote == "USD":
        return units * one_pip
    if base == "USD":
        return units * one_pip / price
    raise UnsupportedPairError(f"{symbol} needs a cross-rate conversion to USD")


def notional_usd(symbol: str, price: float, units: float) -> float:
    """Traded value in USD, the base for percentage-of-notional commissions."""
    _validate_positive(price=price, units=units)
    base, quote = _split_pair(symbol)
    if quote == "USD":
        return units * price
    if base == "USD":
        return units
    raise UnsupportedPairError(f"{symbol} needs a cross-rate conversion to USD")


def _validate_positive(**values: float) -> None:
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}")


class CommissionSchedule(Protocol):
    def per_side_usd(self, *, units: float, notional_usd: float) -> float: ...


@dataclass(frozen=True)
class NoCommission:
    """Spread-only accounts."""

    def per_side_usd(self, *, units: float, notional_usd: float) -> float:
        return 0.0


@dataclass(frozen=True)
class PerLotCommission:
    """A fixed USD amount per standard lot on each side (raw-spread ECN accounts)."""

    usd_per_lot_per_side: float

    def __post_init__(self) -> None:
        if self.usd_per_lot_per_side < 0:
            raise ValueError("commission cannot be negative")

    def per_side_usd(self, *, units: float, notional_usd: float) -> float:
        return self.usd_per_lot_per_side * units / STANDARD_LOT_UNITS


@dataclass(frozen=True)
class NotionalCommission:
    """A fraction of traded notional with a per-order minimum (for example IBKR Pro)."""

    basis_points: float
    minimum_usd: float

    def __post_init__(self) -> None:
        if self.basis_points < 0 or self.minimum_usd < 0:
            raise ValueError("commission cannot be negative")

    def per_side_usd(self, *, units: float, notional_usd: float) -> float:
        return max(self.minimum_usd, notional_usd * self.basis_points / 10_000)


@dataclass(frozen=True)
class FxCostModel:
    """Everything a fill costs, apart from financing (SwapSchedule) and gaps (fx_gap).

    `spread_multiplier` widens the quoted spread symmetrically around the mid for
    stress tests (rollover hour, news); 1.0 uses the quote as observed.
    """

    commission: CommissionSchedule
    slippage_pips: float = 0.0
    spread_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.slippage_pips < 0:
            raise ValueError("slippage cannot be negative")
        if self.spread_multiplier < 1.0:
            raise ValueError("spread_multiplier below 1.0 would model a tighter-than-quoted spread")

    def fill_price(
        self, *, symbol: str, side: Side, action: Action, bid: float, ask: float
    ) -> float:
        """Price actually paid or received. The trader always crosses the spread."""
        if not (bid > 0 and ask > 0):
            raise ValueError("bid and ask must be positive")
        if ask < bid:
            raise ValueError(f"crossed quote: ask {ask} is below bid {bid}")

        mid = (bid + ask) / 2
        half_spread = (ask - bid) / 2 * self.spread_multiplier
        slippage = self.slippage_pips * pip_size(symbol)
        buys = (side == "LONG") == (action == "ENTRY")
        if buys:
            return mid + half_spread + slippage
        return mid - half_spread - slippage

    def commission_usd(self, *, symbol: str, price: float, units: float) -> float:
        """Commission for one side of a trade."""
        return self.commission.per_side_usd(
            units=units, notional_usd=notional_usd(symbol, price, units)
        )


def trade_pnl_usd(
    *, symbol: str, side: Side, units: float, entry_price: float, exit_price: float
) -> float:
    """Gross P&L in USD, converting a non-USD quote currency at the exit price."""
    _validate_positive(units=units, entry_price=entry_price, exit_price=exit_price)
    direction = 1.0 if side == "LONG" else -1.0
    pnl_in_quote_currency = direction * (exit_price - entry_price) * units
    base, quote = _split_pair(symbol)
    if quote == "USD":
        return pnl_in_quote_currency
    if base == "USD":
        return pnl_in_quote_currency / exit_price
    raise UnsupportedPairError(f"{symbol} needs a cross-rate conversion to USD")


@dataclass(frozen=True)
class RoundTripCost:
    """Cost of one completed trade, in the units a trader reasons about."""

    spread_and_slippage_pips: float
    commission_usd: float
    total_usd: float


def round_trip_cost(
    model: FxCostModel,
    *,
    symbol: str,
    side: Side,
    units: float,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
) -> RoundTripCost:
    """Total cost of entering and exiting one trade, versus trading at the mid both times."""
    entry = model.fill_price(symbol=symbol, side=side, action="ENTRY", bid=entry_bid, ask=entry_ask)
    exit_ = model.fill_price(symbol=symbol, side=side, action="EXIT", bid=exit_bid, ask=exit_ask)
    entry_mid = (entry_bid + entry_ask) / 2
    exit_mid = (exit_bid + exit_ask) / 2

    gross_at_mid = trade_pnl_usd(
        symbol=symbol, side=side, units=units, entry_price=entry_mid, exit_price=exit_mid
    )
    gross_filled = trade_pnl_usd(
        symbol=symbol, side=side, units=units, entry_price=entry, exit_price=exit_
    )
    price_cost_usd = gross_at_mid - gross_filled
    commission = model.commission_usd(symbol=symbol, price=entry, units=units) + (
        model.commission_usd(symbol=symbol, price=exit_, units=units)
    )
    pip_usd = pip_value_usd(symbol, exit_mid, units)
    return RoundTripCost(
        spread_and_slippage_pips=price_cost_usd / pip_usd,
        commission_usd=commission,
        total_usd=price_cost_usd + commission,
    )


@dataclass(frozen=True)
class SwapSchedule:
    """Daily financing in pips per position, as brokers publish it (credit is positive).

    The value date advances one business day per rollover, so the Wednesday rollover
    settles Saturday and Sunday too and is charged three times. Some brokers move the
    triple charge for T+1 pairs such as USDCAD; set `triple_weekday` from the broker's
    contract specification (Monday is 0).
    """

    long_pips_per_day: float
    short_pips_per_day: float
    triple_weekday: int = WEDNESDAY

    def __post_init__(self) -> None:
        if not 0 <= self.triple_weekday <= 4:
            raise ValueError("triple_weekday must be a weekday between 0 and 4")


def _require_aware(name: str, moment: datetime) -> None:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def rollover_days_charged(
    entry_time: datetime, exit_time: datetime, *, triple_weekday: int = WEDNESDAY
) -> int:
    """Number of daily financing charges a position pays between two instants.

    A charge happens when the 17:00 New York rollover falls after the entry and at or
    before the exit. There is no rollover on Saturday or Sunday.
    """
    _require_aware("entry_time", entry_time)
    _require_aware("exit_time", exit_time)
    if exit_time < entry_time:
        raise ValueError("exit_time must not be before entry_time")

    first_day: date = entry_time.astimezone(NEW_YORK).date() - timedelta(days=1)
    last_day: date = exit_time.astimezone(NEW_YORK).date() + timedelta(days=1)
    charged = 0
    day = first_day
    while day <= last_day:
        if day.weekday() < 5:
            rollover = datetime.combine(day, ROLLOVER_TIME, tzinfo=NEW_YORK).astimezone(UTC)
            if entry_time < rollover <= exit_time:
                charged += 3 if day.weekday() == triple_weekday else 1
        day += timedelta(days=1)
    return charged


def swap_pnl_usd(
    schedule: SwapSchedule,
    *,
    symbol: str,
    side: Side,
    units: float,
    price: float,
    entry_time: datetime,
    exit_time: datetime,
) -> float:
    """Financing credited (positive) or debited (negative) for holding a position."""
    days = rollover_days_charged(entry_time, exit_time, triple_weekday=schedule.triple_weekday)
    pips_per_day = schedule.long_pips_per_day if side == "LONG" else schedule.short_pips_per_day
    return days * pips_per_day * pip_value_usd(symbol, price, units)

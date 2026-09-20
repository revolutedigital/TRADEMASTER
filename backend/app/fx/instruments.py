"""Forex instruments: pip size, lot arithmetic, and profit in the account currency.

A pip is worth a different amount of money depending on the pair and on the account currency,
and the cross pairs (EURJPY, GBPJPY, EURGBP) need a third currency's rate to be valued at all.
This module does that conversion explicitly and fails closed: when a rate is missing it raises,
so a position is never sized or valued with a guessed number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Side = Literal["LONG", "SHORT"]

STANDARD_LOT_UNITS = 100_000


class UnknownInstrumentError(ValueError):
    """The symbol is not a six-letter currency pair."""


class MissingRateError(LookupError):
    """A conversion rate needed to value money in the account currency is not available."""


@dataclass(frozen=True)
class Instrument:
    """Static facts about a currency pair as quoted by the broker."""

    symbol: str
    base: str
    quote: str
    pip_size: float
    price_decimals: int
    min_lot: float = 0.01
    lot_step: float = 0.01

    @classmethod
    def from_symbol(cls, symbol: str) -> Instrument:
        if len(symbol) != 6 or not symbol.isalpha() or not symbol.isupper():
            raise UnknownInstrumentError(f"{symbol!r} is not a six-letter currency pair")
        base, quote = symbol[:3], symbol[3:]
        yen = quote == "JPY"
        return cls(
            symbol=symbol,
            base=base,
            quote=quote,
            pip_size=0.01 if yen else 0.0001,
            price_decimals=3 if yen else 5,
        )

    @property
    def min_units(self) -> int:
        return round(self.min_lot * STANDARD_LOT_UNITS)


@dataclass(frozen=True)
class ConversionRates:
    """Mid prices of the pairs quoted against the US dollar, used to convert between currencies."""

    mids: Mapping[str, float]

    def usd_per(self, currency: str) -> float:
        """How many US dollars one unit of `currency` is worth."""
        if currency == "USD":
            return 1.0
        direct = self.mids.get(f"{currency}USD")
        if direct is not None:
            return _positive(direct, f"{currency}USD")
        inverse = self.mids.get(f"USD{currency}")
        if inverse is not None:
            return 1.0 / _positive(inverse, f"USD{currency}")
        raise MissingRateError(f"no rate to convert {currency} to USD")

    def convert(self, amount: float, *, source: str, target: str) -> float:
        """Convert money between two currencies through the US dollar."""
        if source == target:
            return amount
        return amount * self.usd_per(source) / self.usd_per(target)


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"rate {name} must be a positive number, got {value!r}")
    return value


def pip_value(
    instrument: Instrument, units: float, rates: ConversionRates, account_currency: str = "USD"
) -> float:
    """Money made or lost, in the account currency, when the price moves one pip."""
    if units <= 0:
        raise ValueError("units must be positive")
    in_quote_currency = units * instrument.pip_size
    return rates.convert(in_quote_currency, source=instrument.quote, target=account_currency)


def units_from_lots(lots: float) -> int:
    return round(lots * STANDARD_LOT_UNITS)


def lots_from_units(units: float) -> float:
    return units / STANDARD_LOT_UNITS


def floor_to_lot_step(units: float, instrument: Instrument) -> int:
    """Round a size down to the broker's lot step; sizes never round up into extra risk."""
    if units <= 0:
        return 0
    step_units = round(instrument.lot_step * STANDARD_LOT_UNITS)
    return int(units // step_units) * step_units


@dataclass(frozen=True)
class SizingResult:
    units: int
    risk_at_stop: float
    risk_budget: float
    reason: str

    @property
    def tradable(self) -> bool:
        return self.units > 0


def size_for_risk(
    instrument: Instrument,
    *,
    equity: float,
    risk_fraction: float,
    stop_distance_pips: float,
    rates: ConversionRates,
    account_currency: str = "USD",
) -> SizingResult:
    """Largest size whose loss at the stop stays within the risk budget, or none.

    If even the smallest lot the broker accepts would lose more than the budget at this stop,
    the trade is refused instead of being taken at a higher risk than the operator allowed.
    """
    if equity <= 0 or not 0 < risk_fraction <= 1 or stop_distance_pips <= 0:
        raise ValueError("equity, risk_fraction and stop_distance_pips must be positive")
    budget = equity * risk_fraction
    per_unit = pip_value(instrument, 1, rates, account_currency) * stop_distance_pips
    units = floor_to_lot_step(budget / per_unit, instrument)
    if units < instrument.min_units:
        smallest_risk = per_unit * instrument.min_units
        return SizingResult(
            0,
            0.0,
            budget,
            f"the smallest lot risks {smallest_risk:.2f} at this stop, above the {budget:.2f} budget",
        )
    return SizingResult(units, per_unit * units, budget, "ok")


def max_stop_pips_for_min_lot(
    instrument: Instrument,
    *,
    equity: float,
    risk_fraction: float,
    rates: ConversionRates,
    account_currency: str = "USD",
) -> float:
    """The widest stop, in pips, that the smallest lot can carry inside the risk budget."""
    per_pip = pip_value(instrument, instrument.min_units, rates, account_currency)
    return equity * risk_fraction / per_pip


def profit(
    instrument: Instrument,
    *,
    side: Side,
    units: float,
    entry_price: float,
    exit_price: float,
    rates: ConversionRates,
    account_currency: str = "USD",
) -> float:
    """Profit or loss in the account currency, valuing the quote currency at the exit."""
    if units <= 0 or entry_price <= 0 or exit_price <= 0:
        raise ValueError("units and prices must be positive")
    direction = 1.0 if side == "LONG" else -1.0
    in_quote_currency = direction * (exit_price - entry_price) * units
    return rates.convert(in_quote_currency, source=instrument.quote, target=account_currency)


def notional(
    instrument: Instrument,
    *,
    units: float,
    price: float,
    rates: ConversionRates,
    account_currency: str = "USD",
) -> float:
    """Traded value in the account currency (the base currency amount at today's rate)."""
    if units <= 0 or price <= 0:
        raise ValueError("units and price must be positive")
    in_quote_currency = units * price
    return rates.convert(in_quote_currency, source=instrument.quote, target=account_currency)


def margin_required(
    instrument: Instrument,
    *,
    units: float,
    price: float,
    leverage: float,
    rates: ConversionRates,
    account_currency: str = "USD",
) -> float:
    """Margin the broker blocks for a position at a given leverage."""
    if leverage < 1:
        raise ValueError("leverage must be at least 1")
    return notional(
        instrument, units=units, price=price, rates=rates, account_currency=account_currency
    ) / leverage

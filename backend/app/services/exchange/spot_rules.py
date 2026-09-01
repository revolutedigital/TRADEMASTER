"""Binance Spot symbol-filter parsing and deterministic order validation."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN
from math import gcd
from typing import Any


class SpotRuleViolation(ValueError):
    """Raised when an order conflicts with the exchange's current symbol rules."""


@dataclass(frozen=True)
class SpotSymbolRules:
    """The executable subset of Binance Spot ``exchangeInfo`` symbol filters."""

    symbol: str
    status: str
    quantity_step: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal
    oco_quantity_step: Decimal
    oco_minimum_quantity: Decimal
    oco_maximum_quantity: Decimal
    price_tick: Decimal
    minimum_notional: Decimal | None
    maximum_notional: Decimal | None

    @classmethod
    def from_exchange_info(cls, exchange_info: dict[str, Any]) -> "SpotSymbolRules":
        """Parse the filters needed for a Spot market entry and protective OCO."""
        symbol = str(exchange_info.get("symbol", ""))
        status = str(exchange_info.get("status", ""))
        filters = {
            str(item.get("filterType")): item
            for item in exchange_info.get("filters", [])
            if isinstance(item, dict)
        }
        market_lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        limit_lot = filters.get("LOT_SIZE") or market_lot
        price = filters.get("PRICE_FILTER")
        if not symbol or not market_lot or not limit_lot or not price:
            raise SpotRuleViolation("exchangeInfo is missing required Spot symbol filters")

        quantity_step = _positive_decimal(market_lot, "stepSize")
        minimum_quantity = _positive_decimal(market_lot, "minQty", allow_zero=True)
        maximum_quantity = _positive_decimal(market_lot, "maxQty")
        limit_quantity_step = _positive_decimal(limit_lot, "stepSize")
        limit_minimum_quantity = _positive_decimal(limit_lot, "minQty", allow_zero=True)
        limit_maximum_quantity = _positive_decimal(limit_lot, "maxQty")
        price_tick = _positive_decimal(price, "tickSize")

        notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
        minimum_notional: Decimal | None = None
        maximum_notional: Decimal | None = None
        if notional_filter:
            if "minNotional" in notional_filter:
                minimum_notional = _positive_decimal(
                    notional_filter, "minNotional", allow_zero=True
                )
            if "maxNotional" in notional_filter:
                maximum_notional = _positive_decimal(
                    notional_filter, "maxNotional", allow_zero=True
                )

        return cls(
            symbol=symbol,
            status=status,
            quantity_step=quantity_step,
            minimum_quantity=minimum_quantity,
            maximum_quantity=maximum_quantity,
            oco_quantity_step=_decimal_lcm(quantity_step, limit_quantity_step),
            oco_minimum_quantity=max(minimum_quantity, limit_minimum_quantity),
            oco_maximum_quantity=min(maximum_quantity, limit_maximum_quantity),
            price_tick=price_tick,
            minimum_notional=minimum_notional,
            maximum_notional=maximum_notional,
        )

    def normalize_market_quantity(self, requested_quantity: Decimal) -> Decimal:
        """Round a requested quantity down to the current market lot step."""
        self.require_tradable()
        return self._normalize_quantity(
            requested_quantity,
            step=self.quantity_step,
            minimum=self.minimum_quantity,
            maximum=self.maximum_quantity,
            label="market",
        )

    def normalize_oco_quantity(self, requested_quantity: Decimal) -> Decimal:
        """Return a quantity valid for both OCO legs and their Spot filters."""
        self.require_tradable()
        if self.oco_minimum_quantity > self.oco_maximum_quantity:
            raise SpotRuleViolation(
                f"{self.symbol} has no compatible MARKET_LOT_SIZE and LOT_SIZE range for OCO"
            )
        return self._normalize_quantity(
            requested_quantity,
            step=self.oco_quantity_step,
            minimum=self.oco_minimum_quantity,
            maximum=self.oco_maximum_quantity,
            label="OCO",
        )

    def minimum_market_quantity_for_notional(self, price: Decimal) -> Decimal:
        """Find the smallest valid market quantity satisfying current min-notional rules."""
        self.require_tradable()
        if price <= 0:
            raise SpotRuleViolation("market price must be positive")
        requested = self.minimum_quantity
        if self.minimum_notional is not None:
            requested = max(requested, self.minimum_notional / price)
        units = (requested / self.quantity_step).to_integral_value(rounding=ROUND_CEILING)
        return self._normalize_quantity(
            units * self.quantity_step,
            step=self.quantity_step,
            minimum=self.minimum_quantity,
            maximum=self.maximum_quantity,
            label="market",
        )

    def _normalize_quantity(
        self,
        requested_quantity: Decimal,
        *,
        step: Decimal,
        minimum: Decimal,
        maximum: Decimal,
        label: str,
    ) -> Decimal:
        if requested_quantity <= 0:
            raise SpotRuleViolation("requested quantity must be positive")
        normalized = requested_quantity.quantize(step, rounding=ROUND_DOWN)
        # Decimal.quantize follows exponent but does not ensure a multiple for
        # uncommon steps (for example 0.005), so use integer step arithmetic.
        normalized = (normalized // step) * step
        if normalized < minimum:
            raise SpotRuleViolation(
                f"{label} quantity {normalized} is below {self.symbol} minimum {minimum}"
            )
        if normalized > maximum:
            raise SpotRuleViolation(
                f"{label} quantity {normalized} is above {self.symbol} maximum {maximum}"
            )
        return normalized

    def normalize_price_down(self, requested_price: Decimal) -> Decimal:
        """Round a sell limit or stop trigger down to its valid price tick."""
        if requested_price <= 0:
            raise SpotRuleViolation("requested price must be positive")
        return (requested_price // self.price_tick) * self.price_tick

    def require_tradable(self) -> None:
        """Reject symbols that Binance has halted or limited to cancellation."""
        if self.status != "TRADING":
            raise SpotRuleViolation(
                f"{self.symbol} is not available for trading (status={self.status or 'unknown'})"
            )

    def validate_notional(self, price: Decimal, quantity: Decimal) -> Decimal:
        """Check exchange notional bounds and return the exact notional value."""
        notional = price * quantity
        if self.minimum_notional is not None and notional < self.minimum_notional:
            raise SpotRuleViolation(
                f"notional {notional} is below {self.symbol} minimum {self.minimum_notional}"
            )
        if self.maximum_notional is not None and notional > self.maximum_notional:
            raise SpotRuleViolation(
                f"notional {notional} exceeds {self.symbol} maximum {self.maximum_notional}"
            )
        return notional

    def prepare_long_exit_oco(
        self,
        *,
        last_price: Decimal,
        quantity: Decimal,
        take_profit_price: Decimal,
        stop_loss_price: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Validate an OCO SELL pair that protects an already-filled long entry."""
        normalized_quantity = self.normalize_oco_quantity(quantity)
        take_profit = self.normalize_price_down(take_profit_price)
        stop_loss = self.normalize_price_down(stop_loss_price)
        if not take_profit > last_price > stop_loss:
            raise SpotRuleViolation(
                "Spot OCO SELL requires take profit above and stop loss below last price"
            )
        self.validate_notional(take_profit, normalized_quantity)
        self.validate_notional(stop_loss, normalized_quantity)
        return normalized_quantity, take_profit, stop_loss


def _positive_decimal(
    values: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> Decimal:
    try:
        value = Decimal(str(values[key]))
    except (KeyError, ValueError) as exc:
        raise SpotRuleViolation(f"exchangeInfo filter is missing a valid {key}") from exc
    if value < 0 or (value == 0 and not allow_zero):
        raise SpotRuleViolation(f"exchangeInfo filter has an invalid {key}")
    return value


def _decimal_lcm(left: Decimal, right: Decimal) -> Decimal:
    """Find the smallest decimal increment that is valid for both step sizes."""
    places = max(-left.as_tuple().exponent, -right.as_tuple().exponent)
    scale = 10**places
    left_units = int(left * scale)
    right_units = int(right * scale)
    common_units = left_units * right_units // gcd(left_units, right_units)
    return Decimal(common_units) / Decimal(scale)

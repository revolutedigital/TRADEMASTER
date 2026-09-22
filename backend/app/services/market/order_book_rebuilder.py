"""Strict Binance USD-M depth snapshot and delta reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


class OrderBookGapError(ValueError):
    """Raised when depth continuity is lost and a new snapshot is required."""


@dataclass(frozen=True)
class OrderBookTop:
    update_id: int
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal


class OrderBookRebuilder:
    """Apply futures diff-depth events without guessing across sequence gaps."""

    def __init__(self) -> None:
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._last_update_id: int | None = None
        self._awaiting_first_event = True

    @property
    def initialized(self) -> bool:
        return self._last_update_id is not None

    @property
    def last_update_id(self) -> int | None:
        return self._last_update_id

    def initialize(self, snapshot: dict[str, Any]) -> None:
        try:
            update_id = int(snapshot["lastUpdateId"])
            bids = self._parse_levels(snapshot["bids"])
            asks = self._parse_levels(snapshot["asks"])
        except (KeyError, TypeError, ValueError, InvalidOperation) as error:
            raise OrderBookGapError("Invalid depth snapshot") from error
        if not bids or not asks:
            raise OrderBookGapError("Depth snapshot has an empty side")
        self._bids = bids
        self._asks = asks
        self._last_update_id = update_id
        self._awaiting_first_event = True
        self._validate_book()

    def apply(self, event: dict[str, Any]) -> OrderBookTop | None:
        if self._last_update_id is None:
            raise OrderBookGapError("Depth snapshot must be initialized first")
        try:
            first_update_id = int(event["U"])
            final_update_id = int(event["u"])
            previous_final_update_id = int(event["pu"])
        except (KeyError, TypeError, ValueError) as error:
            raise OrderBookGapError("Invalid depth update identifiers") from error

        if final_update_id < self._last_update_id:
            return None
        if self._awaiting_first_event:
            if not (first_update_id <= self._last_update_id <= final_update_id):
                raise OrderBookGapError("First depth update does not bridge the REST snapshot")
            self._awaiting_first_event = False
        elif previous_final_update_id != self._last_update_id:
            raise OrderBookGapError(
                f"Depth sequence gap: pu={previous_final_update_id}, "
                f"expected={self._last_update_id}"
            )

        self._apply_levels(self._bids, event.get("b", []))
        self._apply_levels(self._asks, event.get("a", []))
        self._last_update_id = final_update_id
        self._validate_book()
        return self.top()

    def top(self) -> OrderBookTop:
        if self._last_update_id is None or not self._bids or not self._asks:
            raise OrderBookGapError("Order book is not ready")
        bid_price = max(self._bids)
        ask_price = min(self._asks)
        return OrderBookTop(
            update_id=self._last_update_id,
            bid_price=bid_price,
            bid_quantity=self._bids[bid_price],
            ask_price=ask_price,
            ask_quantity=self._asks[ask_price],
        )

    @staticmethod
    def _parse_levels(levels: list[list[str]]) -> dict[Decimal, Decimal]:
        parsed: dict[Decimal, Decimal] = {}
        for raw_price, raw_quantity in levels:
            price = Decimal(raw_price)
            quantity = Decimal(raw_quantity)
            if price <= 0 or quantity < 0:
                raise OrderBookGapError("Depth levels require positive prices and quantities")
            if quantity:
                parsed[price] = quantity
        return parsed

    @staticmethod
    def _apply_levels(book_side: dict[Decimal, Decimal], levels: list[list[str]]) -> None:
        for raw_price, raw_quantity in levels:
            try:
                price = Decimal(raw_price)
                quantity = Decimal(raw_quantity)
            except InvalidOperation as error:
                raise OrderBookGapError("Invalid decimal depth level") from error
            if price <= 0 or quantity < 0:
                raise OrderBookGapError("Depth levels require positive prices and quantities")
            if quantity == 0:
                book_side.pop(price, None)
            else:
                book_side[price] = quantity

    def _validate_book(self) -> None:
        if not self._bids or not self._asks:
            raise OrderBookGapError("Order book side became empty")
        if max(self._bids) >= min(self._asks):
            raise OrderBookGapError("Order book is crossed or locked")

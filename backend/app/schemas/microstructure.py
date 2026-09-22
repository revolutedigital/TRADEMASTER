"""Canonical contracts for event-level market data and dataset lineage."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MarketEventType(StrEnum):
    AGG_TRADE = "AGG_TRADE"
    TRADE = "TRADE"
    BOOK_TICKER = "BOOK_TICKER"
    DEPTH = "DEPTH"
    MARK_PRICE = "MARK_PRICE"
    FUNDING = "FUNDING"
    LIQUIDATION = "LIQUIDATION"


class MicrostructureEvent(BaseModel):
    """One canonical event consumed by recorders, replay, and online features."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    venue: str = "binance"
    product: str
    symbol: str
    event_type: MarketEventType
    event_time: datetime
    receive_time: datetime | None = None
    sequence_id: int | None = None
    first_sequence_id: int | None = None
    last_sequence_id: int | None = None
    price: float | None = Field(default=None, gt=0)
    quantity: float | None = Field(default=None, ge=0)
    quote_quantity: float | None = Field(default=None, ge=0)
    is_buyer_maker: bool | None = None
    side: str | None = None
    bid_price: float | None = Field(default=None, gt=0)
    bid_quantity: float | None = Field(default=None, ge=0)
    ask_price: float | None = Field(default=None, gt=0)
    ask_quantity: float | None = Field(default=None, ge=0)
    payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_quote(self) -> "MicrostructureEvent":
        if self.bid_price is not None and self.ask_price is not None:
            if self.ask_price < self.bid_price:
                raise ValueError("ask_price cannot be below bid_price")
        return self


class DatasetPartitionManifest(BaseModel):
    """Integrity and lineage facts for one normalized daily partition."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    venue: str
    product: str
    symbol: str
    event_type: MarketEventType
    utc_date: str
    source_url: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    normalized_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    row_count: int = Field(ge=0)
    first_event_time: datetime | None
    last_event_time: datetime | None
    first_sequence_id: int | None
    last_sequence_id: int | None
    sequence_gap_count: int = Field(ge=0)
    duplicate_sequence_count: int = Field(ge=0)
    quality_status: str

"""Market data Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel


class OHLCVResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_volume: float
    trade_count: int

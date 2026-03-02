"""gRPC service definitions for internal communication.

Provides protobuf-like message definitions and service stubs.
Production would use grpcio with .proto files compiled via protoc.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator

from app.core.logging import get_logger

logger = get_logger(__name__)


# Proto-like message definitions
@dataclass
class MarketDataRequest:
    symbol: str
    interval: str = "1h"
    limit: int = 100


@dataclass
class MarketDataResponse:
    symbol: str
    candles: list[dict] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: str  # String for Decimal precision
    order_type: str = "MARKET"
    price: str | None = None
    client_order_id: str = ""


@dataclass
class OrderResponse:
    order_id: str
    status: str
    fill_price: str = "0"
    filled_quantity: str = "0"
    message: str = ""


@dataclass
class RiskCheckRequest:
    symbol: str
    side: str
    quantity: str
    portfolio_value: str


@dataclass
class RiskCheckResponse:
    approved: bool
    reason: str = ""
    max_allowed_quantity: str = "0"
    risk_score: float = 0.0


class MarketDataService:
    """gRPC-style service for market data streaming."""

    async def get_candles(self, request: MarketDataRequest) -> MarketDataResponse:
        """Unary RPC: Get historical candles."""
        logger.debug("grpc_get_candles", symbol=request.symbol, limit=request.limit)
        return MarketDataResponse(symbol=request.symbol, candles=[])

    async def stream_prices(self, symbols: list[str]) -> AsyncIterator[dict]:
        """Server streaming RPC: Stream live prices."""
        import asyncio
        while True:
            for symbol in symbols:
                yield {"symbol": symbol, "price": "0", "timestamp": datetime.now(timezone.utc).isoformat()}
            await asyncio.sleep(1)


class OrderService:
    """gRPC-style service for order management."""

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Unary RPC: Place a new order."""
        logger.info("grpc_place_order", symbol=request.symbol, side=request.side)
        return OrderResponse(order_id=request.client_order_id or "generated", status="ACCEPTED")

    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Unary RPC: Cancel an order."""
        return OrderResponse(order_id=order_id, status="CANCELLED")


class RiskService:
    """gRPC-style service for pre-trade risk checks."""

    async def check_order(self, request: RiskCheckRequest) -> RiskCheckResponse:
        """Unary RPC: Pre-trade risk check."""
        logger.debug("grpc_risk_check", symbol=request.symbol, qty=request.quantity)
        return RiskCheckResponse(approved=True, max_allowed_quantity=request.quantity, risk_score=0.3)


# Service instances
market_data_service = MarketDataService()
order_service = OrderService()
risk_service = RiskService()

"""Time-Weighted Average Price (TWAP) execution algorithm."""
import asyncio
from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger
from app.services.exchange.binance_client import binance_client

logger = get_logger(__name__)


@dataclass
class TWAPResult:
    symbol: str
    side: str
    total_quantity: float
    slices_executed: int
    slices_total: int
    avg_fill_price: float
    total_commission: float


class TWAPExecutor:
    """Execute large orders gradually over time using TWAP."""

    async def execute(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        duration_minutes: int = 10,
        num_slices: int = 5,
    ) -> TWAPResult:
        slice_qty = total_qty / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        fills = []

        logger.info(
            "twap_started",
            symbol=symbol,
            side=side,
            total_qty=total_qty,
            slices=num_slices,
            interval_s=interval_seconds,
        )

        for i in range(num_slices):
            try:
                result = await binance_client.place_market_order(symbol, side, slice_qty)
                fill_price = float(result.get("avgPrice", 0)) or float(result.get("price", 0))
                commission = sum(float(f.get("commission", 0)) for f in result.get("fills", []))
                fills.append({"price": fill_price, "qty": slice_qty, "commission": commission})
                logger.info("twap_slice", slice=i + 1, total=num_slices, price=fill_price)
            except Exception as e:
                logger.error("twap_slice_failed", slice=i + 1, error=str(e))

            if i < num_slices - 1:
                await asyncio.sleep(interval_seconds)

        total_filled = sum(f["qty"] for f in fills)
        avg_price = sum(f["price"] * f["qty"] for f in fills) / total_filled if total_filled > 0 else 0
        total_comm = sum(f["commission"] for f in fills)

        logger.info("twap_complete", symbol=symbol, avg_price=round(avg_price, 2), filled=total_filled)

        return TWAPResult(
            symbol=symbol,
            side=side,
            total_quantity=total_filled,
            slices_executed=len(fills),
            slices_total=num_slices,
            avg_fill_price=round(avg_price, 2),
            total_commission=round(total_comm, 4),
        )


twap_executor = TWAPExecutor()

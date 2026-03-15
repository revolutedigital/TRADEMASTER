"""Advanced order types: iceberg, bracket (take-profit + stop-loss), trailing stop."""
import asyncio
from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IcebergConfig:
    total_quantity: Decimal
    visible_quantity: Decimal  # How much to show per slice
    symbol: str
    side: str  # BUY or SELL
    price_limit: Decimal | None = None  # None = market order slices


@dataclass
class BracketConfig:
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal | None  # None = market
    take_profit_price: Decimal
    stop_loss_price: Decimal


@dataclass
class TrailingStopConfig:
    symbol: str
    side: str  # The original position side (BUY = trailing sell stop)
    quantity: Decimal
    trail_pct: float  # e.g., 0.02 = 2%
    activation_price: Decimal | None = None  # Only activate after reaching this price


class AdvancedOrderManager:
    """Manage advanced order types that decompose into simple orders."""

    async def execute_iceberg(self, config: IcebergConfig, executor) -> list[dict]:
        """Execute an iceberg order: split into small visible slices."""
        results = []
        remaining = config.total_quantity
        slice_num = 0

        while remaining > 0:
            slice_qty = min(config.visible_quantity, remaining)
            slice_num += 1

            logger.info("iceberg_slice", symbol=config.symbol, slice=slice_num,
                       qty=float(slice_qty), remaining=float(remaining))

            try:
                result = await executor(
                    symbol=config.symbol,
                    side=config.side,
                    quantity=float(slice_qty),
                )
                results.append({
                    "slice": slice_num,
                    "quantity": float(slice_qty),
                    "result": result,
                })
                remaining -= slice_qty

                # Small delay between slices to avoid detection
                if remaining > 0:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error("iceberg_slice_failed", slice=slice_num, error=str(e))
                results.append({
                    "slice": slice_num,
                    "quantity": float(slice_qty),
                    "error": str(e),
                })
                break

        return results

    def calculate_bracket_orders(self, config: BracketConfig) -> dict:
        """Calculate bracket order parameters (entry + TP + SL)."""
        return {
            "entry": {
                "symbol": config.symbol,
                "side": config.side,
                "quantity": float(config.quantity),
                "price": float(config.entry_price) if config.entry_price else None,
                "type": "LIMIT" if config.entry_price else "MARKET",
            },
            "take_profit": {
                "symbol": config.symbol,
                "side": "SELL" if config.side == "BUY" else "BUY",
                "quantity": float(config.quantity),
                "price": float(config.take_profit_price),
                "type": "LIMIT",
                "trigger": "take_profit",
            },
            "stop_loss": {
                "symbol": config.symbol,
                "side": "SELL" if config.side == "BUY" else "BUY",
                "quantity": float(config.quantity),
                "price": float(config.stop_loss_price),
                "type": "STOP_MARKET",
                "trigger": "stop_loss",
            },
            "risk_reward_ratio": self._calc_rr(config),
        }

    def _calc_rr(self, config: BracketConfig) -> float:
        """Calculate risk/reward ratio."""
        if config.entry_price is None:
            return 0

        if config.side == "BUY":
            reward = float(config.take_profit_price - config.entry_price)
            risk = float(config.entry_price - config.stop_loss_price)
        else:
            reward = float(config.entry_price - config.take_profit_price)
            risk = float(config.stop_loss_price - config.entry_price)

        return round(reward / risk, 2) if risk > 0 else 0

    def calculate_trailing_stop(
        self, config: TrailingStopConfig, current_price: float, peak_price: float,
    ) -> dict:
        """Calculate current trailing stop level."""
        if config.side == "BUY":
            # Trailing sell stop: stop moves up with price
            stop_price = peak_price * (1 - config.trail_pct)
            triggered = current_price <= stop_price
        else:
            # Trailing buy stop: stop moves down with price
            stop_price = peak_price * (1 + config.trail_pct)
            triggered = current_price >= stop_price

        activated = True
        if config.activation_price:
            if config.side == "BUY":
                activated = peak_price >= float(config.activation_price)
            else:
                activated = peak_price <= float(config.activation_price)

        return {
            "symbol": config.symbol,
            "current_price": current_price,
            "peak_price": peak_price,
            "stop_price": round(stop_price, 2),
            "trail_pct": config.trail_pct,
            "activated": activated,
            "triggered": triggered and activated,
        }


advanced_order_manager = AdvancedOrderManager()

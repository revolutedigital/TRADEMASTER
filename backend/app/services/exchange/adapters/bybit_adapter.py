"""Bybit exchange adapter with paper trading support.

Live trading methods raise NotImplementedError until the Bybit API
integration is implemented. Paper trading is fully functional using
the same Redis price cache that Binance uses.
"""

import random
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd

from app.core.logging import get_logger
from app.services.exchange.adapters.base import ExchangeAdapter

logger = get_logger(__name__)

# Paper trading constants (same as Binance for consistency)
PAPER_COMMISSION_RATE = Decimal("0.001")  # 0.1% taker fee
PAPER_SLIPPAGE_BPS = 5  # 5 basis points max


class BybitAdapter(ExchangeAdapter):
    """Bybit exchange adapter.

    Live methods raise NotImplementedError -- Bybit integration is planned.
    Paper trading works by reading prices from Redis (fed by frontend WS)
    and simulating fills with realistic slippage and commission.

    Paper mode tracks a local balance and order book for testing.
    """

    def __init__(self, paper_mode: bool = True) -> None:
        self._connected = False
        self._paper_mode = paper_mode

        # Paper trading state
        self._paper_balance: dict[str, Decimal] = {"USDT": Decimal("10000")}
        self._paper_orders: list[dict[str, Any]] = []
        self._paper_order_counter = 0

    # --- Connection lifecycle ---

    async def connect(self) -> None:
        """Mark as connected. No real Bybit connection yet."""
        self._connected = True
        logger.info("bybit_adapter_connected", mode="paper" if self._paper_mode else "live")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("bybit_adapter_disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # --- Market data ---

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get price from Redis cache (same source as Binance adapter).

        In production, this would call Bybit's /v5/market/tickers endpoint.
        For now, we share the Redis price cache since prices are similar
        across exchanges for major pairs.
        """
        if not self._paper_mode:
            raise NotImplementedError(
                "Bybit live market data not yet implemented. "
                "Use paper_mode=True for simulated trading."
            )

        # Read from Redis (prices fed by frontend browser WebSocket)
        try:
            from app.core.events import event_bus
            if event_bus._redis:
                cached = await event_bus._redis.get(f"price:{symbol}")
                if cached:
                    return Decimal(cached)
        except Exception:
            pass

        raise NotImplementedError(
            f"Cannot get live price for {symbol} on Bybit. "
            "Ensure frontend is running to feed prices to Redis."
        )

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Not implemented for live Bybit. Returns empty DataFrame structure.
        """
        raise NotImplementedError(
            "Bybit kline data not yet implemented. "
            "Use the Binance adapter for historical data."
        )

    # --- Account ---

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        """Get paper balance for an asset."""
        if not self._paper_mode:
            raise NotImplementedError("Bybit live balance not yet implemented")
        return self._paper_balance.get(asset, Decimal("0"))

    # --- Orders ---

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict[str, Any]:
        """Place an order. Only paper mode is supported.

        Paper mode simulates fills with slippage and commission,
        mirroring the Binance paper trading behavior.
        """
        if not self._paper_mode:
            raise NotImplementedError(
                "Bybit live order placement not yet implemented. "
                "Set paper_mode=True for simulated trading."
            )

        if order_type.upper() == "LIMIT" and price is None:
            raise ValueError("Price is required for LIMIT orders")

        return await self._execute_paper_order(symbol, side, quantity, order_type, price)

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Place a paper market order."""
        return await self.place_order(symbol, side, quantity, "MARKET")

    async def cancel_order(self, symbol: str, order_id: str | int) -> dict[str, Any]:
        """Cancel a paper order."""
        if not self._paper_mode:
            raise NotImplementedError("Bybit live order cancellation not yet implemented")

        for order in self._paper_orders:
            if str(order.get("orderId")) == str(order_id) and order["status"] == "NEW":
                order["status"] = "CANCELLED"
                logger.info("bybit_paper_order_cancelled", order_id=order_id, symbol=symbol)
                return {"orderId": order_id, "status": "CANCELLED", "symbol": symbol}

        return {"orderId": order_id, "status": "NOT_FOUND", "symbol": symbol}

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open paper orders."""
        if not self._paper_mode:
            raise NotImplementedError("Bybit live open orders not yet implemented")

        orders = [o for o in self._paper_orders if o["status"] == "NEW"]
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
        return orders

    # --- Identity ---

    @property
    def name(self) -> str:
        return "bybit"

    @property
    def supports_paper_trading(self) -> bool:
        return True

    # --- Paper trading internals ---

    async def _execute_paper_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: float | None,
    ) -> dict[str, Any]:
        """Simulate order execution with slippage and commission."""
        qty = Decimal(str(quantity))
        self._paper_order_counter += 1
        order_id = f"BYBIT-PAPER-{self._paper_order_counter}"
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        if order_type.upper() == "LIMIT" and price is not None:
            fill_price = Decimal(str(price))
        else:
            # Get price and apply slippage
            try:
                current_price = await self.get_ticker_price(symbol)
            except NotImplementedError:
                # Fallback to a reasonable simulation price
                current_price = Decimal("85000")
                logger.warning(
                    "bybit_paper_no_price",
                    symbol=symbol,
                    msg="Using fallback price for paper trading",
                )

            slippage_bps = Decimal(str(random.uniform(0, PAPER_SLIPPAGE_BPS)))
            slippage_pct = slippage_bps / Decimal("10000")
            direction = random.choice([-1, 1])
            if side.upper() == "BUY":
                fill_price = current_price * (Decimal("1") + slippage_pct * direction)
            else:
                fill_price = current_price * (Decimal("1") - slippage_pct * direction)

        fill_price = fill_price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        commission = (fill_price * qty * PAPER_COMMISSION_RATE).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Update paper balances
        cost = fill_price * qty
        base_asset = symbol.replace("USDT", "")

        if side.upper() == "BUY":
            usdt_balance = self._paper_balance.get("USDT", Decimal("0"))
            if usdt_balance >= cost + commission:
                self._paper_balance["USDT"] = usdt_balance - cost - commission
                self._paper_balance[base_asset] = (
                    self._paper_balance.get(base_asset, Decimal("0")) + qty
                )
        else:
            asset_balance = self._paper_balance.get(base_asset, Decimal("0"))
            if asset_balance >= qty:
                self._paper_balance[base_asset] = asset_balance - qty
                self._paper_balance["USDT"] = (
                    self._paper_balance.get("USDT", Decimal("0")) + cost - commission
                )

        order_data = {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "status": "FILLED",
            "executedQty": str(qty),
            "avgPrice": str(fill_price),
            "commission": str(commission),
            "commissionAsset": "USDT",
            "transactTime": ts,
            "paper_mode": True,
            "exchange": "bybit",
        }

        self._paper_orders.append(order_data)

        logger.info(
            "bybit_paper_order_executed",
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=float(qty),
            fill_price=float(fill_price),
            commission=float(commission),
        )

        return order_data

"""Cross-exchange arbitrage engine.

Monitors real-time price spreads between supported exchanges (Binance,
Bybit, OKX) and executes atomic buy+sell when the spread exceeds the
minimum profit threshold after accounting for fees and transfer costs.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Any

from app.core.logging import get_logger
from app.services.exchange.factory import get_exchange_adapter, list_exchanges

logger = get_logger(__name__)


class ArbStatus(str, Enum):
    MONITORING = "monitoring"
    EXECUTING = "executing"
    STOPPED = "stopped"


class LegStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    FAILED = "failed"


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class ArbConfig:
    """Tunable parameters for the arbitrage engine."""

    # Minimum spread (bps) after fees/costs to trigger execution.
    min_profit_bps: Decimal = Decimal("15")

    # Maximum notional exposure across all open arb positions (USD).
    max_exposure_usd: Decimal = Decimal("10000")

    # Maximum time (seconds) to hold an incomplete arb position.
    max_holding_seconds: float = 300.0

    # Assumed round-trip trading fee per exchange (taker, in fraction).
    fee_per_exchange: Decimal = Decimal("0.001")

    # Estimated transfer cost in quote currency (USDT) per transfer.
    transfer_cost_usd: Decimal = Decimal("1.5")

    # Estimated transfer time in seconds (used in risk assessment only).
    transfer_time_seconds: float = 120.0

    # Poll interval (seconds) when scanning for opportunities.
    scan_interval: float = 1.0

    # Maximum number of concurrent open arb positions.
    max_concurrent_positions: int = 3

    # Symbols to monitor.
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class SpreadOpportunity:
    """A detected cross-exchange spread."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    spread_bps: Decimal
    net_profit_bps: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ArbLeg:
    """One side of an arbitrage trade."""
    exchange: str
    side: str  # BUY or SELL
    price: Decimal
    quantity: Decimal
    status: LegStatus = LegStatus.PENDING
    order_id: int | None = None
    fill_price: Decimal = Decimal("0")
    filled_at: datetime | None = None


@dataclass
class ArbPosition:
    """A complete arbitrage position (buy leg + sell leg)."""
    id: str
    symbol: str
    buy_leg: ArbLeg
    sell_leg: ArbLeg
    expected_profit_bps: Decimal
    opened_at: float = field(default_factory=time.monotonic)
    closed: bool = False
    realized_pnl: Decimal = Decimal("0")


@dataclass
class ArbDashboard:
    """Performance snapshot for the dashboard API."""
    status: ArbStatus
    opportunities_found: int
    opportunities_executed: int
    total_realized_pnl: Decimal
    open_positions: int
    win_rate: float
    avg_profit_bps: Decimal
    total_trades: int
    uptime_seconds: float
    last_opportunity: SpreadOpportunity | None


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class CrossExchangeArbitrage:
    """Detect and execute cross-exchange arbitrage opportunities.

    The engine continuously polls ticker prices on all connected
    exchanges, computes the pairwise spread for each monitored symbol,
    and fires an atomic simultaneous buy+sell when the net spread (after
    fees and estimated transfer costs) exceeds the configured minimum.

    Risk controls
    -------------
    * **Max exposure**: total notional of all open arb positions is
      capped at ``config.max_exposure_usd``.
    * **Max holding time**: if a position is not closed within
      ``config.max_holding_seconds`` the engine logs an alert.
    * **Min profit threshold**: opportunities below
      ``config.min_profit_bps`` (net of fees/transfer costs) are
      ignored.
    * **Fill verification**: both legs must fill; if one fails the
      engine attempts to unwind the filled leg.
    """

    def __init__(self, config: ArbConfig | None = None) -> None:
        self._config = config or ArbConfig()
        self._status = ArbStatus.STOPPED
        self._positions: list[ArbPosition] = []
        self._history: list[ArbPosition] = []
        self._opportunities: list[SpreadOpportunity] = []
        self._scan_task: asyncio.Task[None] | None = None
        self._start_time: float = 0.0

        # Counters
        self._opportunities_found: int = 0
        self._opportunities_executed: int = 0
        self._total_pnl = Decimal("0")
        self._wins: int = 0
        self._losses: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin monitoring for arbitrage opportunities."""
        if self._status == ArbStatus.MONITORING:
            logger.warning("arb_already_running")
            return

        self._status = ArbStatus.MONITORING
        self._start_time = time.monotonic()
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info(
            "arb_started",
            symbols=self._config.symbols,
            min_profit_bps=str(self._config.min_profit_bps),
            max_exposure=str(self._config.max_exposure_usd),
        )

    async def stop(self) -> None:
        """Stop the monitoring loop gracefully."""
        self._status = ArbStatus.STOPPED
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        self._scan_task = None
        logger.info(
            "arb_stopped",
            executed=self._opportunities_executed,
            pnl=str(self._total_pnl),
        )

    # ------------------------------------------------------------------
    # Core scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Continuously scan for spread opportunities."""
        while self._status == ArbStatus.MONITORING:
            try:
                await self._check_holding_limits()

                for symbol in self._config.symbols:
                    opportunity = await self._scan_spread(symbol)
                    if opportunity and opportunity.net_profit_bps >= self._config.min_profit_bps:
                        self._opportunities_found += 1
                        self._opportunities.append(opportunity)
                        # Keep history bounded.
                        if len(self._opportunities) > 5000:
                            self._opportunities = self._opportunities[-2500:]

                        logger.info(
                            "arb_opportunity_detected",
                            symbol=symbol,
                            buy=opportunity.buy_exchange,
                            sell=opportunity.sell_exchange,
                            spread_bps=str(opportunity.spread_bps),
                            net_bps=str(opportunity.net_profit_bps),
                        )

                        if self._can_open_position(opportunity):
                            await self._execute_opportunity(opportunity)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("arb_scan_error", error=str(exc))

            await asyncio.sleep(self._config.scan_interval)

    # ------------------------------------------------------------------
    # Spread detection
    # ------------------------------------------------------------------

    async def _scan_spread(self, symbol: str) -> SpreadOpportunity | None:
        """Fetch prices from all exchanges and find the best spread."""
        prices: dict[str, Decimal] = {}

        fetch_tasks = []
        exchange_names = list_exchanges()

        for name in exchange_names:
            fetch_tasks.append(self._fetch_price(name, symbol))

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for name, result in zip(exchange_names, results):
            if isinstance(result, Exception):
                logger.debug("arb_price_fetch_failed", exchange=name, error=str(result))
                continue
            if result and result > 0:
                prices[name] = result

        if len(prices) < 2:
            return None

        # Find the pair with the largest spread.
        best_opportunity: SpreadOpportunity | None = None
        best_net = Decimal("-1")

        exchanges = list(prices.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                low_ex = exchanges[i] if prices[exchanges[i]] <= prices[exchanges[j]] else exchanges[j]
                high_ex = exchanges[j] if low_ex == exchanges[i] else exchanges[i]

                buy_price = prices[low_ex]
                sell_price = prices[high_ex]

                if buy_price <= 0:
                    continue

                spread_bps = ((sell_price - buy_price) / buy_price * Decimal("10000")).quantize(
                    Decimal("0.01")
                )

                # Net profit after fees on both legs + transfer cost.
                total_fee_bps = self._config.fee_per_exchange * 2 * Decimal("10000")
                # Convert transfer cost to bps relative to the trade price.
                # Use a reference notional of 1 unit for bps comparison.
                transfer_bps = (
                    (self._config.transfer_cost_usd / buy_price) * Decimal("10000")
                ).quantize(Decimal("0.01"))
                net_bps = (spread_bps - total_fee_bps - transfer_bps).quantize(Decimal("0.01"))

                if net_bps > best_net:
                    best_net = net_bps
                    best_opportunity = SpreadOpportunity(
                        symbol=symbol,
                        buy_exchange=low_ex,
                        sell_exchange=high_ex,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        spread_bps=spread_bps,
                        net_profit_bps=net_bps,
                    )

        return best_opportunity

    async def _fetch_price(self, exchange_name: str, symbol: str) -> Decimal:
        """Fetch the current ticker price from a single exchange."""
        adapter = get_exchange_adapter(exchange_name)
        if not adapter.is_connected:
            return Decimal("0")
        return await adapter.get_ticker_price(symbol)

    # ------------------------------------------------------------------
    # Risk controls
    # ------------------------------------------------------------------

    def _can_open_position(self, opportunity: SpreadOpportunity) -> bool:
        """Check whether risk limits allow opening a new position."""
        # Concurrent position limit.
        open_count = sum(1 for p in self._positions if not p.closed)
        if open_count >= self._config.max_concurrent_positions:
            logger.debug("arb_max_positions_reached", count=open_count)
            return False

        # Exposure limit.
        current_exposure = self._calculate_exposure()
        if current_exposure >= self._config.max_exposure_usd:
            logger.debug(
                "arb_max_exposure_reached",
                current=str(current_exposure),
                limit=str(self._config.max_exposure_usd),
            )
            return False

        return True

    def _calculate_exposure(self) -> Decimal:
        """Sum notional value of all open positions."""
        total = Decimal("0")
        for pos in self._positions:
            if not pos.closed:
                total += pos.buy_leg.price * pos.buy_leg.quantity
        return total

    async def _check_holding_limits(self) -> None:
        """Flag positions that have exceeded the max holding time."""
        now = time.monotonic()
        for pos in self._positions:
            if pos.closed:
                continue
            age = now - pos.opened_at
            if age > self._config.max_holding_seconds:
                logger.warning(
                    "arb_holding_timeout",
                    position_id=pos.id,
                    symbol=pos.symbol,
                    age_seconds=round(age, 1),
                    limit=self._config.max_holding_seconds,
                )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_opportunity(self, opp: SpreadOpportunity) -> None:
        """Atomically execute both legs of an arbitrage opportunity.

        Both buy and sell orders are submitted simultaneously.  If one
        leg fails, the engine attempts to unwind the successful leg.
        """
        self._status = ArbStatus.EXECUTING

        # Determine order quantity: use max_exposure / price, bounded.
        available_exposure = self._config.max_exposure_usd - self._calculate_exposure()
        quantity = (available_exposure / opp.buy_price).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
        if quantity <= 0:
            self._status = ArbStatus.MONITORING
            return

        position_id = f"arb-{int(time.time() * 1000)}"
        buy_leg = ArbLeg(
            exchange=opp.buy_exchange,
            side="BUY",
            price=opp.buy_price,
            quantity=quantity,
        )
        sell_leg = ArbLeg(
            exchange=opp.sell_exchange,
            side="SELL",
            price=opp.sell_price,
            quantity=quantity,
        )
        position = ArbPosition(
            id=position_id,
            symbol=opp.symbol,
            buy_leg=buy_leg,
            sell_leg=sell_leg,
            expected_profit_bps=opp.net_profit_bps,
        )
        self._positions.append(position)

        logger.info(
            "arb_executing",
            position_id=position_id,
            symbol=opp.symbol,
            buy_exchange=opp.buy_exchange,
            sell_exchange=opp.sell_exchange,
            quantity=str(quantity),
            expected_profit_bps=str(opp.net_profit_bps),
        )

        # Fire both legs simultaneously.
        buy_task = self._place_leg_order(opp.symbol, buy_leg)
        sell_task = self._place_leg_order(opp.symbol, sell_leg)
        buy_result, sell_result = await asyncio.gather(
            buy_task, sell_task, return_exceptions=True
        )

        # Process results.
        buy_ok = self._process_leg_result(buy_leg, buy_result)
        sell_ok = self._process_leg_result(sell_leg, sell_result)

        if buy_ok and sell_ok:
            # Both filled -- compute realized PnL.
            pnl = (sell_leg.fill_price - buy_leg.fill_price) * quantity
            # Subtract fees.
            fee_cost = (
                buy_leg.fill_price * quantity * self._config.fee_per_exchange
                + sell_leg.fill_price * quantity * self._config.fee_per_exchange
            )
            pnl -= fee_cost
            pnl -= self._config.transfer_cost_usd

            position.realized_pnl = pnl.quantize(Decimal("0.01"))
            position.closed = True
            self._total_pnl += position.realized_pnl
            self._opportunities_executed += 1

            if pnl > 0:
                self._wins += 1
            else:
                self._losses += 1

            self._history.append(position)
            self._positions.remove(position)

            logger.info(
                "arb_executed",
                position_id=position_id,
                pnl=str(position.realized_pnl),
                buy_fill=str(buy_leg.fill_price),
                sell_fill=str(sell_leg.fill_price),
            )

        elif buy_ok and not sell_ok:
            # Unwind the buy leg.
            logger.error("arb_sell_leg_failed_unwinding", position_id=position_id)
            await self._unwind_leg(opp.symbol, buy_leg)
            position.closed = True

        elif sell_ok and not buy_ok:
            # Unwind the sell leg.
            logger.error("arb_buy_leg_failed_unwinding", position_id=position_id)
            await self._unwind_leg(opp.symbol, sell_leg)
            position.closed = True

        else:
            # Both failed.
            logger.error("arb_both_legs_failed", position_id=position_id)
            position.closed = True

        self._status = ArbStatus.MONITORING

    async def _place_leg_order(self, symbol: str, leg: ArbLeg) -> dict[str, Any]:
        """Place a market order on the appropriate exchange for one arb leg."""
        adapter = get_exchange_adapter(leg.exchange)
        result = await adapter.place_market_order(
            symbol=symbol,
            side=leg.side,
            quantity=float(leg.quantity),
        )
        return result

    def _process_leg_result(
        self, leg: ArbLeg, result: dict[str, Any] | BaseException
    ) -> bool:
        """Update leg status from order result. Returns True if filled."""
        if isinstance(result, BaseException):
            leg.status = LegStatus.FAILED
            logger.error(
                "arb_leg_error",
                exchange=leg.exchange,
                side=leg.side,
                error=str(result),
            )
            return False

        try:
            fill_price = Decimal(str(
                result.get("avgPrice") or result.get("price", "0")
            ))
            leg.fill_price = fill_price
            leg.status = LegStatus.FILLED
            leg.filled_at = datetime.now(timezone.utc)
            leg.order_id = result.get("orderId")
            return True
        except Exception as exc:
            leg.status = LegStatus.FAILED
            logger.error("arb_leg_parse_error", error=str(exc))
            return False

    async def _unwind_leg(self, symbol: str, leg: ArbLeg) -> None:
        """Attempt to reverse a filled leg to exit the position."""
        opposite_side = "SELL" if leg.side == "BUY" else "BUY"
        try:
            adapter = get_exchange_adapter(leg.exchange)
            await adapter.place_market_order(
                symbol=symbol,
                side=opposite_side,
                quantity=float(leg.quantity),
            )
            logger.info(
                "arb_leg_unwound",
                exchange=leg.exchange,
                original_side=leg.side,
                quantity=str(leg.quantity),
            )
        except Exception as exc:
            logger.error(
                "arb_unwind_failed",
                exchange=leg.exchange,
                side=opposite_side,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Dashboard / reporting
    # ------------------------------------------------------------------

    def get_dashboard(self) -> ArbDashboard:
        """Return a snapshot of the engine's performance for the API."""
        total_trades = self._wins + self._losses
        win_rate = (self._wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_profit = Decimal("0")
        if self._history:
            avg_profit = (
                sum(p.expected_profit_bps for p in self._history)
                / Decimal(str(len(self._history)))
            ).quantize(Decimal("0.01"))

        uptime = time.monotonic() - self._start_time if self._start_time else 0.0

        last_opp = self._opportunities[-1] if self._opportunities else None

        return ArbDashboard(
            status=self._status,
            opportunities_found=self._opportunities_found,
            opportunities_executed=self._opportunities_executed,
            total_realized_pnl=self._total_pnl.quantize(Decimal("0.01")),
            open_positions=sum(1 for p in self._positions if not p.closed),
            win_rate=round(win_rate, 1),
            avg_profit_bps=avg_profit,
            total_trades=total_trades,
            uptime_seconds=round(uptime, 1),
            last_opportunity=last_opp,
        )

    def get_recent_opportunities(self, limit: int = 50) -> list[SpreadOpportunity]:
        """Return the most recent detected opportunities."""
        return self._opportunities[-limit:]

    def get_position_history(self, limit: int = 50) -> list[ArbPosition]:
        """Return closed arbitrage positions."""
        return self._history[-limit:]


# Module-level singleton.
cross_exchange_arb = CrossExchangeArbitrage()

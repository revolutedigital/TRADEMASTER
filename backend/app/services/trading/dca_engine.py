"""Dollar-Cost Averaging engine with smart scheduling and performance tracking.

Supports multiple DCA strategies:
- Fixed schedule (hourly, daily, weekly, monthly)
- Target DCA (split a budget across scheduled intervals)
- Smart DCA (adjust buy size using the Fear & Greed Index)
- Dip-buying DCA (increase allocation when price drops below SMA)
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class DCAFrequency(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DCAStrategy(str, Enum):
    FIXED = "fixed"
    TARGET = "target"
    SMART = "smart"
    DIP_BUYING = "dip_buying"


class DCAStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class DCASchedule:
    """Configuration for a DCA plan."""

    id: str
    user_id: str
    symbol: str
    strategy: DCAStrategy
    frequency: DCAFrequency
    amount_per_buy: float
    total_budget: float | None = None
    total_spent: float = 0.0
    total_quantity: float = 0.0
    num_buys_executed: int = 0
    num_buys_planned: int | None = None
    status: DCAStatus = DCAStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    next_buy_at: datetime | None = None
    last_buy_at: datetime | None = None

    # Smart DCA parameters
    fear_greed_multiplier: bool = False
    base_amount: float | None = None

    # Dip-buying parameters
    sma_period: int = 50
    dip_threshold_pct: float = 5.0
    dip_extra_multiplier: float = 2.0

    # Execution history
    executions: list[dict] = field(default_factory=list)


@dataclass
class DCAPerformance:
    """Comparison of DCA results against a lump-sum baseline."""

    schedule_id: str
    symbol: str
    total_invested: float
    total_quantity: float
    average_cost: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    num_buys: int
    # Lump-sum comparison
    lump_sum_quantity: float
    lump_sum_value: float
    lump_sum_pnl: float
    lump_sum_pnl_pct: float
    dca_vs_lump_sum_pct: float  # Positive means DCA outperformed


class DCAEngine:
    """Dollar-Cost Averaging engine supporting multiple scheduling strategies.

    Usage::

        engine = DCAEngine()
        schedule = await engine.create_schedule(
            user_id="user_123",
            symbol="BTCUSDT",
            strategy=DCAStrategy.SMART,
            frequency=DCAFrequency.WEEKLY,
            amount_per_buy=125.0,
            total_budget=500.0,
        )
        await engine.start()
    """

    FREQUENCY_INTERVALS: dict[DCAFrequency, timedelta] = {
        DCAFrequency.HOURLY: timedelta(hours=1),
        DCAFrequency.DAILY: timedelta(days=1),
        DCAFrequency.WEEKLY: timedelta(weeks=1),
        DCAFrequency.MONTHLY: timedelta(days=30),
    }

    def __init__(self) -> None:
        self._schedules: dict[str, DCASchedule] = {}
        self._running: bool = False
        self._loop_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the DCA scheduling loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("dca_engine_started", active_schedules=len(self._schedules))

    async def stop(self) -> None:
        """Stop the DCA scheduling loop gracefully."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("dca_engine_stopped")

    # ------------------------------------------------------------------
    # Schedule management
    # ------------------------------------------------------------------

    async def create_schedule(
        self,
        user_id: str,
        symbol: str,
        strategy: DCAStrategy,
        frequency: DCAFrequency,
        amount_per_buy: float,
        total_budget: float | None = None,
        sma_period: int = 50,
        dip_threshold_pct: float = 5.0,
        dip_extra_multiplier: float = 2.0,
        fear_greed_multiplier: bool = False,
    ) -> DCASchedule:
        """Create and register a new DCA schedule.

        Args:
            user_id: Owner of the schedule.
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            strategy: Which DCA variant to use.
            frequency: How often to buy.
            amount_per_buy: Base quote-currency amount per execution.
            total_budget: Optional cap on total spend.
            sma_period: SMA lookback for dip-buying strategy.
            dip_threshold_pct: % below SMA that triggers extra buying.
            dip_extra_multiplier: Multiplier applied on dip buys.
            fear_greed_multiplier: Whether to scale size by Fear & Greed.

        Returns:
            The newly created :class:`DCASchedule`.
        """
        if amount_per_buy <= 0:
            raise ValueError("amount_per_buy must be positive")
        if total_budget is not None and total_budget < amount_per_buy:
            raise ValueError("total_budget must be >= amount_per_buy")

        schedule_id = f"dca_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        next_buy = now + self.FREQUENCY_INTERVALS[frequency]

        num_buys_planned: int | None = None
        if strategy == DCAStrategy.TARGET and total_budget is not None:
            num_buys_planned = max(1, int(total_budget / amount_per_buy))

        schedule = DCASchedule(
            id=schedule_id,
            user_id=user_id,
            symbol=symbol,
            strategy=strategy,
            frequency=frequency,
            amount_per_buy=amount_per_buy,
            total_budget=total_budget,
            num_buys_planned=num_buys_planned,
            next_buy_at=next_buy,
            base_amount=amount_per_buy,
            fear_greed_multiplier=fear_greed_multiplier,
            sma_period=sma_period,
            dip_threshold_pct=dip_threshold_pct,
            dip_extra_multiplier=dip_extra_multiplier,
        )

        self._schedules[schedule_id] = schedule
        logger.info(
            "dca_schedule_created",
            id=schedule_id,
            user_id=user_id,
            symbol=symbol,
            strategy=strategy.value,
            frequency=frequency.value,
            amount=amount_per_buy,
            budget=total_budget,
        )
        return schedule

    async def pause_schedule(self, schedule_id: str) -> DCASchedule:
        """Pause an active schedule."""
        schedule = self._get_schedule(schedule_id)
        schedule.status = DCAStatus.PAUSED
        logger.info("dca_schedule_paused", id=schedule_id)
        return schedule

    async def resume_schedule(self, schedule_id: str) -> DCASchedule:
        """Resume a paused schedule, recalculating the next buy time."""
        schedule = self._get_schedule(schedule_id)
        if schedule.status != DCAStatus.PAUSED:
            raise ValueError(f"Schedule {schedule_id} is not paused (status={schedule.status.value})")
        schedule.status = DCAStatus.ACTIVE
        schedule.next_buy_at = datetime.now(timezone.utc) + self.FREQUENCY_INTERVALS[schedule.frequency]
        logger.info("dca_schedule_resumed", id=schedule_id)
        return schedule

    async def cancel_schedule(self, schedule_id: str) -> DCASchedule:
        """Cancel a schedule permanently."""
        schedule = self._get_schedule(schedule_id)
        schedule.status = DCAStatus.CANCELLED
        logger.info("dca_schedule_cancelled", id=schedule_id, total_spent=schedule.total_spent)
        return schedule

    def get_schedule(self, schedule_id: str) -> DCASchedule:
        """Return schedule by id or raise."""
        return self._get_schedule(schedule_id)

    def get_user_schedules(self, user_id: str) -> list[DCASchedule]:
        """Return all schedules belonging to a user."""
        return [s for s in self._schedules.values() if s.user_id == user_id]

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main loop: checks schedules every 30 seconds."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                due = [
                    s
                    for s in self._schedules.values()
                    if s.status == DCAStatus.ACTIVE and s.next_buy_at and s.next_buy_at <= now
                ]
                for schedule in due:
                    try:
                        await self._execute_buy(schedule)
                    except Exception as exc:
                        logger.error(
                            "dca_buy_failed",
                            id=schedule.id,
                            symbol=schedule.symbol,
                            error=str(exc),
                        )
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("dca_loop_error", error=str(exc))
                await asyncio.sleep(60)

    async def _execute_buy(self, schedule: DCASchedule) -> None:
        """Execute a single DCA buy for the given schedule."""
        current_price = await self._get_current_price(schedule.symbol)
        if current_price <= 0:
            logger.warning("dca_invalid_price", id=schedule.id, price=current_price)
            return

        buy_amount = await self._calculate_buy_amount(schedule, current_price)

        # Budget guard
        if schedule.total_budget is not None:
            remaining = schedule.total_budget - schedule.total_spent
            if remaining <= 0:
                schedule.status = DCAStatus.COMPLETED
                logger.info("dca_schedule_completed_budget", id=schedule.id)
                return
            buy_amount = min(buy_amount, remaining)

        quantity = buy_amount / current_price

        # Record execution
        execution = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": current_price,
            "amount": buy_amount,
            "quantity": quantity,
            "buy_number": schedule.num_buys_executed + 1,
        }

        schedule.total_spent += buy_amount
        schedule.total_quantity += quantity
        schedule.num_buys_executed += 1
        schedule.last_buy_at = datetime.now(timezone.utc)
        schedule.next_buy_at = datetime.now(timezone.utc) + self.FREQUENCY_INTERVALS[schedule.frequency]
        schedule.executions.append(execution)

        logger.info(
            "dca_buy_executed",
            id=schedule.id,
            symbol=schedule.symbol,
            price=current_price,
            amount=buy_amount,
            quantity=quantity,
            total_spent=schedule.total_spent,
            buy_number=schedule.num_buys_executed,
        )

        # Check if target plan is done
        if (
            schedule.strategy == DCAStrategy.TARGET
            and schedule.num_buys_planned is not None
            and schedule.num_buys_executed >= schedule.num_buys_planned
        ):
            schedule.status = DCAStatus.COMPLETED
            logger.info("dca_schedule_completed_target", id=schedule.id)

    # ------------------------------------------------------------------
    # Strategy-specific buy-amount calculation
    # ------------------------------------------------------------------

    async def _calculate_buy_amount(self, schedule: DCASchedule, current_price: float) -> float:
        """Determine how much to buy based on the active strategy."""
        base = schedule.amount_per_buy

        if schedule.strategy == DCAStrategy.FIXED:
            return base

        if schedule.strategy == DCAStrategy.TARGET:
            return base

        if schedule.strategy == DCAStrategy.SMART:
            return await self._smart_dca_amount(schedule, base)

        if schedule.strategy == DCAStrategy.DIP_BUYING:
            return await self._dip_buying_amount(schedule, base, current_price)

        return base

    async def _smart_dca_amount(self, schedule: DCASchedule, base_amount: float) -> float:
        """Adjust buy amount using the Fear & Greed Index.

        - Extreme Fear (0-25): buy 1.5x-2.0x
        - Fear (25-45): buy 1.2x
        - Neutral (45-55): buy 1.0x
        - Greed (55-75): buy 0.8x
        - Extreme Greed (75-100): buy 0.5x
        """
        fg_index = await self._get_fear_greed_index()
        if fg_index is None:
            logger.warning("dca_fear_greed_unavailable", id=schedule.id)
            return base_amount

        if fg_index <= 10:
            multiplier = 2.0
        elif fg_index <= 25:
            multiplier = 1.5
        elif fg_index <= 45:
            multiplier = 1.2
        elif fg_index <= 55:
            multiplier = 1.0
        elif fg_index <= 75:
            multiplier = 0.8
        else:
            multiplier = 0.5

        adjusted = base_amount * multiplier
        logger.debug(
            "dca_smart_adjustment",
            id=schedule.id,
            fear_greed=fg_index,
            multiplier=multiplier,
            base=base_amount,
            adjusted=adjusted,
        )
        return adjusted

    async def _dip_buying_amount(
        self,
        schedule: DCASchedule,
        base_amount: float,
        current_price: float,
    ) -> float:
        """Buy extra when price drops more than ``dip_threshold_pct`` below the SMA.

        Returns the base amount when price is at or above the SMA, or
        ``base_amount * dip_extra_multiplier`` when the dip condition is met.
        """
        sma = await self._get_sma(schedule.symbol, schedule.sma_period)
        if sma is None or sma <= 0:
            return base_amount

        pct_below_sma = ((sma - current_price) / sma) * 100.0
        if pct_below_sma >= schedule.dip_threshold_pct:
            adjusted = base_amount * schedule.dip_extra_multiplier
            logger.info(
                "dca_dip_buy_triggered",
                id=schedule.id,
                symbol=schedule.symbol,
                pct_below_sma=round(pct_below_sma, 2),
                multiplier=schedule.dip_extra_multiplier,
                adjusted_amount=adjusted,
            )
            return adjusted

        return base_amount

    # ------------------------------------------------------------------
    # Performance analytics
    # ------------------------------------------------------------------

    async def get_performance(self, schedule_id: str) -> DCAPerformance:
        """Calculate DCA vs lump-sum performance for a schedule.

        Returns a :class:`DCAPerformance` snapshot with unrealized P&L and a
        side-by-side lump-sum comparison based on the price at the first
        execution.
        """
        schedule = self._get_schedule(schedule_id)

        if schedule.num_buys_executed == 0:
            raise ValueError("No executions yet; performance unavailable")

        current_price = await self._get_current_price(schedule.symbol)
        avg_cost = schedule.total_spent / schedule.total_quantity if schedule.total_quantity > 0 else 0
        current_value = schedule.total_quantity * current_price
        unrealized_pnl = current_value - schedule.total_spent
        unrealized_pnl_pct = (unrealized_pnl / schedule.total_spent * 100) if schedule.total_spent > 0 else 0

        # Lump-sum: what if user had invested total_spent at the first buy price?
        first_price = schedule.executions[0]["price"]
        lump_sum_qty = schedule.total_spent / first_price if first_price > 0 else 0
        lump_sum_value = lump_sum_qty * current_price
        lump_sum_pnl = lump_sum_value - schedule.total_spent
        lump_sum_pnl_pct = (lump_sum_pnl / schedule.total_spent * 100) if schedule.total_spent > 0 else 0

        dca_vs_lump = unrealized_pnl_pct - lump_sum_pnl_pct

        performance = DCAPerformance(
            schedule_id=schedule_id,
            symbol=schedule.symbol,
            total_invested=schedule.total_spent,
            total_quantity=schedule.total_quantity,
            average_cost=avg_cost,
            current_price=current_price,
            current_value=current_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
            num_buys=schedule.num_buys_executed,
            lump_sum_quantity=lump_sum_qty,
            lump_sum_value=lump_sum_value,
            lump_sum_pnl=lump_sum_pnl,
            lump_sum_pnl_pct=round(lump_sum_pnl_pct, 2),
            dca_vs_lump_sum_pct=round(dca_vs_lump, 2),
        )

        logger.info(
            "dca_performance_calculated",
            id=schedule_id,
            avg_cost=round(avg_cost, 2),
            current_price=current_price,
            pnl_pct=performance.unrealized_pnl_pct,
            vs_lump=performance.dca_vs_lump_sum_pct,
        )
        return performance

    # ------------------------------------------------------------------
    # External data hooks (override or inject in production)
    # ------------------------------------------------------------------

    async def _get_current_price(self, symbol: str) -> float:
        """Fetch the latest price for *symbol*.

        In production this delegates to the exchange client or market-data
        service.  The default implementation returns ``0.0`` so the engine
        can be instantiated without live connectivity during tests.
        """
        # Integration point: replace with exchange client call
        # e.g. return await binance_client.get_ticker_price(symbol)
        logger.debug("dca_price_fetch", symbol=symbol)
        return 0.0

    async def _get_fear_greed_index(self) -> int | None:
        """Return the current Fear & Greed Index (0-100).

        Override to pull from a live data source such as alternative.me or
        an internal sentiment pipeline.
        """
        logger.debug("dca_fear_greed_fetch")
        return None

    async def _get_sma(self, symbol: str, period: int) -> float | None:
        """Return the Simple Moving Average for *symbol* over *period* candles.

        Override to integrate with the indicators service.
        """
        logger.debug("dca_sma_fetch", symbol=symbol, period=period)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_schedule(self, schedule_id: str) -> DCASchedule:
        schedule = self._schedules.get(schedule_id)
        if schedule is None:
            raise KeyError(f"DCA schedule not found: {schedule_id}")
        return schedule


dca_engine = DCAEngine()

"""Real-time risk engine with sub-millisecond pre-trade risk checks."""

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from collections import deque

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class RiskCheckResult(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    WARNING = "warning"


@dataclass
class RiskLimit:
    """Configurable risk limit."""
    name: str
    max_value: float
    current_value: float = 0.0
    breach_count: int = 0
    last_breach: float = 0.0

    @property
    def utilization(self) -> float:
        if self.max_value == 0:
            return 0.0
        return self.current_value / self.max_value

    @property
    def is_breached(self) -> bool:
        return self.current_value > self.max_value


@dataclass
class PreTradeCheck:
    """Result of a single pre-trade risk check."""
    check_name: str
    result: RiskCheckResult
    message: str
    latency_us: float  # microseconds
    limit_utilization: float = 0.0


@dataclass
class RiskCheckResponse:
    """Aggregated pre-trade risk check response."""
    order_id: str
    overall_result: RiskCheckResult
    checks: list[PreTradeCheck]
    total_latency_us: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "result": self.overall_result.value,
            "checks": [
                {
                    "name": c.check_name,
                    "result": c.result.value,
                    "message": c.message,
                    "latency_us": round(c.latency_us, 1),
                    "utilization": round(c.limit_utilization, 4),
                }
                for c in self.checks
            ],
            "total_latency_us": round(self.total_latency_us, 1),
            "timestamp": self.timestamp,
        }


class RealTimeRiskEngine:
    """
    Ultra-fast pre-trade risk engine.

    Performs sub-millisecond risk checks before every order:
    - Position limits (per-symbol, total)
    - Notional limits (per-order, daily)
    - Concentration limits
    - Rate limits (orders per second)
    - Drawdown circuit breaker
    - Volatility-adjusted sizing
    - Fat finger protection
    """

    def __init__(self):
        # Risk limits
        self.limits: dict[str, RiskLimit] = {
            "max_position_pct": RiskLimit("Max Position % of Portfolio", 0.25),
            "max_single_order_pct": RiskLimit("Max Single Order % of Portfolio", 0.10),
            "max_daily_notional": RiskLimit("Max Daily Notional (USD)", 100_000.0),
            "max_daily_trades": RiskLimit("Max Daily Trades", 50),
            "max_daily_loss_pct": RiskLimit("Max Daily Loss %", 0.05),
            "max_drawdown_pct": RiskLimit("Max Drawdown %", 0.10),
            "max_concentration_pct": RiskLimit("Max Single Asset Concentration", 0.50),
            "max_orders_per_minute": RiskLimit("Max Orders Per Minute", 30),
        }

        # State tracking
        self._daily_notional: float = 0.0
        self._daily_trades: int = 0
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._positions: dict[str, float] = {}  # symbol -> notional
        self._order_timestamps: deque[float] = deque(maxlen=100)
        self._price_cache: dict[str, float] = {}
        self._volatility_cache: dict[str, float] = {}

        # Performance tracking
        self._check_latencies: deque[float] = deque(maxlen=1000)
        self._total_checks: int = 0
        self._total_rejections: int = 0

        logger.info("realtime_risk_engine_initialized")

    def check_order(self, order: dict) -> RiskCheckResponse:
        """
        Run all pre-trade risk checks on an order.

        Args:
            order: dict with keys: order_id, symbol, side, quantity, price, notional

        Returns:
            RiskCheckResponse with all check results
        """
        start = time.perf_counter_ns()
        checks: list[PreTradeCheck] = []

        symbol = order.get("symbol", "UNKNOWN")
        side = order.get("side", "BUY")
        notional = float(order.get("notional", 0))
        price = float(order.get("price", 0))
        quantity = float(order.get("quantity", 0))
        order_id = order.get("order_id", "unknown")

        if notional == 0 and price > 0 and quantity > 0:
            notional = price * quantity

        # 1. Position size check
        checks.append(self._check_position_size(symbol, notional, side))

        # 2. Single order size check
        checks.append(self._check_order_size(notional))

        # 3. Daily notional check
        checks.append(self._check_daily_notional(notional))

        # 4. Daily trade count check
        checks.append(self._check_daily_trades())

        # 5. Daily loss check
        checks.append(self._check_daily_loss())

        # 6. Drawdown check
        checks.append(self._check_drawdown())

        # 7. Concentration check
        checks.append(self._check_concentration(symbol, notional, side))

        # 8. Rate limit check
        checks.append(self._check_rate_limit())

        # 9. Fat finger check
        checks.append(self._check_fat_finger(symbol, price))

        # 10. Volatility check
        checks.append(self._check_volatility(symbol, notional))

        # Determine overall result
        has_rejection = any(c.result == RiskCheckResult.REJECTED for c in checks)
        has_warning = any(c.result == RiskCheckResult.WARNING for c in checks)

        if has_rejection:
            overall = RiskCheckResult.REJECTED
            self._total_rejections += 1
        elif has_warning:
            overall = RiskCheckResult.WARNING
        else:
            overall = RiskCheckResult.APPROVED

        elapsed_ns = time.perf_counter_ns() - start
        total_latency_us = elapsed_ns / 1000.0

        self._check_latencies.append(total_latency_us)
        self._total_checks += 1

        # Update state if approved
        if overall != RiskCheckResult.REJECTED:
            self._record_order(symbol, side, notional)

        response = RiskCheckResponse(
            order_id=order_id,
            overall_result=overall,
            checks=checks,
            total_latency_us=total_latency_us,
        )

        if overall == RiskCheckResult.REJECTED:
            logger.warning("order_rejected", order_id=order_id, symbol=symbol,
                          reasons=[c.message for c in checks if c.result == RiskCheckResult.REJECTED])

        return response

    def _timed_check(self, name: str, check_fn) -> PreTradeCheck:
        """Run a check and measure its latency."""
        start = time.perf_counter_ns()
        result, message, utilization = check_fn()
        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck(name, result, message, elapsed, utilization)

    def _check_position_size(self, symbol: str, notional: float, side: str) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_position_pct"]
        current_position = self._positions.get(symbol, 0.0)

        if side == "BUY":
            new_position = current_position + notional
        else:
            new_position = max(0, current_position - notional)

        if self._current_equity > 0:
            position_pct = new_position / self._current_equity
        else:
            position_pct = 0.0

        limit.current_value = position_pct

        if position_pct > limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Position {position_pct:.1%} exceeds limit {limit.max_value:.1%}"
        elif position_pct > limit.max_value * 0.8:
            result = RiskCheckResult.WARNING
            msg = f"Position {position_pct:.1%} approaching limit {limit.max_value:.1%}"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Position {position_pct:.1%} within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("position_size", result, msg, elapsed, limit.utilization)

    def _check_order_size(self, notional: float) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_single_order_pct"]

        if self._current_equity > 0:
            order_pct = notional / self._current_equity
        else:
            order_pct = 0.0

        limit.current_value = order_pct

        if order_pct > limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Order size {order_pct:.1%} exceeds limit {limit.max_value:.1%}"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Order size {order_pct:.1%} within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("order_size", result, msg, elapsed, limit.utilization)

    def _check_daily_notional(self, notional: float) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_daily_notional"]
        projected = self._daily_notional + notional
        limit.current_value = projected

        if projected > limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Daily notional ${projected:,.0f} exceeds limit ${limit.max_value:,.0f}"
        elif projected > limit.max_value * 0.9:
            result = RiskCheckResult.WARNING
            msg = f"Daily notional ${projected:,.0f} approaching limit"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Daily notional ${projected:,.0f} within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("daily_notional", result, msg, elapsed, limit.utilization)

    def _check_daily_trades(self) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_daily_trades"]
        projected = self._daily_trades + 1
        limit.current_value = float(projected)

        if projected > limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Daily trades {projected} exceeds limit {int(limit.max_value)}"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Daily trades {projected} within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("daily_trades", result, msg, elapsed, limit.utilization)

    def _check_daily_loss(self) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_daily_loss_pct"]

        if self._current_equity > 0:
            loss_pct = abs(min(0, self._daily_pnl)) / self._current_equity
        else:
            loss_pct = 0.0

        limit.current_value = loss_pct

        if loss_pct > limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Daily loss {loss_pct:.1%} exceeds limit {limit.max_value:.1%}"
        elif loss_pct > limit.max_value * 0.8:
            result = RiskCheckResult.WARNING
            msg = f"Daily loss {loss_pct:.1%} approaching limit"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Daily loss {loss_pct:.1%} within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("daily_loss", result, msg, elapsed, limit.utilization)

    def _check_drawdown(self) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_drawdown_pct"]

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
        else:
            drawdown = 0.0

        limit.current_value = drawdown

        if drawdown > limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Drawdown {drawdown:.1%} exceeds limit {limit.max_value:.1%} - CIRCUIT BREAKER"
        elif drawdown > limit.max_value * 0.7:
            result = RiskCheckResult.WARNING
            msg = f"Drawdown {drawdown:.1%} approaching limit"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Drawdown {drawdown:.1%} within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("drawdown", result, msg, elapsed, limit.utilization)

    def _check_concentration(self, symbol: str, notional: float, side: str) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_concentration_pct"]
        total_exposure = sum(self._positions.values())

        current = self._positions.get(symbol, 0.0)
        if side == "BUY":
            new_position = current + notional
        else:
            new_position = max(0, current - notional)

        new_total = total_exposure - current + new_position
        if new_total > 0:
            concentration = new_position / new_total
        else:
            concentration = 0.0

        limit.current_value = concentration

        if concentration > limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Concentration {concentration:.1%} exceeds limit {limit.max_value:.1%}"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Concentration {concentration:.1%} within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("concentration", result, msg, elapsed, limit.utilization)

    def _check_rate_limit(self) -> PreTradeCheck:
        start = time.perf_counter_ns()
        limit = self.limits["max_orders_per_minute"]
        now = time.time()

        # Count orders in last 60 seconds
        recent = sum(1 for t in self._order_timestamps if now - t < 60)
        limit.current_value = float(recent)

        if recent >= limit.max_value:
            result = RiskCheckResult.REJECTED
            msg = f"Order rate {recent}/min exceeds limit {int(limit.max_value)}/min"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Order rate {recent}/min within limit"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("rate_limit", result, msg, elapsed, limit.utilization)

    def _check_fat_finger(self, symbol: str, price: float) -> PreTradeCheck:
        start = time.perf_counter_ns()
        cached_price = self._price_cache.get(symbol)

        if cached_price and cached_price > 0 and price > 0:
            deviation = abs(price - cached_price) / cached_price
            if deviation > 0.10:  # 10% deviation
                result = RiskCheckResult.REJECTED
                msg = f"Price ${price:,.2f} deviates {deviation:.1%} from market ${cached_price:,.2f}"
            elif deviation > 0.05:
                result = RiskCheckResult.WARNING
                msg = f"Price deviation {deviation:.1%} - verify order"
            else:
                result = RiskCheckResult.APPROVED
                msg = "Price within normal range"
        else:
            result = RiskCheckResult.APPROVED
            msg = "No reference price - skipping fat finger check"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("fat_finger", result, msg, elapsed, 0.0)

    def _check_volatility(self, symbol: str, notional: float) -> PreTradeCheck:
        start = time.perf_counter_ns()
        vol = self._volatility_cache.get(symbol, 0.0)

        if vol > 0.05:  # High volatility (>5% daily)
            adjusted_max = self.limits["max_single_order_pct"].max_value * 0.5
            if self._current_equity > 0:
                order_pct = notional / self._current_equity
                if order_pct > adjusted_max:
                    result = RiskCheckResult.WARNING
                    msg = f"High vol ({vol:.1%}): order size reduced limit to {adjusted_max:.1%}"
                else:
                    result = RiskCheckResult.APPROVED
                    msg = f"Order within vol-adjusted limit"
            else:
                result = RiskCheckResult.APPROVED
                msg = "No equity reference"
        else:
            result = RiskCheckResult.APPROVED
            msg = f"Volatility normal ({vol:.1%})"

        elapsed = (time.perf_counter_ns() - start) / 1000.0
        return PreTradeCheck("volatility_adjust", result, msg, elapsed, 0.0)

    def _record_order(self, symbol: str, side: str, notional: float) -> None:
        """Record order for state tracking."""
        self._order_timestamps.append(time.time())
        self._daily_notional += notional
        self._daily_trades += 1

        if side == "BUY":
            self._positions[symbol] = self._positions.get(symbol, 0.0) + notional
        else:
            self._positions[symbol] = max(0, self._positions.get(symbol, 0.0) - notional)

    def update_equity(self, equity: float) -> None:
        """Update current equity for risk calculations."""
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    def update_pnl(self, pnl: float) -> None:
        """Update daily P&L."""
        self._daily_pnl = pnl

    def update_price(self, symbol: str, price: float) -> None:
        """Update cached price for fat finger checks."""
        self._price_cache[symbol] = price

    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update cached volatility."""
        self._volatility_cache[symbol] = volatility

    def reset_daily(self) -> None:
        """Reset daily counters (call at market open or midnight UTC)."""
        self._daily_notional = 0.0
        self._daily_trades = 0
        self._daily_pnl = 0.0
        logger.info("daily_risk_counters_reset")

    def get_stats(self) -> dict:
        """Get risk engine performance statistics."""
        latencies = list(self._check_latencies)
        return {
            "total_checks": self._total_checks,
            "total_rejections": self._total_rejections,
            "rejection_rate": self._total_rejections / max(self._total_checks, 1),
            "latency_stats": {
                "mean_us": float(np.mean(latencies)) if latencies else 0,
                "p50_us": float(np.percentile(latencies, 50)) if latencies else 0,
                "p95_us": float(np.percentile(latencies, 95)) if latencies else 0,
                "p99_us": float(np.percentile(latencies, 99)) if latencies else 0,
                "max_us": float(np.max(latencies)) if latencies else 0,
            },
            "current_state": {
                "equity": self._current_equity,
                "peak_equity": self._peak_equity,
                "daily_notional": self._daily_notional,
                "daily_trades": self._daily_trades,
                "daily_pnl": self._daily_pnl,
                "positions": dict(self._positions),
            },
            "limits": {
                name: {
                    "max": limit.max_value,
                    "current": limit.current_value,
                    "utilization": round(limit.utilization, 4),
                    "breached": limit.is_breached,
                }
                for name, limit in self.limits.items()
            },
        }


# Module-level instance
realtime_risk_engine = RealTimeRiskEngine()

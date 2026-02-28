"""Drawdown circuit breaker: halts trading when losses exceed limits.

State is persisted to Redis so it survives server restarts.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from app.core.logging import get_logger

logger = get_logger(__name__)

REDIS_KEY = "trademaster:circuit_breaker"


class CircuitBreakerState(StrEnum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"  # 50% position sizes
    PAUSED = "PAUSED"  # No new trades
    HALTED = "HALTED"  # Full stop, manual restart required


@dataclass
class DrawdownSnapshot:
    """Tracks P&L for drawdown calculation."""

    timestamp: datetime
    pnl: float


class DrawdownCircuitBreaker:
    """Monitors drawdowns and halts trading when limits are breached.

    Thresholds (configurable):
    - Daily drawdown > 3% -> PAUSED
    - Weekly drawdown > 7% -> PAUSED
    - Monthly drawdown > 10% -> REDUCED (50% sizes)
    - Total drawdown > 15% -> HALTED (manual restart)
    """

    def __init__(
        self,
        max_daily_drawdown: float = 0.03,
        max_weekly_drawdown: float = 0.07,
        max_monthly_drawdown: float = 0.10,
        max_total_drawdown: float = 0.15,
    ):
        self.max_daily_drawdown = max_daily_drawdown
        self.max_weekly_drawdown = max_weekly_drawdown
        self.max_monthly_drawdown = max_monthly_drawdown
        self.max_total_drawdown = max_total_drawdown

        self._state = CircuitBreakerState.NORMAL
        self._peak_equity: float = 0
        self._initial_equity: float = 0
        self._pnl_history: list[DrawdownSnapshot] = []
        self._halted_manually: bool = False

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def can_trade(self) -> bool:
        return self._state in (CircuitBreakerState.NORMAL, CircuitBreakerState.REDUCED)

    @property
    def position_size_multiplier(self) -> float:
        """Returns multiplier for position sizes based on current state."""
        if self._state == CircuitBreakerState.REDUCED:
            return 0.5
        if self._state in (CircuitBreakerState.PAUSED, CircuitBreakerState.HALTED):
            return 0.0
        return 1.0

    def initialize(self, equity: float) -> None:
        """Set initial equity level."""
        self._initial_equity = equity
        self._peak_equity = equity
        self._state = CircuitBreakerState.NORMAL
        self._halted_manually = False
        self._pnl_history = [
            DrawdownSnapshot(timestamp=datetime.now(timezone.utc), pnl=0.0)
        ]

    async def restore_from_redis(self) -> bool:
        """Restore circuit breaker state from Redis. Returns True if restored."""
        try:
            from app.core.events import event_bus
            if not event_bus._redis:
                return False

            raw = await event_bus._redis.get(REDIS_KEY)
            if not raw:
                return False

            data = json.loads(raw)
            self._state = CircuitBreakerState(data["state"])
            self._peak_equity = data["peak_equity"]
            self._initial_equity = data["initial_equity"]
            self._halted_manually = data.get("halted_manually", False)

            # Restore P&L history
            self._pnl_history = [
                DrawdownSnapshot(
                    timestamp=datetime.fromisoformat(s["timestamp"]),
                    pnl=s["pnl"],
                )
                for s in data.get("pnl_history", [])
            ]
            self._cleanup_old_history()

            logger.info(
                "circuit_breaker_restored",
                state=self._state,
                peak_equity=self._peak_equity,
                history_points=len(self._pnl_history),
            )
            return True
        except Exception as e:
            logger.warning("circuit_breaker_restore_failed", error=str(e))
            return False

    async def _save_to_redis(self) -> None:
        """Persist current state to Redis."""
        try:
            from app.core.events import event_bus
            if not event_bus._redis:
                return

            # Keep only last 2000 snapshots to avoid huge Redis keys
            history = self._pnl_history[-2000:]
            data = {
                "state": str(self._state),
                "peak_equity": self._peak_equity,
                "initial_equity": self._initial_equity,
                "halted_manually": self._halted_manually,
                "pnl_history": [
                    {"timestamp": s.timestamp.isoformat(), "pnl": s.pnl}
                    for s in history
                ],
            }
            await event_bus._redis.set(REDIS_KEY, json.dumps(data), ex=86400 * 31)
        except Exception as e:
            logger.warning("circuit_breaker_save_failed", error=str(e))

    def update(self, current_equity: float) -> CircuitBreakerState:
        """Update with current equity and check all drawdown limits.

        Should be called on every portfolio update.
        """
        if self._halted_manually:
            return CircuitBreakerState.HALTED

        now = datetime.now(timezone.utc)
        self._peak_equity = max(self._peak_equity, current_equity)

        # Record P&L snapshot
        pnl = current_equity - self._initial_equity
        self._pnl_history.append(DrawdownSnapshot(timestamp=now, pnl=pnl))
        self._cleanup_old_history()

        # Check total drawdown from peak
        if self._peak_equity > 0:
            total_dd = (self._peak_equity - current_equity) / self._peak_equity
        else:
            total_dd = 0

        # Check period drawdowns
        daily_dd = self._period_drawdown(hours=24)
        weekly_dd = self._period_drawdown(hours=168)
        monthly_dd = self._period_drawdown(hours=720)

        prev_state = self._state

        # Evaluate thresholds (most severe first)
        if total_dd >= self.max_total_drawdown:
            self._state = CircuitBreakerState.HALTED
        elif daily_dd >= self.max_daily_drawdown or weekly_dd >= self.max_weekly_drawdown:
            self._state = CircuitBreakerState.PAUSED
        elif monthly_dd >= self.max_monthly_drawdown:
            self._state = CircuitBreakerState.REDUCED
        elif self._state != CircuitBreakerState.HALTED:
            # Allow recovery from PAUSED/REDUCED but not from HALTED
            self._state = CircuitBreakerState.NORMAL

        if self._state != prev_state:
            logger.warning(
                "circuit_breaker_state_change",
                prev=prev_state,
                new=self._state,
                total_dd=round(total_dd, 4),
                daily_dd=round(daily_dd, 4),
                weekly_dd=round(weekly_dd, 4),
            )

        return self._state

    async def update_and_persist(self, current_equity: float) -> CircuitBreakerState:
        """Update drawdown state and persist to Redis."""
        state = self.update(current_equity)
        await self._save_to_redis()
        return state

    def manual_reset(self, new_equity: float) -> None:
        """Manually reset the circuit breaker (after HALTED state)."""
        self._halted_manually = False
        self.initialize(new_equity)
        logger.info("circuit_breaker_manual_reset", equity=new_equity)

    def _period_drawdown(self, hours: int) -> float:
        """Calculate drawdown over the last N hours."""
        if not self._pnl_history:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        period_snapshots = [s for s in self._pnl_history if s.timestamp >= cutoff]

        if not period_snapshots:
            return 0

        peak_pnl = max(s.pnl for s in period_snapshots)
        current_pnl = period_snapshots[-1].pnl

        if self._initial_equity <= 0:
            return 0

        return max(0, (peak_pnl - current_pnl) / self._initial_equity)

    def _cleanup_old_history(self) -> None:
        """Remove P&L snapshots older than 31 days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=31)
        self._pnl_history = [s for s in self._pnl_history if s.timestamp >= cutoff]

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "state": self._state,
            "circuit_breaker_state": self._state,
            "can_trade": self.can_trade,
            "position_size_multiplier": self.position_size_multiplier,
            "peak_equity": self._peak_equity,
            "daily_drawdown": round(self._period_drawdown(24), 4),
            "weekly_drawdown": round(self._period_drawdown(168), 4),
            "monthly_drawdown": round(self._period_drawdown(720), 4),
            "max_drawdown": round(
                (self._peak_equity - (self._peak_equity * (1 - self._period_drawdown(720)))) / self._peak_equity
                if self._peak_equity > 0 else 0, 4
            ),
        }


circuit_breaker = DrawdownCircuitBreaker()

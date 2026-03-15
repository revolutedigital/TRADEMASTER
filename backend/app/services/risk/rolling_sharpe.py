"""Rolling Sharpe monitor: tracks live trading performance and auto-pauses if degraded.

Computes Sharpe ratio from the last N closed trades. If rolling Sharpe drops
below 0 for a sustained period, signals that trading should be paused.

This is a live risk guardrail — different from drift detection (which monitors
model accuracy). Rolling Sharpe monitors actual P&L quality.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

ROLLING_WINDOW = 30  # Last 30 trades
MIN_TRADES_FOR_CHECK = 10  # Need at least 10 trades
SHARPE_PAUSE_THRESHOLD = -0.5  # Pause if Sharpe < -0.5
SHARPE_RESUME_THRESHOLD = 0.3  # Resume only when Sharpe > 0.3
PAUSE_COOLDOWN_SECONDS = 3600  # Min 1 hour pause


@dataclass
class TradeReturn:
    """Record of a single trade's return."""

    symbol: str
    return_pct: float
    side: str
    timestamp: float


@dataclass
class RollingSharpeStatus:
    """Current rolling Sharpe status."""

    sharpe: float
    is_paused: bool
    trades_in_window: int
    avg_return_pct: float
    win_rate: float
    pause_reason: str
    pause_since: float | None  # Timestamp when paused
    consecutive_negative_sharpe: int


class RollingSharpeMonitor:
    """Monitors rolling Sharpe ratio and auto-pauses trading when degraded."""

    def __init__(self) -> None:
        self._trades: deque[TradeReturn] = deque(maxlen=ROLLING_WINDOW * 2)
        self._is_paused: bool = False
        self._pause_since: float | None = None
        self._consecutive_negative: int = 0
        self._last_check_sharpe: float = 0.0

    def record_trade(self, symbol: str, return_pct: float, side: str) -> None:
        """Record a completed trade return."""
        self._trades.append(TradeReturn(
            symbol=symbol,
            return_pct=return_pct,
            side=side,
            timestamp=time.time(),
        ))

    def check(self) -> RollingSharpeStatus:
        """Check rolling Sharpe and determine if trading should be paused.

        Returns current status with pause recommendation.
        """
        recent = list(self._trades)[-ROLLING_WINDOW:]

        if len(recent) < MIN_TRADES_FOR_CHECK:
            return RollingSharpeStatus(
                sharpe=0.0, is_paused=self._is_paused,
                trades_in_window=len(recent),
                avg_return_pct=0.0, win_rate=0.0,
                pause_reason="insufficient_trades",
                pause_since=self._pause_since,
                consecutive_negative_sharpe=0,
            )

        returns = np.array([t.return_pct for t in recent])
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = mean_ret / max(std_ret, 1e-8) * np.sqrt(252)  # Annualized

        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns)

        self._last_check_sharpe = sharpe

        # Track consecutive negative Sharpe checks
        if sharpe < 0:
            self._consecutive_negative += 1
        else:
            self._consecutive_negative = 0

        now = time.time()
        pause_reason = ""

        # Pause logic
        if not self._is_paused:
            if sharpe < SHARPE_PAUSE_THRESHOLD and self._consecutive_negative >= 3:
                self._is_paused = True
                self._pause_since = now
                pause_reason = f"sharpe={sharpe:.2f} < {SHARPE_PAUSE_THRESHOLD} for 3+ checks"
                logger.warning(
                    "rolling_sharpe_pause",
                    sharpe=round(sharpe, 4),
                    win_rate=round(win_rate, 4),
                    avg_return=round(mean_ret * 100, 4),
                    consecutive_negative=self._consecutive_negative,
                )
        else:
            # Resume logic
            pause_duration = now - (self._pause_since or now)
            if sharpe > SHARPE_RESUME_THRESHOLD and pause_duration > PAUSE_COOLDOWN_SECONDS:
                self._is_paused = False
                self._pause_since = None
                self._consecutive_negative = 0
                pause_reason = "resumed"
                logger.info(
                    "rolling_sharpe_resumed",
                    sharpe=round(sharpe, 4),
                    pause_duration_min=round(pause_duration / 60, 1),
                )
            else:
                pause_reason = f"paused (sharpe={sharpe:.2f}, need > {SHARPE_RESUME_THRESHOLD})"

        return RollingSharpeStatus(
            sharpe=round(sharpe, 4),
            is_paused=self._is_paused,
            trades_in_window=len(recent),
            avg_return_pct=round(mean_ret * 100, 4),
            win_rate=round(win_rate, 4),
            pause_reason=pause_reason,
            pause_since=self._pause_since,
            consecutive_negative_sharpe=self._consecutive_negative,
        )

    @property
    def is_paused(self) -> bool:
        """Quick check if trading is currently paused."""
        return self._is_paused

    def get_status(self) -> dict:
        """Get full status for API."""
        status = self.check()
        return {
            "sharpe": status.sharpe,
            "is_paused": status.is_paused,
            "trades_in_window": status.trades_in_window,
            "avg_return_pct": status.avg_return_pct,
            "win_rate": status.win_rate,
            "pause_reason": status.pause_reason,
            "pause_since": status.pause_since,
            "consecutive_negative_sharpe": status.consecutive_negative_sharpe,
        }

    def force_resume(self) -> None:
        """Manually resume trading (override auto-pause)."""
        self._is_paused = False
        self._pause_since = None
        self._consecutive_negative = 0
        logger.info("rolling_sharpe_force_resumed")


rolling_sharpe_monitor = RollingSharpeMonitor()

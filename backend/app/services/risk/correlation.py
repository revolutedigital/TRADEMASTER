"""Cross-asset correlation filter: blocks highly correlated same-direction positions.

If BTC is LONG, don't also go LONG ETH (correlation ~0.85).
Uses rolling return correlation from DB candles + known defaults.
"""

from collections import deque
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.market import OHLCV
from app.models.portfolio import Position

logger = get_logger(__name__)

# Known high-correlation pairs (fallback when insufficient data)
_DEFAULT_CORRELATIONS: dict[tuple[str, str], float] = {
    ("BTCUSDT", "ETHUSDT"): 0.85,
    ("BNBUSDT", "BTCUSDT"): 0.75,
    ("BTCUSDT", "SOLUSDT"): 0.80,
    ("BTCUSDT", "XRPUSDT"): 0.70,
    ("ADAUSDT", "BTCUSDT"): 0.72,
    ("AVAXUSDT", "BTCUSDT"): 0.78,
    ("BTCUSDT", "DOTUSDT"): 0.74,
    ("BTCUSDT", "LINKUSDT"): 0.76,
    ("BNBUSDT", "ETHUSDT"): 0.72,
    ("ETHUSDT", "SOLUSDT"): 0.80,
    ("AVAXUSDT", "ETHUSDT"): 0.76,
    ("DOGEUSDT", "BTCUSDT"): 0.65,
}

# Maximum correlation allowed for same-direction positions
MAX_CORRELATION_SAME_DIRECTION = 0.70
MIN_CANDLES_FOR_CORRELATION = 30


class CorrelationFilter:
    """Checks if a new position would be too correlated with existing ones."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._returns: dict[str, deque] = {}
        # Cache correlations for 15 minutes to avoid DB spam
        self._cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._cache_ttl = 900  # 15 min

    def record_return(self, symbol: str, daily_return: float) -> None:
        """Record a daily return for in-memory tracking."""
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self._window)
        self._returns[symbol].append(daily_return)

    async def check_can_open(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
    ) -> tuple[bool, str | None]:
        """Check if opening this position violates correlation limits.

        Returns:
            (is_allowed, reason) — reason is None if allowed.
        """
        result = await db.execute(
            select(Position).where(Position.is_open == True)
        )
        open_positions = list(result.scalars().all())

        if not open_positions:
            return True, None

        for pos in open_positions:
            if pos.symbol == symbol:
                continue

            # Only block same-direction correlated positions
            pos_direction = "BUY" if pos.side == "LONG" else "SELL"
            if pos_direction != side:
                continue  # Opposite direction = natural hedge, allow

            corr = await self._get_correlation(db, symbol, pos.symbol)

            if corr >= MAX_CORRELATION_SAME_DIRECTION:
                reason = (
                    f"Correlation block: {symbol} {side} corr={corr:.2f} "
                    f"with open {pos.symbol} {pos.side}"
                )
                logger.info(
                    "correlation_filter_blocked",
                    symbol=symbol,
                    side=side,
                    correlated_with=pos.symbol,
                    correlation=round(corr, 3),
                )
                return False, reason

        return True, None

    async def _get_correlation(
        self,
        db: AsyncSession,
        symbol_a: str,
        symbol_b: str,
    ) -> float:
        """Get correlation between two symbols (cached)."""
        import time

        key = tuple(sorted([symbol_a, symbol_b]))
        now = time.time()

        # Check cache
        if key in self._cache:
            value, cached_at = self._cache[key]
            if now - cached_at < self._cache_ttl:
                return value

        corr = await self._compute_correlation(db, symbol_a, symbol_b)
        self._cache[key] = (corr, now)
        return corr

    async def _compute_correlation(
        self,
        db: AsyncSession,
        symbol_a: str,
        symbol_b: str,
    ) -> float:
        """Compute rolling return correlation from DB candles."""
        closes_a = await self._fetch_closes(db, symbol_a)
        closes_b = await self._fetch_closes(db, symbol_b)

        if (
            len(closes_a) < MIN_CANDLES_FOR_CORRELATION
            or len(closes_b) < MIN_CANDLES_FOR_CORRELATION
        ):
            return self._default_correlation(symbol_a, symbol_b)

        min_len = min(len(closes_a), len(closes_b))
        a = np.array(closes_a[-min_len:])
        b = np.array(closes_b[-min_len:])

        if len(a) < 2:
            return self._default_correlation(symbol_a, symbol_b)

        returns_a = np.diff(a) / a[:-1]
        returns_b = np.diff(b) / b[:-1]

        if np.std(returns_a) == 0 or np.std(returns_b) == 0:
            return 0.0

        corr = float(np.corrcoef(returns_a, returns_b)[0, 1])
        if np.isnan(corr):
            return self._default_correlation(symbol_a, symbol_b)

        return abs(corr)

    async def _fetch_closes(self, db: AsyncSession, symbol: str) -> list[float]:
        """Fetch last 100 close prices (15m candles)."""
        result = await db.execute(
            select(OHLCV.close)
            .where(OHLCV.symbol == symbol, OHLCV.interval == "15m")
            .order_by(OHLCV.open_time.desc())
            .limit(100)
        )
        rows = result.all()
        return [float(r[0]) for r in reversed(rows)]

    @staticmethod
    def _default_correlation(symbol_a: str, symbol_b: str) -> float:
        """Look up default correlation for known pairs."""
        key = tuple(sorted([symbol_a, symbol_b]))
        return _DEFAULT_CORRELATIONS.get(key, 0.5)

    def get_rolling_correlation(self, symbol_a: str, symbol_b: str) -> float | None:
        """In-memory rolling correlation (for API display)."""
        if symbol_a not in self._returns or symbol_b not in self._returns:
            return None

        a = np.array(self._returns[symbol_a])
        b = np.array(self._returns[symbol_b])

        min_len = min(len(a), len(b))
        if min_len < 10:
            return None

        a = a[-min_len:]
        b = b[-min_len:]

        corr = float(np.corrcoef(a, b)[0, 1])
        return corr if not np.isnan(corr) else None

    def check_concentration_risk(self, threshold: float = 0.85) -> dict:
        """Check if any asset pairs have dangerously high correlation."""
        symbols = list(self._returns.keys())
        high_correlations = []

        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                corr = self.get_rolling_correlation(sym_a, sym_b)
                if corr is not None and abs(corr) > threshold:
                    high_correlations.append({
                        "pair": f"{sym_a}-{sym_b}",
                        "correlation": round(corr, 4),
                        "risk": "high" if abs(corr) > 0.95 else "moderate",
                    })

        return {
            "has_concentration_risk": len(high_correlations) > 0,
            "high_correlations": high_correlations,
            "threshold": threshold,
        }

    def get_correlation_matrix(self) -> dict:
        """Get full correlation matrix for all tracked symbols."""
        symbols = sorted(self._returns.keys())
        matrix = {}
        for sym_a in symbols:
            row = {}
            for sym_b in symbols:
                if sym_a == sym_b:
                    row[sym_b] = 1.0
                else:
                    corr = self.get_rolling_correlation(sym_a, sym_b)
                    row[sym_b] = round(corr, 4) if corr is not None else None
            matrix[sym_a] = row
        return matrix


correlation_filter = CorrelationFilter()
# Backwards-compatible alias
correlation_tracker = correlation_filter

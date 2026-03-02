"""Cross-asset correlation tracking for portfolio risk management."""

from datetime import datetime, timezone
from collections import deque

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class CorrelationTracker:
    """Tracks rolling correlation between assets for concentration risk detection."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._returns: dict[str, deque] = {}

    def record_return(self, symbol: str, daily_return: float) -> None:
        """Record a daily return for a symbol."""
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self._window)
        self._returns[symbol].append(daily_return)

    def get_rolling_correlation(self, symbol_a: str, symbol_b: str) -> float | None:
        """Calculate rolling Pearson correlation between two assets."""
        if symbol_a not in self._returns or symbol_b not in self._returns:
            return None

        a = np.array(self._returns[symbol_a])
        b = np.array(self._returns[symbol_b])

        min_len = min(len(a), len(b))
        if min_len < 10:
            return None

        a = a[-min_len:]
        b = b[-min_len:]

        correlation = float(np.corrcoef(a, b)[0, 1])
        return correlation

    def check_concentration_risk(self, threshold: float = 0.85) -> dict:
        """Check if any asset pairs have dangerously high correlation.

        If BTC-ETH correlation > threshold, they should be treated
        as a single exposure for risk management.
        """
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


correlation_tracker = CorrelationTracker()

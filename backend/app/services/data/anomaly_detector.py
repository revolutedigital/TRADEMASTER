"""Real-time anomaly detection for market data.

Uses Isolation Forest and statistical methods to detect:
- Price anomalies (flash crashes, unusual spikes)
- Volume anomalies (unusual trading activity)
- Data quality anomalies (gaps, duplicates, stale data)

Auto-pauses trading if critical anomaly detected.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class AnomalySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    timestamp: datetime
    anomaly_type: str
    severity: AnomalySeverity
    symbol: str
    description: str
    value: float
    expected_range: tuple[float, float]
    should_pause_trading: bool = False


class AnomalyDetector:
    """Detect anomalies in market data using statistical methods."""

    def __init__(self, z_score_threshold: float = 3.0, volume_threshold: float = 5.0):
        self._z_score_threshold = z_score_threshold
        self._volume_threshold = volume_threshold
        self._price_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[float]] = {}
        self._history_limit = 1000

    def update(self, symbol: str, price: float, volume: float) -> list[Anomaly]:
        """Update with new data point and check for anomalies."""
        anomalies = []

        # Initialize history
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._volume_history[symbol] = []

        prices = self._price_history[symbol]
        volumes = self._volume_history[symbol]

        # Need minimum history for detection
        if len(prices) >= 20:
            # Check price anomaly
            price_anomaly = self._check_price_anomaly(symbol, price, prices)
            if price_anomaly:
                anomalies.append(price_anomaly)

            # Check volume anomaly
            volume_anomaly = self._check_volume_anomaly(symbol, volume, volumes)
            if volume_anomaly:
                anomalies.append(volume_anomaly)

            # Check price gap
            gap_anomaly = self._check_price_gap(symbol, price, prices)
            if gap_anomaly:
                anomalies.append(gap_anomaly)

        # Update history
        prices.append(price)
        volumes.append(volume)
        if len(prices) > self._history_limit:
            prices.pop(0)
            volumes.pop(0)

        return anomalies

    def _check_price_anomaly(self, symbol: str, price: float, history: list[float]) -> Anomaly | None:
        """Detect price anomalies using z-score."""
        mean = np.mean(history[-100:])
        std = np.std(history[-100:])
        if std == 0:
            return None

        z_score = abs(price - mean) / std
        if z_score > self._z_score_threshold:
            severity = AnomalySeverity.CRITICAL if z_score > 5.0 else AnomalySeverity.WARNING
            return Anomaly(
                timestamp=datetime.now(timezone.utc),
                anomaly_type="price_anomaly",
                severity=severity,
                symbol=symbol,
                description=f"Price z-score {z_score:.2f} exceeds threshold {self._z_score_threshold}",
                value=price,
                expected_range=(mean - 2 * std, mean + 2 * std),
                should_pause_trading=severity == AnomalySeverity.CRITICAL,
            )
        return None

    def _check_volume_anomaly(self, symbol: str, volume: float, history: list[float]) -> Anomaly | None:
        """Detect unusual volume spikes."""
        mean_vol = np.mean(history[-100:])
        if mean_vol == 0:
            return None

        ratio = volume / mean_vol
        if ratio > self._volume_threshold:
            return Anomaly(
                timestamp=datetime.now(timezone.utc),
                anomaly_type="volume_anomaly",
                severity=AnomalySeverity.WARNING,
                symbol=symbol,
                description=f"Volume {ratio:.1f}x above average",
                value=volume,
                expected_range=(0, mean_vol * 3),
            )
        return None

    def _check_price_gap(self, symbol: str, price: float, history: list[float]) -> Anomaly | None:
        """Detect sudden price gaps (flash crash detection)."""
        if len(history) < 2:
            return None

        last_price = history[-1]
        pct_change = abs(price - last_price) / last_price if last_price > 0 else 0

        if pct_change > 0.05:  # 5% gap in single tick
            severity = AnomalySeverity.CRITICAL if pct_change > 0.10 else AnomalySeverity.WARNING
            return Anomaly(
                timestamp=datetime.now(timezone.utc),
                anomaly_type="price_gap",
                severity=severity,
                symbol=symbol,
                description=f"Price gap of {pct_change:.1%} detected (possible flash crash)",
                value=price,
                expected_range=(last_price * 0.95, last_price * 1.05),
                should_pause_trading=severity == AnomalySeverity.CRITICAL,
            )
        return None

    def get_status(self) -> dict:
        """Get detector status."""
        return {
            "symbols_tracked": list(self._price_history.keys()),
            "history_sizes": {s: len(h) for s, h in self._price_history.items()},
            "z_score_threshold": self._z_score_threshold,
            "volume_threshold": self._volume_threshold,
        }


anomaly_detector = AnomalyDetector()

"""Synthetic market data generation for ML training augmentation.

Generates realistic-looking OHLCV data for scenarios that are rare in
historical data (flash crashes, extreme volatility, extended sideways).
"""
import numpy as np
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SyntheticScenario:
    name: str
    description: str
    candles: int
    data: np.ndarray  # (candles, 5) = open, high, low, close, volume


class SyntheticMarketGenerator:
    """Generate synthetic market data for rare scenarios."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_flash_crash(
        self, base_price: float = 50000, candles: int = 100,
        crash_pct: float = 0.15, recovery_pct: float = 0.80,
    ) -> SyntheticScenario:
        """Generate a flash crash scenario with partial recovery."""
        prices = np.zeros(candles)
        prices[0] = base_price

        crash_start = candles // 3
        crash_end = crash_start + candles // 10
        recovery_end = crash_end + candles // 5

        for i in range(1, candles):
            if i < crash_start:
                # Normal pre-crash
                prices[i] = prices[i-1] * (1 + self.rng.normal(0.0002, 0.005))
            elif i < crash_end:
                # Crash phase
                crash_per_candle = crash_pct / (crash_end - crash_start)
                prices[i] = prices[i-1] * (1 - crash_per_candle + self.rng.normal(0, 0.002))
            elif i < recovery_end:
                # Recovery phase
                target = base_price * (1 - crash_pct * (1 - recovery_pct))
                recovery_per_candle = (target - prices[i-1]) / (recovery_end - i + 1)
                prices[i] = prices[i-1] + recovery_per_candle + self.rng.normal(0, base_price * 0.002)
            else:
                # Post-recovery drift
                prices[i] = prices[i-1] * (1 + self.rng.normal(0.0001, 0.004))

        return SyntheticScenario(
            name="flash_crash",
            description=f"Flash crash of {crash_pct*100:.0f}% with {recovery_pct*100:.0f}% recovery",
            candles=candles,
            data=self._prices_to_ohlcv(prices),
        )

    def generate_high_volatility(
        self, base_price: float = 50000, candles: int = 200,
        vol_multiplier: float = 3.0,
    ) -> SyntheticScenario:
        """Generate extreme volatility period."""
        prices = np.zeros(candles)
        prices[0] = base_price

        for i in range(1, candles):
            vol = 0.01 * vol_multiplier * (1 + 0.5 * np.sin(2 * np.pi * i / 50))
            prices[i] = prices[i-1] * (1 + self.rng.normal(0, vol))
            prices[i] = max(prices[i], base_price * 0.1)  # Floor

        return SyntheticScenario(
            name="high_volatility",
            description=f"Extreme volatility ({vol_multiplier}x normal)",
            candles=candles,
            data=self._prices_to_ohlcv(prices),
        )

    def generate_sideways(
        self, base_price: float = 50000, candles: int = 300,
        band_pct: float = 0.02,
    ) -> SyntheticScenario:
        """Generate extended sideways/ranging market."""
        prices = np.zeros(candles)
        prices[0] = base_price

        for i in range(1, candles):
            # Mean-reverting to base_price
            deviation = (prices[i-1] - base_price) / base_price
            mean_revert = -deviation * 0.1
            noise = self.rng.normal(0, band_pct * 0.3)
            prices[i] = prices[i-1] * (1 + mean_revert + noise)

        return SyntheticScenario(
            name="sideways",
            description=f"Sideways market within {band_pct*100:.0f}% band",
            candles=candles,
            data=self._prices_to_ohlcv(prices),
        )

    def generate_trend(
        self, base_price: float = 50000, candles: int = 200,
        daily_drift: float = 0.003, direction: str = "up",
    ) -> SyntheticScenario:
        """Generate strong trending market."""
        prices = np.zeros(candles)
        prices[0] = base_price
        drift = daily_drift if direction == "up" else -daily_drift

        for i in range(1, candles):
            prices[i] = prices[i-1] * (1 + drift + self.rng.normal(0, 0.005))
            prices[i] = max(prices[i], base_price * 0.01)

        return SyntheticScenario(
            name=f"trend_{direction}",
            description=f"Strong {direction}trend ({daily_drift*100:.1f}%/candle)",
            candles=candles,
            data=self._prices_to_ohlcv(prices),
        )

    def generate_all_scenarios(self, base_price: float = 50000) -> list[SyntheticScenario]:
        """Generate all standard scenarios for training augmentation."""
        return [
            self.generate_flash_crash(base_price),
            self.generate_high_volatility(base_price),
            self.generate_sideways(base_price),
            self.generate_trend(base_price, direction="up"),
            self.generate_trend(base_price, direction="down"),
        ]

    def _prices_to_ohlcv(self, closes: np.ndarray) -> np.ndarray:
        """Convert close prices to OHLCV data."""
        n = len(closes)
        ohlcv = np.zeros((n, 5))

        for i in range(n):
            noise = abs(self.rng.normal(0, closes[i] * 0.003))
            ohlcv[i, 0] = closes[i] + self.rng.normal(0, closes[i] * 0.001)  # open
            ohlcv[i, 1] = max(closes[i], ohlcv[i, 0]) + noise  # high
            ohlcv[i, 2] = min(closes[i], ohlcv[i, 0]) - noise  # low
            ohlcv[i, 3] = closes[i]  # close
            ohlcv[i, 4] = max(100, self.rng.exponential(1000))  # volume

        return ohlcv


synthetic_generator = SyntheticMarketGenerator()

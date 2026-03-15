"""Kelly Criterion for optimal position sizing."""

from dataclasses import dataclass
from decimal import Decimal

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KellyResult:
    full_kelly: float          # Full Kelly fraction
    half_kelly: float          # Conservative half-Kelly
    quarter_kelly: float       # Ultra-conservative quarter-Kelly
    optimal_size_usd: float    # Dollar amount at half-Kelly
    edge: float                # Expected edge per trade
    recommended_fraction: float  # What we actually recommend (capped)


class KellyCalculator:
    """Compute optimal bet size using the Kelly Criterion.

    Full Kelly: f* = (p * b - q) / b
    Where: p = win_rate, q = 1-p, b = avg_win / avg_loss (win/loss ratio)

    We use half-Kelly by default as full Kelly is too aggressive for trading.
    """

    MAX_FRACTION = 0.25  # Never risk more than 25% of portfolio

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio_value: float,
    ) -> KellyResult:
        """Calculate Kelly fraction and optimal position size.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade P&L (positive)
            avg_loss: Average losing trade P&L (positive, absolute value)
            portfolio_value: Current portfolio value in USD
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return KellyResult(
                full_kelly=0,
                half_kelly=0,
                quarter_kelly=0,
                optimal_size_usd=0,
                edge=0,
                recommended_fraction=0,
            )

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss  # Win/loss ratio

        # Kelly fraction: f* = (p * b - q) / b
        full = (p * b - q) / b

        # Edge: expected value per dollar risked
        edge = p * b - q

        if full <= 0:
            # Negative edge — don't trade
            logger.warning("kelly_negative_edge", win_rate=win_rate, ratio=b, edge=edge)
            return KellyResult(
                full_kelly=full,
                half_kelly=0,
                quarter_kelly=0,
                optimal_size_usd=0,
                edge=edge,
                recommended_fraction=0,
            )

        half = full * 0.5
        quarter = full * 0.25

        # Cap at MAX_FRACTION
        recommended = min(half, self.MAX_FRACTION)
        optimal_usd = portfolio_value * recommended

        logger.info(
            "kelly_calculated",
            win_rate=round(win_rate, 3),
            ratio=round(b, 3),
            full_kelly=round(full, 4),
            half_kelly=round(half, 4),
            recommended=round(recommended, 4),
            optimal_usd=round(optimal_usd, 2),
        )

        return KellyResult(
            full_kelly=full,
            half_kelly=half,
            quarter_kelly=quarter,
            optimal_size_usd=optimal_usd,
            edge=edge,
            recommended_fraction=recommended,
        )


kelly_calculator = KellyCalculator()

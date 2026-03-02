"""Kelly Criterion for optimal position sizing."""
from decimal import Decimal
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KellyResult:
    full_kelly: float
    fractional_kelly: float
    recommended_size: float
    recommended_size_usd: float
    fraction_used: float


class KellyCalculator:
    """Calculate optimal position sizes using the Kelly Criterion.

    Full Kelly is theoretically optimal but has high variance.
    Fractional Kelly (default 0.5) reduces variance significantly
    while retaining ~75% of the growth rate.
    """

    def full_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate full Kelly fraction.

        Kelly% = W/L - (1-W)/A
        where W=win_rate, L=avg_loss, A=avg_win
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        kelly = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
        return max(0.0, kelly)

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio_value: float,
        fraction: float = 0.5,
    ) -> KellyResult:
        """Calculate recommended position size."""
        full = self.full_kelly_fraction(win_rate, avg_win, avg_loss)
        fractional = full * fraction
        size_usd = portfolio_value * max(0, fractional)

        result = KellyResult(
            full_kelly=round(full, 4),
            fractional_kelly=round(fractional, 4),
            recommended_size=round(fractional, 4),
            recommended_size_usd=round(size_usd, 2),
            fraction_used=fraction,
        )

        logger.info(
            "kelly_calculated",
            win_rate=win_rate,
            full_kelly=result.full_kelly,
            fractional=result.fractional_kelly,
            size_usd=result.recommended_size_usd,
        )
        return result


kelly_calculator = KellyCalculator()

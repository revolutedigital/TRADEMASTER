"""OHLCV data validation pipeline.

Validates incoming market data before database insertion to ensure
data integrity and detect anomalies.
"""

from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

# Maximum allowed single-candle price change (20%)
MAX_PRICE_CHANGE_PCT = 0.20

# Maximum allowed volume spike multiplier vs rolling average
MAX_VOLUME_SPIKE = 50.0


@dataclass
class ValidationResult:
    """Result of OHLCV validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """Validates OHLCV candle data for integrity and anomalies."""

    def validate_ohlcv(self, data: dict) -> ValidationResult:
        """Validate a single OHLCV candle.

        Checks:
        - OHLCV relationship constraints (high >= low, etc.)
        - Non-negative values
        - Timestamp ordering
        - Reasonable price ranges
        """
        errors: list[str] = []
        warnings: list[str] = []

        open_price = float(data.get("open", 0))
        high = float(data.get("high", 0))
        low = float(data.get("low", 0))
        close = float(data.get("close", 0))
        volume = float(data.get("volume", 0))

        # Basic OHLCV constraints
        if high < low:
            errors.append(f"high ({high}) < low ({low})")

        if high < open_price:
            errors.append(f"high ({high}) < open ({open_price})")

        if high < close:
            errors.append(f"high ({high}) < close ({close})")

        if low > open_price:
            errors.append(f"low ({low}) > open ({open_price})")

        if low > close:
            errors.append(f"low ({low}) > close ({close})")

        # Non-negative checks
        if close <= 0:
            errors.append(f"non-positive close price: {close}")

        if open_price <= 0:
            errors.append(f"non-positive open price: {open_price}")

        if volume < 0:
            errors.append(f"negative volume: {volume}")

        if volume == 0:
            warnings.append("zero volume candle")

        # Timestamp checks
        open_time = data.get("open_time")
        close_time = data.get("close_time")
        if open_time and close_time and open_time >= close_time:
            errors.append(f"open_time ({open_time}) >= close_time ({close_time})")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_price_continuity(
        self,
        current_close: float,
        previous_close: float,
    ) -> ValidationResult:
        """Check for suspicious price gaps between consecutive candles."""
        errors: list[str] = []
        warnings: list[str] = []

        if previous_close <= 0:
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        change_pct = abs(current_close - previous_close) / previous_close

        if change_pct > MAX_PRICE_CHANGE_PCT:
            warnings.append(
                f"Large price gap: {change_pct:.1%} change "
                f"({previous_close} -> {current_close})"
            )

        return ValidationResult(
            is_valid=True,
            errors=errors,
            warnings=warnings,
        )

    def validate_volume_spike(
        self,
        current_volume: float,
        avg_volume: float,
    ) -> ValidationResult:
        """Check for suspicious volume spikes."""
        warnings: list[str] = []

        if avg_volume > 0 and current_volume > avg_volume * MAX_VOLUME_SPIKE:
            warnings.append(
                f"Volume spike: {current_volume:.0f} is "
                f"{current_volume / avg_volume:.0f}x the average ({avg_volume:.0f})"
            )

        return ValidationResult(is_valid=True, errors=[], warnings=warnings)


# Module-level singleton
data_validator = DataValidator()

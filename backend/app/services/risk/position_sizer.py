"""Position sizing: Fractional Kelly, fixed fraction, volatility-scaled."""

import math
from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSize:
    """Result of a position sizing calculation."""

    quantity: float
    notional_value: float  # quantity * price
    risk_amount: float  # capital at risk (quantity * |entry - stop|)
    risk_pct: float  # risk as % of total equity
    method: str


class PositionSizer:
    """Calculates optimal position sizes for trades."""

    def __init__(
        self,
        max_risk_per_trade: float = 0.02,
        max_single_asset_exposure: float = 0.30,
        kelly_fraction: float = 0.15,
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_single_asset_exposure = max_single_asset_exposure
        self.kelly_fraction = kelly_fraction

    def fractional_kelly(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        price: float,
        stop_distance_pct: float,
    ) -> PositionSize:
        """Fractional Kelly criterion position sizing.

        Kelly % = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss
        We use kelly_fraction (15%) of full Kelly for safety.
        """
        if avg_loss == 0 or equity <= 0 or price <= 0:
            return PositionSize(0, 0, 0, 0, "fractional_kelly")

        r = avg_win / abs(avg_loss)
        full_kelly = win_rate - (1 - win_rate) / r

        # Clamp: Kelly can be negative (don't trade) or very large
        full_kelly = max(0, min(full_kelly, 1.0))
        kelly_pct = full_kelly * self.kelly_fraction

        # Cap at max risk per trade
        risk_pct = min(kelly_pct, self.max_risk_per_trade)
        risk_amount = equity * risk_pct

        # Position size from risk amount and stop distance
        if stop_distance_pct <= 0:
            return PositionSize(0, 0, 0, 0, "fractional_kelly")

        notional = risk_amount / stop_distance_pct
        # Cap at max single asset exposure
        max_notional = equity * self.max_single_asset_exposure
        notional = min(notional, max_notional)

        quantity = notional / price

        return PositionSize(
            quantity=quantity,
            notional_value=notional,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method="fractional_kelly",
        )

    def fixed_fraction(
        self,
        equity: float,
        price: float,
        stop_distance_pct: float,
    ) -> PositionSize:
        """Fixed fraction (2% risk) position sizing.

        Simpler alternative when Kelly inputs aren't available.
        """
        if equity <= 0 or price <= 0 or stop_distance_pct <= 0:
            return PositionSize(0, 0, 0, 0, "fixed_fraction")

        risk_amount = equity * self.max_risk_per_trade
        notional = risk_amount / stop_distance_pct
        max_notional = equity * self.max_single_asset_exposure
        notional = min(notional, max_notional)
        quantity = notional / price

        return PositionSize(
            quantity=quantity,
            notional_value=notional,
            risk_amount=risk_amount,
            risk_pct=self.max_risk_per_trade,
            method="fixed_fraction",
        )

    def volatility_scaled(
        self,
        equity: float,
        price: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> PositionSize:
        """ATR-based volatility-scaled position sizing.

        stop_distance = atr * multiplier
        """
        if equity <= 0 or price <= 0 or atr <= 0:
            return PositionSize(0, 0, 0, 0, "volatility_scaled")

        stop_distance = atr * atr_multiplier
        stop_distance_pct = stop_distance / price

        return self.fixed_fraction(equity, price, stop_distance_pct)


position_sizer = PositionSizer()

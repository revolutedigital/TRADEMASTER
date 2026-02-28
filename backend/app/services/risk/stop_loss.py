"""Stop loss strategies: ATR-based, trailing, time-based."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StopLossLevel:
    """Calculated stop loss and take profit levels."""

    stop_price: float
    take_profit_price: float | None
    trailing_active: bool
    method: str


class StopLossCalculator:
    """Calculates stop loss and take profit levels for positions."""

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        trailing_activation_pct: float = 0.015,
        trailing_distance_pct: float = 0.01,
        risk_reward_ratio: float = 2.0,
        time_exit_hours: int = 1,
    ):
        self.atr_multiplier = atr_multiplier
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.time_exit_hours = time_exit_hours

    def atr_based(
        self,
        entry_price: float,
        atr: float,
        side: str,
    ) -> StopLossLevel:
        """Calculate ATR-based stop loss and take profit.

        Stop: entry +/- (atr * multiplier)
        TP: entry +/- (atr * multiplier * risk_reward_ratio)
        """
        stop_distance = atr * self.atr_multiplier
        tp_distance = stop_distance * self.risk_reward_ratio

        if side == "LONG":
            stop_price = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_price = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        return StopLossLevel(
            stop_price=stop_price,
            take_profit_price=take_profit,
            trailing_active=False,
            method="atr_based",
        )

    def percentage_based(
        self,
        entry_price: float,
        stop_pct: float,
        side: str,
    ) -> StopLossLevel:
        """Simple percentage-based stop loss."""
        if side == "LONG":
            stop_price = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + stop_pct * self.risk_reward_ratio)
        else:
            stop_price = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - stop_pct * self.risk_reward_ratio)

        return StopLossLevel(
            stop_price=stop_price,
            take_profit_price=take_profit,
            trailing_active=False,
            method="percentage_based",
        )

    def update_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        side: str,
    ) -> float:
        """Update trailing stop if price has moved favorably.

        Trailing activates when profit exceeds trailing_activation_pct.
        Then trails at trailing_distance_pct from the best price.
        """
        if side == "LONG":
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self.trailing_activation_pct:
                new_stop = current_price * (1 - self.trailing_distance_pct)
                return max(current_stop, new_stop)
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= self.trailing_activation_pct:
                new_stop = current_price * (1 + self.trailing_distance_pct)
                return min(current_stop, new_stop)

        return current_stop

    def should_time_exit(self, opened_at: datetime) -> bool:
        """Check if position should be closed due to time limit."""
        elapsed = datetime.now(timezone.utc) - opened_at
        return elapsed > timedelta(hours=self.time_exit_hours)

    def is_stop_hit(
        self,
        current_price: float,
        stop_price: float,
        side: str,
    ) -> bool:
        """Check if stop loss has been triggered."""
        if side == "LONG":
            return current_price <= stop_price
        return current_price >= stop_price

    def is_take_profit_hit(
        self,
        current_price: float,
        take_profit_price: float,
        side: str,
    ) -> bool:
        """Check if take profit has been triggered."""
        if side == "LONG":
            return current_price >= take_profit_price
        return current_price <= take_profit_price


stop_loss_calculator = StopLossCalculator()

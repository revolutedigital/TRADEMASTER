"""Risk management gate: every trade must pass through here before execution."""

from dataclasses import dataclass

from app.config import settings
from app.core.exceptions import RiskLimitExceededError, DrawdownCircuitBreakerError
from app.core.logging import get_logger
from app.services.risk.drawdown import circuit_breaker, CircuitBreakerState
from app.services.risk.position_sizer import PositionSizer, PositionSize
from app.services.risk.stop_loss import StopLossCalculator, StopLossLevel

logger = get_logger(__name__)

# Binance minimum notional value
MIN_NOTIONAL_USDT = 10.0


@dataclass
class TradeProposal:
    """A proposed trade that needs risk validation."""

    symbol: str
    side: str  # BUY or SELL
    signal_strength: float  # -1.0 to +1.0
    entry_price: float
    atr: float  # Current ATR(14)
    current_equity: float
    current_exposure: float  # Total notional of open positions
    symbol_exposure: float  # Notional of open positions for this symbol


@dataclass
class ApprovedTrade:
    """A trade that passed all risk checks."""

    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_loss: StopLossLevel
    position_size: PositionSize
    risk_checks_passed: list[str]


class RiskManager:
    """Pre-trade validation chain. ALL checks must pass before execution.

    Checks:
    1. Circuit breaker state
    2. Position sizing (fractional Kelly / fixed fraction)
    3. Exposure limits (single asset + total portfolio)
    4. Minimum notional value
    5. Stop loss assignment
    """

    def __init__(self):
        self._position_sizer = PositionSizer(
            max_risk_per_trade=settings.trading_max_risk_per_trade,
            max_single_asset_exposure=settings.trading_max_single_asset_exposure,
            kelly_fraction=0.15,
        )
        self._stop_loss_calc = StopLossCalculator()

    def validate_trade(self, proposal: TradeProposal) -> ApprovedTrade:
        """Run all risk checks on a trade proposal.

        Raises:
            DrawdownCircuitBreakerError: If circuit breaker is triggered.
            RiskLimitExceededError: If any risk limit is breached.
        """
        checks_passed = []

        # 1. Circuit breaker check
        cb_state = circuit_breaker.update(proposal.current_equity)
        if cb_state == CircuitBreakerState.HALTED:
            raise DrawdownCircuitBreakerError(
                f"Trading HALTED. Circuit breaker state: {cb_state}"
            )
        if cb_state == CircuitBreakerState.PAUSED:
            raise DrawdownCircuitBreakerError(
                f"Trading PAUSED. Circuit breaker state: {cb_state}"
            )
        checks_passed.append("circuit_breaker")

        # 2. Position sizing
        position_size = self._position_sizer.volatility_scaled(
            equity=proposal.current_equity,
            price=proposal.entry_price,
            atr=proposal.atr,
        )

        # Apply circuit breaker multiplier (REDUCED = 50%)
        multiplier = circuit_breaker.position_size_multiplier
        position_size.quantity *= multiplier
        position_size.notional_value *= multiplier
        position_size.risk_amount *= multiplier

        if position_size.quantity <= 0:
            raise RiskLimitExceededError("Position size calculated as zero")
        checks_passed.append("position_sizing")

        # 3. Exposure check - single asset
        new_symbol_exposure = proposal.symbol_exposure + position_size.notional_value
        max_symbol = proposal.current_equity * settings.trading_max_single_asset_exposure
        if new_symbol_exposure > max_symbol:
            raise RiskLimitExceededError(
                f"Single asset exposure {new_symbol_exposure:.2f} exceeds max {max_symbol:.2f}"
            )
        checks_passed.append("single_asset_exposure")

        # 4. Exposure check - total portfolio
        new_total_exposure = proposal.current_exposure + position_size.notional_value
        max_total = proposal.current_equity * settings.trading_max_portfolio_exposure
        if new_total_exposure > max_total:
            raise RiskLimitExceededError(
                f"Total exposure {new_total_exposure:.2f} exceeds max {max_total:.2f}"
            )
        checks_passed.append("total_exposure")

        # 5. Minimum notional check
        if position_size.notional_value < MIN_NOTIONAL_USDT:
            raise RiskLimitExceededError(
                f"Notional {position_size.notional_value:.2f} below minimum {MIN_NOTIONAL_USDT}"
            )
        checks_passed.append("min_notional")

        # 6. Calculate stop loss
        trade_side = "LONG" if proposal.side == "BUY" else "SHORT"
        stop_loss = self._stop_loss_calc.atr_based(
            entry_price=proposal.entry_price,
            atr=proposal.atr,
            side=trade_side,
        )
        checks_passed.append("stop_loss_assigned")

        logger.info(
            "trade_approved",
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=round(position_size.quantity, 8),
            risk_pct=round(position_size.risk_pct, 4),
            stop=round(stop_loss.stop_price, 2),
            tp=round(stop_loss.take_profit_price, 2) if stop_loss.take_profit_price else None,
            checks=checks_passed,
        )

        return ApprovedTrade(
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=position_size.quantity,
            entry_price=proposal.entry_price,
            stop_loss=stop_loss,
            position_size=position_size,
            risk_checks_passed=checks_passed,
        )


risk_manager = RiskManager()

"""Adaptive risk management: adjust parameters based on market regime."""
from dataclasses import dataclass

from app.core.logging import get_logger
from app.services.ml.regime import MarketRegime

logger = get_logger(__name__)


@dataclass
class AdaptiveRiskParams:
    max_exposure: float
    position_size_multiplier: float
    stop_loss_multiplier: float
    max_concurrent_positions: int
    regime: str
    reason: str


class AdaptiveRiskManager:
    """Adjust risk parameters dynamically based on detected market regime."""

    REGIME_PARAMS = {
        MarketRegime.BULL: {
            "max_exposure": 0.80,
            "position_size_multiplier": 1.2,
            "stop_loss_multiplier": 1.0,
            "max_concurrent_positions": 4,
        },
        MarketRegime.BEAR: {
            "max_exposure": 0.30,
            "position_size_multiplier": 0.5,
            "stop_loss_multiplier": 0.7,
            "max_concurrent_positions": 2,
        },
        MarketRegime.SIDEWAYS: {
            "max_exposure": 0.50,
            "position_size_multiplier": 0.8,
            "stop_loss_multiplier": 0.9,
            "max_concurrent_positions": 3,
        },
        MarketRegime.HIGH_VOLATILITY: {
            "max_exposure": 0.20,
            "position_size_multiplier": 0.3,
            "stop_loss_multiplier": 0.5,
            "max_concurrent_positions": 1,
        },
    }

    def get_params(self, regime: MarketRegime) -> AdaptiveRiskParams:
        """Get risk parameters adjusted for the current market regime."""
        params = self.REGIME_PARAMS.get(regime, self.REGIME_PARAMS[MarketRegime.SIDEWAYS])

        result = AdaptiveRiskParams(
            max_exposure=params["max_exposure"],
            position_size_multiplier=params["position_size_multiplier"],
            stop_loss_multiplier=params["stop_loss_multiplier"],
            max_concurrent_positions=params["max_concurrent_positions"],
            regime=regime.value,
            reason=f"Adjusted for {regime.value} market conditions",
        )

        logger.info("adaptive_risk_params", regime=regime.value, exposure=result.max_exposure)
        return result


adaptive_risk_manager = AdaptiveRiskManager()

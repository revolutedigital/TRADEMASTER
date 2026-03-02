"""Feature flags for gradual feature rollout."""
from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureFlags:
    """Simple in-memory feature flag system.

    Flags can be toggled at runtime via the admin API.
    """

    def __init__(self):
        self._flags: dict[str, bool] = {
            "multi_exchange": False,
            "sentiment_analysis": False,
            "twap_execution": False,
            "portfolio_optimizer": True,
            "price_alerts": True,
            "trading_journal": True,
            "tax_reporting": True,
            "smart_order_routing": False,
            "advanced_risk_metrics": True,
        }

    def is_enabled(self, flag: str) -> bool:
        return self._flags.get(flag, False)

    def enable(self, flag: str) -> None:
        self._flags[flag] = True
        logger.info("feature_flag_enabled", flag=flag)

    def disable(self, flag: str) -> None:
        self._flags[flag] = False
        logger.info("feature_flag_disabled", flag=flag)

    def list_flags(self) -> dict[str, bool]:
        return dict(self._flags)

    def toggle(self, flag: str) -> bool:
        current = self._flags.get(flag, False)
        self._flags[flag] = not current
        logger.info("feature_flag_toggled", flag=flag, enabled=not current)
        return not current


feature_flags = FeatureFlags()

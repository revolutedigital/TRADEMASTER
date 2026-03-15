"""Runtime feature flags for gradual feature rollout."""

from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureFlags:
    """Simple feature flag system backed by an in-memory dict.

    Flags can be toggled at runtime via the admin API.
    In production, this would be backed by Redis for multi-instance consistency.
    """

    _flags: dict[str, bool] = {
        "multi_exchange": False,
        "sentiment_analysis": False,
        "twap_execution": False,
        "portfolio_optimizer": True,
        "monte_carlo_risk": True,
        "kelly_sizing": True,
        "advanced_alerts": True,
        "dca_automation": False,
    }

    def is_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled."""
        return self._flags.get(flag, False)

    def enable(self, flag: str) -> None:
        """Enable a feature flag."""
        self._flags[flag] = True
        logger.info("feature_flag_enabled", flag=flag)

    def disable(self, flag: str) -> None:
        """Disable a feature flag."""
        self._flags[flag] = False
        logger.info("feature_flag_disabled", flag=flag)

    def get_all(self) -> dict[str, bool]:
        """Return all flags and their states."""
        return dict(self._flags)

    def set_flag(self, flag: str, enabled: bool) -> None:
        """Set a flag to a specific state."""
        self._flags[flag] = enabled
        logger.info("feature_flag_set", flag=flag, enabled=enabled)

    def list_flags(self) -> dict[str, bool]:
        """Alias for get_all — backward compat with admin API."""
        return self.get_all()

    def toggle(self, flag: str) -> bool:
        """Toggle a flag and return the new state."""
        current = self._flags.get(flag, False)
        self._flags[flag] = not current
        logger.info("feature_flag_toggled", flag=flag, enabled=not current)
        return not current


feature_flags = FeatureFlags()

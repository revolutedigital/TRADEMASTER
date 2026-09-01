"""Fail-closed control plane for Binance Spot live execution."""

import hmac
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from app.config import Settings, TradingExecutionMode, settings
from app.services.exchange.live_execution_readiness import (
    live_protection_readiness,
    testnet_protection_readiness,
)


class LiveTradingSafetyError(Exception):
    """Raised when a mandatory live-execution safety gate is not satisfied."""


class LiveTradingGuard:
    """Keeps real-capital execution locked unless an operator arms it briefly."""

    def __init__(
        self,
        app_settings: Settings | None = None,
        totp_validator: Callable[[str, str], bool] | None = None,
        native_protection_ready: Callable[[], bool] | None = None,
    ) -> None:
        self._settings = app_settings or settings
        self._totp_validator = totp_validator or self._verify_totp
        # Production requires a recent Testnet release proof and a fresh live
        # account reconciliation. Injection keeps the arming mechanics testable.
        self._native_protection_ready = native_protection_ready or (
            lambda: (
                testnet_protection_readiness.is_ready(
                    self._settings.live_trading_testnet_verification_max_age_days * 86_400
                )
                and live_protection_readiness.is_ready(
                    self._settings.live_trading_reconciliation_max_age_seconds
                )
            )
        )
        self._armed_until: datetime | None = None
        self._armed_by: str | None = None
        self._disarm_reason = "not armed"

    @staticmethod
    def _verify_totp(secret: str, code: str) -> bool:
        from app.core.totp import TOTPManager

        return TOTPManager.verify_totp(secret, code)

    def _expired(self) -> bool:
        return self._armed_until is None or datetime.now(UTC) >= self._armed_until

    @property
    def is_armed(self) -> bool:
        if self._expired():
            if self._armed_until is not None:
                self.disarm("arming window expired")
            return False
        return True

    def _blockers(self) -> list[str]:
        blockers: list[str] = []
        if self._settings.execution_mode != TradingExecutionMode.LIVE:
            blockers.append("execution mode is not LIVE")
        if not self._settings.live_trading_enabled:
            blockers.append("LIVE_TRADING_ENABLED is false")
        if not self._settings.is_production:
            blockers.append("APP_ENV must be production")
        if not self._settings.totp_enabled or not self._settings.totp_secret:
            blockers.append("TOTP is not configured")
        if len(self._settings.live_trading_arm_code) < 20:
            blockers.append("LIVE_TRADING_ARM_CODE is not configured")
        if not self._settings.binance_api_key or not self._settings.binance_api_secret:
            blockers.append("Binance live API credentials are not configured")
        if (
            self._settings.execution_mode == TradingExecutionMode.LIVE
            and not self._native_protection_ready()
        ):
            blockers.append(
                "Binance Spot testnet protection verification is required before production release"
            )
        return blockers

    def status(self) -> dict[str, object]:
        blockers = self._blockers()
        armed = self.is_armed
        return {
            "execution_mode": self._settings.execution_mode.value,
            "live_enabled": self._settings.live_trading_enabled,
            "armed": armed,
            "armed_until": self._armed_until.isoformat() if armed and self._armed_until else None,
            "armable": not blockers,
            "blockers": blockers,
            "max_notional_per_order": self._settings.live_trading_max_notional_per_order,
            "max_daily_notional": self._settings.live_trading_max_daily_notional,
            "reconciliation": live_protection_readiness.status(
                self._settings.live_trading_reconciliation_max_age_seconds
            ),
            "testnet_verification": testnet_protection_readiness.status(
                self._settings.live_trading_testnet_verification_max_age_days * 86_400
            ),
        }

    def arm(self, *, actor: str, arm_code: str, totp_code: str) -> dict[str, object]:
        blockers = self._blockers()
        if blockers:
            raise LiveTradingSafetyError("; ".join(blockers))
        if not hmac.compare_digest(arm_code, self._settings.live_trading_arm_code):
            raise LiveTradingSafetyError("invalid live trading arm code")
        if not self._totp_validator(self._settings.totp_secret, totp_code):
            raise LiveTradingSafetyError("invalid TOTP code")

        self._armed_by = actor
        self._armed_until = datetime.now(UTC) + timedelta(
            minutes=self._settings.live_trading_arm_ttl_minutes
        )
        self._disarm_reason = ""
        return self.status()

    def disarm(self, reason: str) -> dict[str, object]:
        self._armed_until = None
        self._armed_by = None
        self._disarm_reason = reason
        return self.status()

    def require_live_order(self, notional: float) -> None:
        if self._settings.execution_mode != TradingExecutionMode.LIVE:
            return
        blockers = self._blockers()
        if blockers:
            raise LiveTradingSafetyError("; ".join(blockers))
        if not self.is_armed:
            raise LiveTradingSafetyError("live execution is not armed")
        if notional > self._settings.live_trading_max_notional_per_order:
            raise LiveTradingSafetyError(
                "order notional exceeds LIVE_TRADING_MAX_NOTIONAL_PER_ORDER"
            )

    def require_live_exit(self, totp_code: str) -> None:
        """Authorize an operator-initiated exit without requiring an entry arm.

        A stale or disarmed entry session must never prevent reducing existing
        market exposure. TOTP still protects the destructive exchange action.
        """
        if self._settings.execution_mode != TradingExecutionMode.LIVE:
            raise LiveTradingSafetyError("exchange close is only available in LIVE mode")
        if not self._settings.is_production:
            raise LiveTradingSafetyError("exchange close requires APP_ENV=production")
        if not self._settings.totp_enabled or not self._settings.totp_secret:
            raise LiveTradingSafetyError("TOTP is not configured")
        if not self._settings.binance_api_key or not self._settings.binance_api_secret:
            raise LiveTradingSafetyError("Binance live API credentials are not configured")
        if not self._totp_validator(self._settings.totp_secret, totp_code):
            raise LiveTradingSafetyError("invalid TOTP code")

    def require_live_strategy_exit(self) -> None:
        """Allow an armed strategy to reduce one already protected Spot long.

        This deliberately authorizes only an exit after the same readiness and
        short-lived arming controls used for a LIVE entry. It is not a general
        SELL permission and must be paired with the exact native OCO position
        checks in ``SpotPositionCloser``.
        """
        blockers = self._blockers()
        if blockers:
            raise LiveTradingSafetyError("; ".join(blockers))
        if not self.is_armed:
            raise LiveTradingSafetyError("live strategy exit is not armed")


live_trading_guard = LiveTradingGuard()

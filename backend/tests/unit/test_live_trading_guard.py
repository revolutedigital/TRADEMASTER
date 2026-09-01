"""Tests for the fail-closed live-execution safety control plane."""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services.exchange.live_trading_guard import (
    LiveTradingGuard,
    LiveTradingSafetyError,
)


def _live_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "production",
        "jwt_secret_key": "j" * 32,
        "admin_password": "strong-admin-password",
        "paper_mode": False,
        "binance_testnet": False,
        "binance_api_key": "live-key",
        "binance_api_secret": "live-secret",
        "totp_enabled": True,
        "totp_secret": "JBSWY3DPEHPK3PXP",
        "live_trading_enabled": True,
        "live_trading_arm_code": "a" * 20,
        "live_trading_max_notional_per_order": 100.0,
        "live_trading_max_daily_notional": 300.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_default_configuration_is_paper_and_not_armable() -> None:
    guard = LiveTradingGuard(Settings(_env_file=None))

    status = guard.status()

    assert status["execution_mode"] == "PAPER"
    assert status["armed"] is False
    assert status["armable"] is False
    assert "execution mode is not LIVE" in status["blockers"]


def test_live_configuration_requires_all_static_safety_prerequisites() -> None:
    with pytest.raises(ValidationError, match="requires TOTP_ENABLED"):
        _live_settings(totp_enabled=False, totp_secret="")


def test_live_order_is_rejected_until_operator_arms_the_session() -> None:
    guard = LiveTradingGuard(
        _live_settings(),
        totp_validator=lambda _secret, _code: True,
        native_protection_ready=lambda: True,
    )

    with pytest.raises(LiveTradingSafetyError, match="not armed"):
        guard.require_live_order(10.0)


def test_arming_requires_the_server_code_and_a_valid_second_factor() -> None:
    guard = LiveTradingGuard(
        _live_settings(),
        totp_validator=lambda _secret, code: code == "123456",
        native_protection_ready=lambda: True,
    )

    with pytest.raises(LiveTradingSafetyError, match="invalid live trading arm code"):
        guard.arm(actor="igor", arm_code="b" * 20, totp_code="123456")

    with pytest.raises(LiveTradingSafetyError, match="invalid TOTP code"):
        guard.arm(actor="igor", arm_code="a" * 20, totp_code="000000")

    status = guard.arm(actor="igor", arm_code="a" * 20, totp_code="123456")

    assert status["armed"] is True
    assert status["armed_until"] is not None


def test_live_order_respects_the_absolute_per_order_cap() -> None:
    guard = LiveTradingGuard(
        _live_settings(),
        totp_validator=lambda _secret, _code: True,
        native_protection_ready=lambda: True,
    )
    guard.arm(actor="igor", arm_code="a" * 20, totp_code="123456")

    guard.require_live_order(100.0)
    with pytest.raises(LiveTradingSafetyError, match="MAX_NOTIONAL_PER_ORDER"):
        guard.require_live_order(100.01)


def test_disarm_revokes_permission_immediately() -> None:
    guard = LiveTradingGuard(
        _live_settings(),
        totp_validator=lambda _secret, _code: True,
        native_protection_ready=lambda: True,
    )
    guard.arm(actor="igor", arm_code="a" * 20, totp_code="123456")

    status = guard.disarm("operator requested stop")

    assert status["armed"] is False
    with pytest.raises(LiveTradingSafetyError, match="not armed"):
        guard.require_live_order(10.0)


def test_live_mode_cannot_arm_without_native_exchange_protection() -> None:
    guard = LiveTradingGuard(_live_settings(), totp_validator=lambda _secret, _code: True)

    status = guard.status()

    assert status["armable"] is False
    assert (
        "Binance Spot testnet protection verification is required before production release"
        in status["blockers"]
    )


def test_live_exit_requires_totp_but_not_an_entry_arming_window() -> None:
    guard = LiveTradingGuard(
        _live_settings(),
        totp_validator=lambda _secret, code: code == "123456",
        native_protection_ready=lambda: False,
    )

    guard.require_live_exit("123456")

    with pytest.raises(LiveTradingSafetyError, match="invalid TOTP code"):
        guard.require_live_exit("000000")


def test_strategy_exit_requires_the_same_armed_live_session_as_an_entry() -> None:
    guard = LiveTradingGuard(
        _live_settings(),
        totp_validator=lambda _secret, _code: True,
        native_protection_ready=lambda: True,
    )

    with pytest.raises(LiveTradingSafetyError, match="not armed"):
        guard.require_live_strategy_exit()

    guard.arm(actor="igor", arm_code="a" * 20, totp_code="123456")
    guard.require_live_strategy_exit()

    guard.disarm("operator requested stop")
    with pytest.raises(LiveTradingSafetyError, match="not armed"):
        guard.require_live_strategy_exit()

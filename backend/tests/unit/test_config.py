"""Tests for application configuration."""

from app.config import Settings


def test_default_settings():
    s = Settings(
        binance_api_key="test_key",
        binance_api_secret="test_secret",
        binance_testnet_api_key="test_tn_key",
        binance_testnet_api_secret="test_tn_secret",
    )
    assert s.app_env == "development"
    assert s.binance_testnet is True
    assert s.trading_symbols == "BTCUSDT,ETHUSDT"
    assert s.symbols_list == ["BTCUSDT", "ETHUSDT"]
    assert s.trading_max_risk_per_trade == 0.02


def test_active_api_key_testnet():
    s = Settings(
        binance_testnet=True,
        binance_api_key="main_key",
        binance_testnet_api_key="testnet_key",
        binance_api_secret="main_secret",
        binance_testnet_api_secret="testnet_secret",
    )
    assert s.active_api_key == "testnet_key"
    assert s.active_api_secret == "testnet_secret"


def test_active_api_key_production():
    s = Settings(
        binance_testnet=False,
        binance_api_key="main_key",
        binance_testnet_api_key="testnet_key",
        binance_api_secret="main_secret",
        binance_testnet_api_secret="testnet_secret",
    )
    assert s.active_api_key == "main_key"
    assert s.active_api_secret == "main_secret"


def test_is_production():
    s = Settings(app_env="production")
    assert s.is_production is True

    s = Settings(app_env="development")
    assert s.is_production is False

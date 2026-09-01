"""The settings screen must expose only the limits the engine actually enforces."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.v1.settings import RiskConfig, get_settings, update_risk_settings
from app.config import TradingExecutionMode


@pytest.mark.asyncio
async def test_settings_reads_the_deployed_risk_limits() -> None:
    runtime_settings = SimpleNamespace(
        execution_mode=TradingExecutionMode.TESTNET,
        symbols_list=["BTCUSDT"],
        trading_max_risk_per_trade=0.01,
        trading_max_portfolio_exposure=0.40,
        trading_max_single_asset_exposure=0.20,
        trading_max_daily_drawdown=0.02,
        trading_max_weekly_drawdown=0.05,
        trading_max_monthly_drawdown=0.08,
        trading_max_total_drawdown=0.12,
        trading_kelly_fraction=0.10,
    )

    with patch("app.api.v1.settings.settings", runtime_settings):
        response = await get_settings({"sub": "operator"})

    assert response.trading.max_risk_per_trade == 0.01
    assert response.trading.max_total_exposure == 0.40
    assert response.risk.max_daily_drawdown == 0.02
    assert response.risk.kelly_fraction == 0.10


@pytest.mark.asyncio
async def test_runtime_risk_override_is_rejected() -> None:
    with pytest.raises(HTTPException, match="immutable at runtime") as error:
        await update_risk_settings(
            RiskConfig(),
            {"sub": "operator"},
        )

    assert error.value.status_code == 409

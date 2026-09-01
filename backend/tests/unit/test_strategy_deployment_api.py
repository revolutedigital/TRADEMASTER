"""Safety gates for strategy deployment API actions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.strategy_deployments import activate_deployment
from app.config import TradingExecutionMode
from app.schemas.trading import StrategyDeploymentActivationRequest


@pytest.mark.asyncio
async def test_activation_is_rejected_while_the_trading_engine_is_running() -> None:
    body = StrategyDeploymentActivationRequest(confirmation_phrase="ACTIVATE STRATEGY")
    database = AsyncMock()
    engine = SimpleNamespace(_running=True)
    runtime_settings = SimpleNamespace(execution_mode=TradingExecutionMode.PAPER)

    with (
        patch("app.api.v1.strategy_deployments.settings", runtime_settings),
        patch(
            "app.api.v1.strategy_deployments.activate_strategy_deployment",
            new=AsyncMock(),
        ) as activate_service,
        pytest.raises(HTTPException, match="stop the trading engine") as error,
    ):
        await activate_deployment(
            deployment_id=17,
            body=body,
            request=MagicMock(),
            db=database,
            user={"sub": "operator"},
            engine=engine,
        )

    assert error.value.status_code == 409
    activate_service.assert_not_awaited()

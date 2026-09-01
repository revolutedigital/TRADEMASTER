"""Tests for evidence-gated technical strategy deployments."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from app.config import TradingExecutionMode
from app.services.backtest.walk_forward import WalkForwardResult, WalkForwardWindow
from app.services.strategy_deployments import (
    ACTIVE,
    APPROVED,
    REJECTED,
    ActiveTechnicalStrategy,
    StrategyDeploymentSourceError,
    _source_rejection_reasons,
    activate_strategy_deployment,
    create_strategy_deployment,
    get_active_technical_strategy,
    required_walk_forward_candles,
)


def _source_backtest(**overrides):
    values = {
        "id": 73,
        "symbol": "BTCUSDT",
        "interval": "1h",
        "execution_profile": "spot_long_only",
        "strategy_config_json": json.dumps(
            {
                "kind": "technical_ensemble",
                "indicators": ["sma"],
                "indicator_params": {"sma": {"sma_short": 10, "sma_long": 30}},
                "min_confirmations": 1,
                "execution_profile": "spot_long_only",
            }
        ),
        "total_trades": 24,
        "total_return_pct": 0.08,
        "profit_factor": 1.4,
        "max_drawdown_pct": 0.1,
        "initial_capital": 10_000,
        "signal_threshold": 0.3,
        "atr_stop_multiplier": 2.0,
        "risk_reward_ratio": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _walk_forward_result() -> WalkForwardResult:
    windows = [
        WalkForwardWindow(
            window_idx=idx,
            train_start=0,
            train_end=100,
            test_start=100,
            test_end=150,
            train_trades=10,
            test_trades=8,
            train_return_pct=0.05,
            train_sharpe=1.4,
            test_win_rate=0.6,
            test_return_pct=0.03,
            test_sharpe=1.1,
            test_max_dd_pct=0.08,
            test_profit_factor=1.3,
        )
        for idx in range(3)
    ]
    return WalkForwardResult(
        windows=windows,
        total_test_trades=24,
        avg_win_rate=0.6,
        avg_return_pct=0.03,
        avg_sharpe=1.1,
        avg_max_dd_pct=0.08,
        avg_profit_factor=1.3,
        consistency_score=0.67,
        overfitting_score=0.2,
    )


def _fresh_candles(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0] * count,
            "close_time": pd.date_range(
                end=datetime.now(UTC),
                periods=count,
                freq="h",
            ),
        }
    )


def test_required_walk_forward_history_respects_the_candle_interval() -> None:
    assert required_walk_forward_candles("1h") == 2_520
    assert required_walk_forward_candles("4h") == 630
    assert required_walk_forward_candles("15m") == 10_080


def test_source_evidence_rejects_unfit_metrics() -> None:
    source = _source_backtest(
        total_trades=19,
        total_return_pct=0,
        profit_factor=float("inf"),
        max_drawdown_pct=0.151,
    )

    reasons = _source_rejection_reasons(source)

    assert len(reasons) == 4


@pytest.mark.asyncio
async def test_create_persists_approved_fresh_walk_forward_evidence() -> None:
    database = AsyncMock()
    database.scalar = AsyncMock(return_value=_source_backtest())
    database.add = MagicMock()
    database.flush = AsyncMock()
    candles = _fresh_candles(2_520)

    with (
        patch(
            "app.services.strategy_deployments.market_data_collector.get_latest_candles",
            new=AsyncMock(return_value=candles),
        ),
        patch(
            "app.services.strategy_deployments.run_walk_forward",
            return_value=_walk_forward_result(),
        ),
    ):
        deployment = await create_strategy_deployment(
            database,
            source_backtest_id=73,
            target_execution_mode="PAPER",
        )

    assert deployment.status == APPROVED
    assert deployment.walk_forward_windows == 3
    assert deployment.total_test_trades == 24
    assert json.loads(deployment.execution_config_json)["risk_reward_ratio"] == 2.0
    database.add.assert_called_once_with(deployment)


@pytest.mark.asyncio
async def test_create_persists_rejection_when_fresh_history_is_insufficient() -> None:
    database = AsyncMock()
    database.scalar = AsyncMock(return_value=_source_backtest())
    database.add = MagicMock()
    database.flush = AsyncMock()

    with patch(
        "app.services.strategy_deployments.market_data_collector.get_latest_candles",
        new=AsyncMock(return_value=pd.DataFrame({"close": [100.0] * 100})),
    ):
        deployment = await create_strategy_deployment(
            database,
            source_backtest_id=73,
            target_execution_mode="PAPER",
        )

    assert deployment.status == REJECTED
    assert "Insufficient fresh candle history" in deployment.rejection_reason


@pytest.mark.asyncio
async def test_create_rejects_stale_candle_history_before_walk_forward() -> None:
    database = AsyncMock()
    database.scalar = AsyncMock(return_value=_source_backtest())
    database.add = MagicMock()
    database.flush = AsyncMock()
    stale_candles = _fresh_candles(2_520)
    stale_candles["close_time"] = stale_candles["close_time"] - timedelta(days=1)

    with (
        patch(
            "app.services.strategy_deployments.market_data_collector.get_latest_candles",
            new=AsyncMock(return_value=stale_candles),
        ),
        patch(
            "app.services.strategy_deployments.run_walk_forward",
        ) as run_walk_forward,
    ):
        deployment = await create_strategy_deployment(
            database,
            source_backtest_id=73,
            target_execution_mode="PAPER",
        )

    assert deployment.status == REJECTED
    assert "recent and continuous" in deployment.rejection_reason
    run_walk_forward.assert_not_called()


@pytest.mark.asyncio
async def test_create_rejects_discontinuous_candle_history_before_walk_forward() -> None:
    database = AsyncMock()
    database.scalar = AsyncMock(return_value=_source_backtest())
    database.add = MagicMock()
    database.flush = AsyncMock()
    discontinuous_candles = _fresh_candles(2_520)
    discontinuous_candles.loc[250, "close_time"] = discontinuous_candles.loc[249, "close_time"]

    with (
        patch(
            "app.services.strategy_deployments.market_data_collector.get_latest_candles",
            new=AsyncMock(return_value=discontinuous_candles),
        ),
        patch(
            "app.services.strategy_deployments.run_walk_forward",
        ) as run_walk_forward,
    ):
        deployment = await create_strategy_deployment(
            database,
            source_backtest_id=73,
            target_execution_mode="PAPER",
        )

    assert deployment.status == REJECTED
    assert "recent and continuous" in deployment.rejection_reason
    run_walk_forward.assert_not_called()


@pytest.mark.asyncio
async def test_activation_requires_matching_runtime_mode_and_replaces_old_active() -> None:
    deployment = SimpleNamespace(
        id=11,
        symbol="BTCUSDT",
        interval="1h",
        target_execution_mode="PAPER",
        status=APPROVED,
        activated_at=None,
        deactivated_at=None,
    )
    database = AsyncMock()
    database.scalar = AsyncMock(return_value=deployment)
    database.execute = AsyncMock()

    activated = await activate_strategy_deployment(
        database,
        deployment_id=11,
        execution_mode=TradingExecutionMode.PAPER,
    )

    assert activated.status == ACTIVE
    assert activated.activated_at is not None
    assert database.execute.await_count == 2

    with pytest.raises(StrategyDeploymentSourceError, match="does not match"):
        await activate_strategy_deployment(
            database,
            deployment_id=11,
            execution_mode=TradingExecutionMode.TESTNET,
        )


@pytest.mark.asyncio
async def test_active_runtime_strategy_fails_closed_for_invalid_configuration() -> None:
    deployment = SimpleNamespace(
        id=12,
        strategy_config_json="{not-json}",
        execution_config_json="{}",
    )
    database = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [deployment]
    database.execute = AsyncMock(return_value=result)

    active = await get_active_technical_strategy(
        database,
        symbol="BTCUSDT",
        interval="1h",
        execution_mode=TradingExecutionMode.PAPER,
    )

    assert active is None


@pytest.mark.asyncio
async def test_active_runtime_strategy_restores_the_exact_validated_parameters() -> None:
    deployment = SimpleNamespace(
        id=12,
        strategy_config_json=json.dumps(
            {
                "kind": "technical_ensemble",
                "indicators": ["rsi"],
                "min_confirmations": 1,
                "execution_profile": "spot_long_only",
            }
        ),
        execution_config_json=json.dumps(
            {
                "signal_threshold": 0.4,
                "atr_stop_multiplier": 1.5,
                "risk_reward_ratio": 3.0,
            }
        ),
    )
    database = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [deployment]
    database.execute = AsyncMock(return_value=result)

    active = await get_active_technical_strategy(
        database,
        symbol="BTCUSDT",
        interval="1h",
        execution_mode=TradingExecutionMode.PAPER,
    )

    assert isinstance(active, ActiveTechnicalStrategy)
    assert active.deployment_id == 12
    assert active.strategy.indicators == ["rsi"]
    assert active.signal_threshold == 0.4
    assert active.atr_stop_multiplier == 1.5
    assert active.risk_reward_ratio == 3.0


@pytest.mark.asyncio
async def test_active_runtime_strategy_fails_closed_when_legacy_rows_are_ambiguous() -> None:
    first = SimpleNamespace(id=12)
    second = SimpleNamespace(id=13)
    result = MagicMock()
    result.scalars.return_value.all.return_value = [first, second]
    database = AsyncMock()
    database.execute = AsyncMock(return_value=result)

    active = await get_active_technical_strategy(
        database,
        symbol="BTCUSDT",
        interval="1h",
        execution_mode=TradingExecutionMode.TESTNET,
    )

    assert active is None

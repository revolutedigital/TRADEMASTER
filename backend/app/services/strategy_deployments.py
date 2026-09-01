"""Evidence-gated technical-strategy deployment and runtime selection."""

import json
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode
from app.core.logging import get_logger
from app.models.backtest import BacktestResult
from app.models.portfolio import Position
from app.models.strategy_deployment import StrategyDeployment
from app.schemas.trading import TechnicalStrategyConfig
from app.services.backtest.technical_strategy import build_technical_strategy_signals
from app.services.backtest.walk_forward import candles_per_day, run_walk_forward
from app.services.market.data_collector import market_data_collector
from app.services.market.freshness import has_recent_closed_candle

logger = get_logger(__name__)

APPROVED = "APPROVED"
ACTIVE = "ACTIVE"
REJECTED = "REJECTED"
DISABLED = "DISABLED"

RUNTIME_INTERVALS = frozenset({"15m", "1h", "4h"})
WALK_FORWARD_HISTORY_LIMIT = 12_000
WALK_FORWARD_TRAIN_DAYS = 60
WALK_FORWARD_TEST_DAYS = 15
WALK_FORWARD_STEP_DAYS = 15
MIN_WALK_FORWARD_WINDOWS = 3
MIN_SOURCE_TRADES = 20
MIN_OOS_TRADES = 20
MIN_PROFIT_FACTOR = 1.05
MIN_CONSISTENCY_SCORE = 0.60
MAX_DRAWDOWN_PCT = 0.15
MAX_OVERFITTING_SCORE = 0.50


class StrategyDeploymentSourceError(Exception):
    """Raised when a source backtest can never be deployed safely."""


@dataclass(frozen=True)
class ActiveTechnicalStrategy:
    """Validated configuration the engine may use for a closed candle."""

    deployment_id: int
    strategy: TechnicalStrategyConfig
    signal_threshold: float
    atr_stop_multiplier: float
    risk_reward_ratio: float


def required_walk_forward_candles(interval: str) -> int:
    """Return history needed for three independent 15-day out-of-sample windows."""
    required_days = (
        WALK_FORWARD_TRAIN_DAYS
        + WALK_FORWARD_TEST_DAYS
        + (MIN_WALK_FORWARD_WINDOWS - 1) * WALK_FORWARD_STEP_DAYS
    )
    return math.ceil(required_days * candles_per_day(interval))


async def create_strategy_deployment(
    db: AsyncSession,
    *,
    source_backtest_id: int,
    target_execution_mode: str,
) -> StrategyDeployment:
    """Persist approval or rejection evidence without enabling the engine."""
    source = await db.scalar(
        select(BacktestResult).where(BacktestResult.id == source_backtest_id)
    )
    if source is None:
        raise LookupError("source backtest was not found")

    strategy = _parse_deployable_strategy(source)
    reasons = _source_rejection_reasons(source)
    walk_forward = None
    required_candles = required_walk_forward_candles(source.interval)
    candles = await market_data_collector.get_latest_candles(
        db=db,
        symbol=source.symbol,
        interval=source.interval,
        limit=WALK_FORWARD_HISTORY_LIMIT,
    )

    if len(candles) < required_candles:
        reasons.append(
            "Insufficient fresh candle history for three independent "
            f"walk-forward windows: need {required_candles}, found {len(candles)}"
        )
    elif not has_recent_closed_candle(
        candles,
        source.interval,
        required_candles=required_candles,
    ):
        reasons.append(
            "Candle history must be recent and continuous before "
            "walk-forward deployment evidence can be trusted"
        )
    else:
        signals, _definition = build_technical_strategy_signals(candles, strategy)
        walk_forward = run_walk_forward(
            df=candles,
            signals=signals,
            train_days=WALK_FORWARD_TRAIN_DAYS,
            test_days=WALK_FORWARD_TEST_DAYS,
            step_days=WALK_FORWARD_STEP_DAYS,
            initial_capital=float(source.initial_capital),
            signal_threshold=float(source.signal_threshold),
            atr_stop_multiplier=float(source.atr_stop_multiplier),
            risk_reward_ratio=float(source.risk_reward_ratio),
            allow_short=False,
            candles_per_day=candles_per_day(source.interval),
        )
        reasons.extend(_walk_forward_rejection_reasons(walk_forward))

    execution_config = {
        "signal_threshold": float(source.signal_threshold),
        "atr_stop_multiplier": float(source.atr_stop_multiplier),
        "risk_reward_ratio": float(source.risk_reward_ratio),
    }
    deployment = StrategyDeployment(
        source_backtest_id=source.id,
        symbol=source.symbol,
        interval=source.interval,
        strategy_config_json=strategy.model_dump_json(),
        execution_config_json=json.dumps(execution_config, sort_keys=True),
        target_execution_mode=target_execution_mode,
        status=REJECTED if reasons else APPROVED,
        total_test_trades=walk_forward.total_test_trades if walk_forward else 0,
        walk_forward_windows=len(walk_forward.windows) if walk_forward else 0,
        avg_return_pct=walk_forward.avg_return_pct if walk_forward else 0,
        avg_sharpe=walk_forward.avg_sharpe if walk_forward else 0,
        avg_max_drawdown_pct=walk_forward.avg_max_dd_pct if walk_forward else 0,
        avg_profit_factor=walk_forward.avg_profit_factor if walk_forward else 0,
        consistency_score=walk_forward.consistency_score if walk_forward else 0,
        overfitting_score=walk_forward.overfitting_score if walk_forward else 0,
        rejection_reason="; ".join(reasons) if reasons else None,
    )
    db.add(deployment)
    await db.flush()
    return deployment


async def activate_strategy_deployment(
    db: AsyncSession,
    *,
    deployment_id: int,
    execution_mode: TradingExecutionMode,
) -> StrategyDeployment:
    """Atomically replace the active strategy for its exact runtime scope."""
    deployment = await db.scalar(
        select(StrategyDeployment)
        .where(StrategyDeployment.id == deployment_id)
        .with_for_update()
    )
    if deployment is None:
        raise LookupError("strategy deployment was not found")
    if deployment.status not in {APPROVED, ACTIVE}:
        raise StrategyDeploymentSourceError(
            "only an APPROVED strategy deployment can be activated"
        )
    if deployment.target_execution_mode != execution_mode.value:
        raise StrategyDeploymentSourceError(
            "strategy target execution mode does not match the active runtime mode"
        )

    await _acquire_strategy_activation_lock(
        db,
        symbol=deployment.symbol,
        interval=deployment.interval,
        execution_mode=deployment.target_execution_mode,
    )

    await db.execute(
        update(StrategyDeployment)
        .where(
            StrategyDeployment.symbol == deployment.symbol,
            StrategyDeployment.interval == deployment.interval,
            StrategyDeployment.target_execution_mode == deployment.target_execution_mode,
            StrategyDeployment.status == ACTIVE,
            StrategyDeployment.id != deployment.id,
        )
        .values(status=DISABLED, deactivated_at=datetime.now(UTC))
    )
    deployment.status = ACTIVE
    deployment.activated_at = datetime.now(UTC)
    deployment.deactivated_at = None
    return deployment


async def get_active_technical_strategy(
    db: AsyncSession,
    *,
    symbol: str,
    interval: str,
    execution_mode: TradingExecutionMode,
) -> ActiveTechnicalStrategy | None:
    """Load the only runtime-eligible technical strategy, failing closed on drift."""
    result = await db.execute(
        select(StrategyDeployment)
        .where(
            StrategyDeployment.symbol == symbol,
            StrategyDeployment.interval == interval,
            StrategyDeployment.target_execution_mode == execution_mode.value,
            StrategyDeployment.status == ACTIVE,
        )
        .order_by(StrategyDeployment.activated_at.desc(), StrategyDeployment.id.desc())
        .limit(2)
    )
    deployments = list(result.scalars().all())
    if not deployments:
        return None
    if len(deployments) != 1:
        logger.critical(
            "active_strategy_deployment_ambiguous",
            symbol=symbol,
            interval=interval,
            execution_mode=execution_mode.value,
            deployment_ids=[deployment.id for deployment in deployments],
        )
        return None
    deployment = deployments[0]

    try:
        strategy = TechnicalStrategyConfig.model_validate_json(deployment.strategy_config_json)
        execution_config = json.loads(deployment.execution_config_json)
        signal_threshold = float(execution_config["signal_threshold"])
        atr_stop_multiplier = float(execution_config["atr_stop_multiplier"])
        risk_reward_ratio = float(execution_config["risk_reward_ratio"])
        _validate_execution_config(
            signal_threshold=signal_threshold,
            atr_stop_multiplier=atr_stop_multiplier,
            risk_reward_ratio=risk_reward_ratio,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.critical(
            "active_strategy_deployment_invalid",
            deployment_id=deployment.id,
            error=str(exc),
        )
        return None

    return ActiveTechnicalStrategy(
        deployment_id=deployment.id,
        strategy=strategy,
        signal_threshold=signal_threshold,
        atr_stop_multiplier=atr_stop_multiplier,
        risk_reward_ratio=risk_reward_ratio,
    )


async def get_runtime_symbols(
    db: AsyncSession,
    *,
    execution_mode: TradingExecutionMode,
    base_symbols: Iterable[str] | None = None,
) -> list[str]:
    """Return only assets that need live monitoring in this runtime ledger."""
    from app.config import settings

    active_strategy_symbols = await db.scalars(
        select(StrategyDeployment.symbol).where(
            StrategyDeployment.target_execution_mode == execution_mode.value,
            StrategyDeployment.status == ACTIVE,
        )
    )
    open_position_symbols = await db.scalars(
        select(Position.symbol).where(
            Position.execution_mode == execution_mode.value,
            Position.is_open.is_(True),
        )
    )
    return sorted(
        {
            *(base_symbols if base_symbols is not None else settings.symbols_list),
            *(str(symbol).upper() for symbol in active_strategy_symbols),
            *(str(symbol).upper() for symbol in open_position_symbols),
        }
    )


async def _acquire_strategy_activation_lock(
    db: AsyncSession,
    *,
    symbol: str,
    interval: str,
    execution_mode: str,
) -> None:
    """Serialize activation for one runtime scope until the transaction commits."""
    lock_material = f"trademaster:strategy-activation:{symbol}:{interval}:{execution_mode}".encode()
    lock_key = int.from_bytes(
        hashlib.blake2b(lock_material, digest_size=8).digest(),
        byteorder="big",
        signed=True,
    )
    try:
        await db.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )
    except Exception as exc:
        raise StrategyDeploymentSourceError(
            "could not acquire the strategy activation lock"
        ) from exc


def _parse_deployable_strategy(source: BacktestResult) -> TechnicalStrategyConfig:
    if source.execution_profile != "spot_long_only":
        raise StrategyDeploymentSourceError(
            "only a Spot long-only technical backtest can be deployed"
        )
    if source.interval not in RUNTIME_INTERVALS:
        raise StrategyDeploymentSourceError(
            "the trading engine supports strategy deployments only on 15m, 1h, or 4h"
        )
    try:
        strategy = TechnicalStrategyConfig.model_validate_json(source.strategy_config_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyDeploymentSourceError(
            "source backtest does not contain a valid technical strategy configuration"
        ) from exc
    if strategy.execution_profile != "spot_long_only":
        raise StrategyDeploymentSourceError(
            "source strategy is not restricted to Spot long-only execution"
        )
    return strategy


def _source_rejection_reasons(source: BacktestResult) -> list[str]:
    reasons: list[str] = []
    if source.total_trades < MIN_SOURCE_TRADES:
        reasons.append(f"Source backtest needs at least {MIN_SOURCE_TRADES} trades")
    if float(source.total_return_pct) <= 0:
        reasons.append("Source backtest return must be positive")
    if not _meets_profit_factor(float(source.profit_factor)):
        reasons.append(f"Source profit factor must be finite and at least {MIN_PROFIT_FACTOR:.2f}")
    if float(source.max_drawdown_pct) > MAX_DRAWDOWN_PCT:
        reasons.append(
            f"Source max drawdown exceeds {MAX_DRAWDOWN_PCT:.0%}"
        )
    return reasons


def _walk_forward_rejection_reasons(walk_forward) -> list[str]:
    reasons: list[str] = []
    traded_windows = sum(window.test_trades > 0 for window in walk_forward.windows)
    if traded_windows < MIN_WALK_FORWARD_WINDOWS:
        reasons.append(
            "Walk-forward needs at least "
            f"{MIN_WALK_FORWARD_WINDOWS} out-of-sample windows with trades"
        )
    if walk_forward.total_test_trades < MIN_OOS_TRADES:
        reasons.append(
            f"Walk-forward needs at least {MIN_OOS_TRADES} out-of-sample trades"
        )
    if walk_forward.avg_return_pct <= 0:
        reasons.append("Walk-forward mean return must be positive")
    if not _meets_profit_factor(walk_forward.avg_profit_factor):
        reasons.append(
            "Walk-forward profit factor must be finite and at least "
            f"{MIN_PROFIT_FACTOR:.2f}"
        )
    if walk_forward.consistency_score < MIN_CONSISTENCY_SCORE:
        reasons.append(
            "Walk-forward profitable-window consistency must be at least "
            f"{MIN_CONSISTENCY_SCORE:.0%}"
        )
    if walk_forward.avg_max_dd_pct > MAX_DRAWDOWN_PCT:
        reasons.append(
            f"Walk-forward mean drawdown exceeds {MAX_DRAWDOWN_PCT:.0%}"
        )
    if walk_forward.overfitting_score > MAX_OVERFITTING_SCORE:
        reasons.append(
            "Walk-forward overfitting score exceeds "
            f"{MAX_OVERFITTING_SCORE:.2f}"
        )
    return reasons


def _meets_profit_factor(value: float) -> bool:
    return math.isfinite(value) and value >= MIN_PROFIT_FACTOR


def _validate_execution_config(
    *,
    signal_threshold: float,
    atr_stop_multiplier: float,
    risk_reward_ratio: float,
) -> None:
    values = (signal_threshold, atr_stop_multiplier, risk_reward_ratio)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("strategy execution configuration must be finite")
    if not 0.1 <= signal_threshold <= 0.9:
        raise ValueError("strategy signal threshold is out of range")
    if not 0.5 <= atr_stop_multiplier <= 5:
        raise ValueError("strategy ATR stop multiplier is out of range")
    if not 0.5 <= risk_reward_ratio <= 10:
        raise ValueError("strategy risk/reward ratio is out of range")

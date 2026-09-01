"""One-click research workflow: asset study, predictive model, and safe strategy choice."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger
from app.models.backtest import BacktestResult
from app.schemas.trading import TechnicalStrategyConfig
from app.services.backtest.engine import BacktestEngine, BacktestResult as EngineBacktestResult
from app.services.backtest.technical_strategy import build_technical_strategy_signals
from app.services.market.data_collector import market_data_collector
from app.services.market.spot_asset_catalog import SpotAsset, spot_asset_catalog
from app.services.ml.features import feature_engineer
from app.services.ml.models.xgboost_model import XGBoostTradingModel
from app.services.ml.preprocessor import Preprocessor
from app.services.ml.tracking import ml_tracker
from app.services.strategy_deployments import create_strategy_deployment

RESEARCH_INTERVAL = "1h"
RESEARCH_HISTORY_DAYS = 365
MIN_RESEARCH_CANDLES = 3_000
RESEARCH_CAPITAL = 10_000.0
MIN_PREDICTIVE_BALANCED_ACCURACY = 0.40

logger = get_logger(__name__)


class AssetIntelligenceError(Exception):
    """The chosen asset cannot yield trustworthy research evidence yet."""


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    strategy: TechnicalStrategyConfig
    signal_threshold: float
    atr_stop_multiplier: float
    risk_reward_ratio: float


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: StrategyCandidate
    result: EngineBacktestResult
    score: float


async def study_asset(db: AsyncSession, *, symbol: str) -> dict[str, object]:
    """Produce research evidence for exactly one liquid Spot asset.

    This function intentionally stops before activation. Its output can only
    create an APPROVED/REJECTED strategy deployment; existing activation and
    Testnet/LIVE safety gates remain the sole route to execution.
    """
    asset = await spot_asset_catalog.require(symbol)
    await market_data_collector.ensure_public_history(
        db,
        symbol=asset.symbol,
        interval=RESEARCH_INTERVAL,
        days_back=RESEARCH_HISTORY_DAYS,
    )
    candles = await market_data_collector.get_latest_candles(
        db,
        symbol=asset.symbol,
        interval=RESEARCH_INTERVAL,
        limit=12_000,
    )
    if len(candles) < MIN_RESEARCH_CANDLES:
        raise AssetIntelligenceError(
            f"{asset.symbol} has only {len(candles)} usable candles; at least "
            f"{MIN_RESEARCH_CANDLES} are required for study and walk-forward validation"
        )

    market_study = _market_study(candles, asset)
    predictive_model = await _train_predictive_model(db, candles, asset.symbol)
    predictive_blocker = _predictive_model_blocker(predictive_model)
    if predictive_blocker:
        return {
            "symbol": asset.symbol,
            "execution_mode": settings.execution_mode.value,
            "market_study": market_study,
            "predictive_model": predictive_model,
            "recommendation": {
                "strategy_name": "Nenhuma estratégia liberada",
                "backtest_id": None,
                "deployment_id": None,
                "deployment_status": "UNAVAILABLE",
                "reasons": [predictive_blocker],
            },
        }

    selected = _choose_candidate(
        candles,
        market_study["trend"],
        str(predictive_model["latest_signal"]),
    )
    source_backtest = _persist_backtest(db, asset.symbol, selected)
    await db.flush()

    deployment = await create_strategy_deployment(
        db,
        source_backtest_id=source_backtest.id,
        target_execution_mode=settings.execution_mode.value,
    )
    await db.flush()

    reasons = (
        []
        if deployment.status == "APPROVED"
        else [part for part in (deployment.rejection_reason or "No deployment evidence available").split("; ") if part]
    )
    return {
        "symbol": asset.symbol,
        "execution_mode": settings.execution_mode.value,
        "market_study": market_study,
        "predictive_model": predictive_model,
        "recommendation": {
            "strategy_name": selected.candidate.name,
            "backtest_id": source_backtest.id,
            "deployment_id": deployment.id,
            "deployment_status": deployment.status if deployment.status in {"APPROVED", "REJECTED"} else "UNAVAILABLE",
            "reasons": reasons,
        },
    }


def _market_study(candles: pd.DataFrame, asset: SpotAsset) -> dict[str, object]:
    close = pd.to_numeric(candles["close"], errors="coerce").dropna()
    if len(close) < 200:
        raise AssetIntelligenceError("Asset history is insufficient for a market-regime study")

    last_price = float(close.iloc[-1])
    sma_50 = float(close.iloc[-50:].mean())
    sma_200 = float(close.iloc[-200:].mean())
    if last_price > sma_50 > sma_200:
        trend = "UPTREND"
    elif last_price < sma_50 < sma_200:
        trend = "DOWNTREND"
    else:
        trend = "RANGE"

    returns = close.pct_change().tail(168).dropna()
    volatility_pct = float(returns.std(ddof=0) * math.sqrt(24) * 100) if not returns.empty else 0.0
    return {
        "trend": trend,
        "volatility_pct": round(volatility_pct, 4),
        "liquidity_quote_volume_24h": asset.quote_volume_24h,
        "candles": len(candles),
    }


async def _train_predictive_model(
    db: AsyncSession,
    candles: pd.DataFrame,
    symbol: str,
) -> dict[str, object]:
    """Train a temporal XGBoost research model and retain auditable metrics."""
    started_at = time.perf_counter()
    try:
        features = feature_engineer.build_features(candles)
        feature_columns = feature_engineer.get_feature_columns(features)
        prepared = Preprocessor(threshold=0.007).create_target(features, horizon=5)
        split = Preprocessor(threshold=0.007).prepare_tabular(prepared, feature_columns)
        if min(len(split.X_train), len(split.X_val), len(split.X_test)) < 100:
            raise AssetIntelligenceError("Asset history is insufficient for predictive-model validation")

        model = XGBoostTradingModel()
        training = model.train(
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
            n_estimators=160,
            max_depth=3,
            learning_rate=0.05,
            feature_names=split.feature_names,
        )
        test_predictions = model._model.predict(split.X_test)  # type: ignore[union-attr]
        test_accuracy = float(np.mean(test_predictions == split.y_test))
        test_balanced_accuracy = float(balanced_accuracy_score(split.y_test, test_predictions))
        latest_features = features[feature_columns].dropna().iloc[-1:].to_numpy()
        latest_features = split.scaler.transform(latest_features)
        prediction = model.predict(latest_features[0])
        dataset_hash = hashlib.sha256(
            pd.util.hash_pandas_object(candles[["open_time", "close"]], index=False).values.tobytes()
        ).hexdigest()
        await ml_tracker.log_training_run(
            db,
            model_type="xgboost",
            symbol=symbol,
            metrics_dict={
                "train_accuracy": round(training.accuracy, 6),
                "validation_accuracy": round(training.val_accuracy, 6),
                "test_accuracy": round(test_accuracy, 6),
                "test_balanced_accuracy": round(test_balanced_accuracy, 6),
                "best_epoch": training.best_epoch,
            },
            hyperparams={"n_estimators": 160, "max_depth": 3, "learning_rate": 0.05, "horizon": 5},
            dataset_info={"hash": dataset_hash, "size": len(prepared)},
            duration_seconds=round(time.perf_counter() - started_at, 4),
        )
        return {
            "model_type": "xgboost",
            "trained": True,
            "validation_accuracy": round(test_balanced_accuracy, 4),
            "samples": len(prepared),
            "latest_signal": prediction.action_label,
        }
    except Exception as exc:
        try:
            await ml_tracker.log_training_run(
                db,
                model_type="xgboost",
                symbol=symbol,
                metrics_dict={},
                hyperparams={"horizon": 5},
                dataset_info={"size": len(candles)},
                duration_seconds=round(time.perf_counter() - started_at, 4),
                status="failed",
                error_message=str(exc)[:500],
            )
        except Exception as tracking_exc:
            logger.warning(
                "asset_intelligence_training_failure_untracked",
                symbol=symbol,
                error=str(tracking_exc),
            )
        return {
            "model_type": "xgboost",
            "trained": False,
            "validation_accuracy": None,
            "samples": len(candles),
            "latest_signal": "UNAVAILABLE",
        }


def _predictive_model_blocker(predictive_model: dict[str, object]) -> str | None:
    if not predictive_model.get("trained"):
        return "O modelo preditivo não produziu evidência válida para este ativo"

    validation_accuracy = predictive_model.get("validation_accuracy")
    if not isinstance(validation_accuracy, (float, int)) or validation_accuracy < MIN_PREDICTIVE_BALANCED_ACCURACY:
        return (
            "O modelo preditivo não superou a qualidade mínima de validação "
            f"balanceada ({MIN_PREDICTIVE_BALANCED_ACCURACY:.0%})"
        )
    if predictive_model.get("latest_signal") == "SELL":
        return "O modelo preditivo atual indica baixa; Spot long-only não abre posição contra esse sinal"
    return None


def _choose_candidate(
    candles: pd.DataFrame,
    trend: object,
    predictive_signal: str,
) -> CandidateEvaluation:
    candidates = _candidates_for_trend(str(trend))
    if predictive_signal == "HOLD":
        # A neutral model does not veto the asset, but it makes the deployed
        # technical strategy wait for a materially stronger confirmation.
        candidates = [
            replace(candidate, signal_threshold=max(candidate.signal_threshold, 0.45))
            for candidate in candidates
        ]
    evaluations = [_evaluate_candidate(candles, candidate) for candidate in candidates]
    return max(evaluations, key=lambda evaluation: evaluation.score)


def _evaluate_candidate(candles: pd.DataFrame, candidate: StrategyCandidate) -> CandidateEvaluation:
    signals, _definition = build_technical_strategy_signals(candles, candidate.strategy)
    result = BacktestEngine(
        initial_capital=RESEARCH_CAPITAL,
        signal_threshold=candidate.signal_threshold,
        atr_stop_multiplier=candidate.atr_stop_multiplier,
        risk_reward_ratio=candidate.risk_reward_ratio,
        allow_short=False,
    ).run(candles, signals=signals)
    metrics = result.metrics
    if metrics.total_trades < 20 or not math.isfinite(metrics.profit_factor):
        score = -10_000.0 + metrics.total_trades
    else:
        # Ranking is deliberately conservative. Walk-forward validation is a
        # separate mandatory deployment gate; a high in-sample score alone can
        # never make the strategy executable.
        score = (
            metrics.total_return_pct * 4
            + min(metrics.profit_factor, 3.0)
            + min(metrics.sharpe_ratio, 3.0) * 0.5
            - metrics.max_drawdown_pct * 4
        )
    return CandidateEvaluation(candidate=candidate, result=result, score=score)


def _persist_backtest(
    db: AsyncSession,
    symbol: str,
    selected: CandidateEvaluation,
) -> BacktestResult:
    metrics = selected.result.metrics
    candidate = selected.candidate
    row = BacktestResult(
        symbol=symbol,
        interval=RESEARCH_INTERVAL,
        initial_capital=RESEARCH_CAPITAL,
        signal_threshold=candidate.signal_threshold,
        atr_stop_multiplier=candidate.atr_stop_multiplier,
        risk_reward_ratio=candidate.risk_reward_ratio,
        strategy_name=candidate.name,
        execution_profile="spot_long_only",
        strategy_config_json=candidate.strategy.model_dump_json(),
        total_return=metrics.total_return,
        total_trades=metrics.total_trades,
        winning_trades=metrics.winning_trades,
        losing_trades=metrics.losing_trades,
        win_rate=metrics.win_rate,
        total_return_pct=metrics.total_return_pct,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        max_drawdown_pct=metrics.max_drawdown_pct,
        profit_factor=metrics.profit_factor,
        expectancy=metrics.expectancy,
        equity_curve_json=_equity_curve_json(selected.result.equity_curve),
    )
    db.add(row)
    return row


def _equity_curve_json(equity_curve: list[float]) -> str:
    import json

    return json.dumps(equity_curve[-500:])


def _candidates_for_trend(trend: str) -> list[StrategyCandidate]:
    trend_candidates = [
        StrategyCandidate(
            name="Tendência SMA + RSI",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["sma", "rsi"],
                indicator_params={
                    "sma": {"sma_short": 10, "sma_long": 30},
                    "rsi": {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30},
                },
                min_confirmations=2,
            ),
            signal_threshold=0.3,
            atr_stop_multiplier=2.0,
            risk_reward_ratio=2.0,
        ),
        StrategyCandidate(
            name="Tendência EMA + MACD",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["ema", "macd"],
                indicator_params={
                    "ema": {"ema_short": 12, "ema_long": 26},
                    "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
                },
                min_confirmations=2,
            ),
            signal_threshold=0.3,
            atr_stop_multiplier=2.2,
            risk_reward_ratio=2.2,
        ),
        StrategyCandidate(
            name="Rompimento com confirmação EMA",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["breakout", "ema"],
                indicator_params={
                    "breakout": {"breakout_lookback": 20},
                    "ema": {"ema_short": 12, "ema_long": 26},
                },
                min_confirmations=1,
            ),
            signal_threshold=0.3,
            atr_stop_multiplier=2.5,
            risk_reward_ratio=2.5,
        ),
    ]
    range_candidates = [
        StrategyCandidate(
            name="Reversão RSI + Bollinger",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["rsi", "bollinger"],
                indicator_params={
                    "rsi": {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30},
                    "bollinger": {"bb_period": 20, "bb_std": 2},
                },
                min_confirmations=1,
            ),
            signal_threshold=0.3,
            atr_stop_multiplier=1.8,
            risk_reward_ratio=1.8,
        ),
        StrategyCandidate(
            name="Reversão Engolfo + RSI",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["engulfing", "rsi"],
                indicator_params={
                    "rsi": {"rsi_period": 14, "rsi_overbought": 68, "rsi_oversold": 32},
                },
                min_confirmations=1,
            ),
            signal_threshold=0.3,
            atr_stop_multiplier=1.8,
            risk_reward_ratio=1.8,
        ),
    ]
    if trend == "RANGE":
        return range_candidates + trend_candidates[:1]
    if trend == "DOWNTREND":
        return [trend_candidates[2], range_candidates[0], trend_candidates[1]]
    return trend_candidates + range_candidates[:1]

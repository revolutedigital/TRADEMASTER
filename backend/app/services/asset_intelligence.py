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
from app.services.backtest.walk_forward import candles_per_day, run_walk_forward
from app.services.market.data_collector import market_data_collector
from app.services.market.pattern_intelligence import assess_closed_candle_pattern
from app.services.market.spot_asset_catalog import SpotAsset, spot_asset_catalog
from app.services.ml.features import feature_engineer
from app.services.ml.models.xgboost_model import XGBoostTradingModel
from app.services.ml.preprocessor import Preprocessor
from app.services.ml.tracking import ml_tracker
from app.services.strategy_deployments import (
    WALK_FORWARD_EMBARGO_CANDLES,
    WALK_FORWARD_STEP_DAYS,
    WALK_FORWARD_TEST_DAYS,
    WALK_FORWARD_TRAIN_DAYS,
    create_strategy_deployment,
    required_walk_forward_candles,
)

RESEARCH_INTERVAL = "1h"
RESEARCH_HISTORY_DAYS = 365
MIN_RESEARCH_CANDLES = 3_000
MIN_SELECTION_CANDLES = 3_000
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
    selection_validation: object
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
    candles = _closed_candles_only(candles)
    if len(candles) < MIN_RESEARCH_CANDLES:
        raise AssetIntelligenceError(
            f"{asset.symbol} has only {len(candles)} usable candles; at least "
            f"{MIN_RESEARCH_CANDLES} are required for study and walk-forward validation"
        )

    validation_candle_count = required_walk_forward_candles(RESEARCH_INTERVAL)
    selection_candles = candles.iloc[:-validation_candle_count].copy()
    validation_candles = candles.iloc[-validation_candle_count:].copy()
    if len(selection_candles) < MIN_SELECTION_CANDLES:
        raise AssetIntelligenceError(
            "Asset history is insufficient after reserving the independent validation period"
        )

    market_study = _market_study(candles, asset)
    pattern_study = assess_closed_candle_pattern(candles).as_dict()
    predictive_model = await _train_predictive_model(
        db,
        selection_candles,
        asset.symbol,
        inference_candles=candles,
    )
    predictive_blocker = _predictive_model_blocker(predictive_model)
    if predictive_blocker:
        return {
            "symbol": asset.symbol,
            "execution_mode": settings.execution_mode.value,
            "market_study": market_study,
            "pattern_study": pattern_study,
            "predictive_model": predictive_model,
            "recommendation": {
                "strategy_name": "Nenhuma estratégia liberada",
                "backtest_id": None,
                "deployment_id": None,
                "deployment_status": "UNAVAILABLE",
                "reasons": [predictive_blocker],
            },
        }

    if pattern_study["pattern"] == "OBSERVATION_ONLY":
        return {
            "symbol": asset.symbol,
            "execution_mode": settings.execution_mode.value,
            "market_study": market_study,
            "pattern_study": pattern_study,
            "predictive_model": predictive_model,
            "recommendation": {
                "strategy_name": "Nenhuma estratégia liberada",
                "backtest_id": None,
                "deployment_id": None,
                "deployment_status": "UNAVAILABLE",
                "reasons": [str(pattern_study["explanation"])],
            },
        }

    selected = _choose_candidate(
        selection_candles,
        str(pattern_study["pattern"]),
        str(predictive_model["latest_signal"]),
    )
    selected = _with_pattern_research_context(
        selected,
        regime=str(pattern_study["regime"]),
        pattern=str(pattern_study["pattern"]),
        selection_candles=selection_candles,
        validation_candles=validation_candles,
    )
    source_backtest = _persist_backtest(db, asset.symbol, selected)
    await db.flush()

    deployment = await create_strategy_deployment(
        db,
        source_backtest_id=source_backtest.id,
        target_execution_mode=settings.execution_mode.value,
        validation_candles=validation_candles,
    )
    await db.flush()

    reasons = (
        []
        if deployment.status == "APPROVED"
        else [
            part
            for part in (deployment.rejection_reason or "No deployment evidence available").split(
                "; "
            )
            if part
        ]
    )
    return {
        "symbol": asset.symbol,
        "execution_mode": settings.execution_mode.value,
        "market_study": market_study,
        "pattern_study": pattern_study,
        "predictive_model": predictive_model,
        "recommendation": {
            "strategy_name": selected.candidate.name,
            "backtest_id": source_backtest.id,
            "deployment_id": deployment.id,
            "deployment_status": deployment.status
            if deployment.status in {"APPROVED", "REJECTED"}
            else "UNAVAILABLE",
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
    *,
    inference_candles: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Train a temporal XGBoost research model and retain auditable metrics."""
    started_at = time.perf_counter()
    try:
        features = feature_engineer.build_features(candles)
        feature_columns = feature_engineer.get_feature_columns(features)
        prepared = Preprocessor(threshold=0.007).create_target(features, horizon=5)
        split = Preprocessor(threshold=0.007).prepare_tabular(prepared, feature_columns)
        if min(len(split.X_train), len(split.X_val), len(split.X_test)) < 100:
            raise AssetIntelligenceError(
                "Asset history is insufficient for predictive-model validation"
            )

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
        inference_features = feature_engineer.build_features(
            inference_candles if inference_candles is not None else candles
        )
        latest_features = inference_features[feature_columns].dropna().iloc[-1:].to_numpy()
        latest_features = split.scaler.transform(latest_features)
        prediction = model.predict(latest_features[0])
        dataset_hash = hashlib.sha256(
            pd.util.hash_pandas_object(
                candles[["open_time", "close"]], index=False
            ).values.tobytes()
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
    if (
        not isinstance(validation_accuracy, (float, int))
        or validation_accuracy < MIN_PREDICTIVE_BALANCED_ACCURACY
    ):
        return (
            "O modelo preditivo não superou a qualidade mínima de validação "
            f"balanceada ({MIN_PREDICTIVE_BALANCED_ACCURACY:.0%})"
        )
    if predictive_model.get("latest_signal") == "SELL":
        return "O modelo preditivo atual indica baixa; Spot long-only não abre posição contra esse sinal"
    return None


def _choose_candidate(
    candles: pd.DataFrame,
    pattern_or_trend: object,
    predictive_signal: str,
) -> CandidateEvaluation:
    pattern = _normalize_pattern_name(str(pattern_or_trend))
    candidates = _candidates_for_pattern(pattern)
    if predictive_signal == "HOLD":
        # Technical signals are discrete -1/0/1 votes. Raising a numeric
        # threshold would therefore be cosmetic, so HOLD is represented by a
        # stricter flow confirmation inside every automatic candidate.
        candidates = [_require_stronger_flow(candidate) for candidate in candidates]
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
    selection_validation = run_walk_forward(
        df=candles,
        signals=signals,
        train_days=WALK_FORWARD_TRAIN_DAYS,
        test_days=WALK_FORWARD_TEST_DAYS,
        step_days=WALK_FORWARD_STEP_DAYS,
        initial_capital=RESEARCH_CAPITAL,
        signal_threshold=candidate.signal_threshold,
        atr_stop_multiplier=candidate.atr_stop_multiplier,
        risk_reward_ratio=candidate.risk_reward_ratio,
        allow_short=False,
        candles_per_day=candles_per_day(RESEARCH_INTERVAL),
        embargo_candles=WALK_FORWARD_EMBARGO_CANDLES,
    )
    metrics = result.metrics
    if selection_validation.total_test_trades < 50 or not math.isfinite(
        selection_validation.avg_profit_factor
    ):
        score = -10_000.0 + selection_validation.total_test_trades
    else:
        # Candidate selection is based on its earlier out-of-sample windows,
        # never its whole-history backtest. The later holdout remains untouched
        # until deployment validation.
        score = (
            selection_validation.avg_return_pct * 4
            + min(selection_validation.avg_profit_factor, 3.0)
            + min(selection_validation.avg_sharpe, 3.0) * 0.5
            - selection_validation.avg_max_dd_pct * 4
            + selection_validation.consistency_score
        )
    return CandidateEvaluation(
        candidate=candidate,
        result=result,
        selection_validation=selection_validation,
        score=score,
    )


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


def _closed_candles_only(candles: pd.DataFrame) -> pd.DataFrame:
    """Discard in-progress candles before any feature, target, or pattern step."""
    if candles.empty or "close_time" not in candles:
        return candles.copy()
    frame = candles.copy()
    close_time = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    return frame.loc[close_time.notna() & (close_time <= now)].reset_index(drop=True)


def _with_pattern_research_context(
    selected: CandidateEvaluation,
    *,
    regime: str,
    pattern: str,
    selection_candles: pd.DataFrame,
    validation_candles: pd.DataFrame,
) -> CandidateEvaluation:
    context = {
        "library_version": "pattern-library-v1",
        "regime": regime,
        "pattern": pattern,
        "selection_end_time": _timestamp_text(selection_candles.iloc[-1]["close_time"]),
        "validation_start_time": _timestamp_text(validation_candles.iloc[0]["open_time"]),
        "selection_oos_windows": len(selected.selection_validation.windows),
        "selection_oos_trades": selected.selection_validation.total_test_trades,
        "selection_oos_return_pct": selected.selection_validation.avg_return_pct,
        "selection_oos_profit_factor": _finite_or_zero(
            selected.selection_validation.avg_profit_factor
        ),
    }
    strategy = TechnicalStrategyConfig.model_validate(
        selected.candidate.strategy.model_dump() | {"research_context": context}
    )
    return replace(
        selected,
        candidate=replace(selected.candidate, strategy=strategy),
    )


def _timestamp_text(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat()


def _finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def _normalize_pattern_name(value: str) -> str:
    legacy_mapping = {
        "UPTREND": "TREND_CONTINUATION",
        "DOWNTREND": "TREND_CONTINUATION",
        "RANGE": "MEAN_REVERSION",
    }
    return legacy_mapping.get(value, value)


def _require_stronger_flow(candidate: StrategyCandidate) -> StrategyCandidate:
    parameters = dict(candidate.strategy.indicator_params)
    flow_parameters = dict(parameters.get("volume_confirmation", {}))
    if flow_parameters:
        flow_parameters["min_relative_volume"] = max(
            float(flow_parameters.get("min_relative_volume", 1.0)),
            1.15,
        )
        parameters["volume_confirmation"] = flow_parameters
    return replace(
        candidate,
        strategy=TechnicalStrategyConfig.model_validate(
            candidate.strategy.model_dump() | {"indicator_params": parameters}
        ),
    )


def _candidates_for_pattern(pattern: str) -> list[StrategyCandidate]:
    trend_candidates = [
        StrategyCandidate(
            name="Continuação EMA + MACD + fluxo",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["ema", "macd", "volume_confirmation"],
                indicator_params={
                    "ema": {"ema_short": 12, "ema_long": 26},
                    "macd": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
                    "volume_confirmation": {
                        "volume_lookback": 48,
                        "min_relative_volume": 0.9,
                        "min_taker_imbalance": -0.1,
                    },
                },
                min_confirmations=2,
            ),
            signal_threshold=0.5,
            atr_stop_multiplier=2.2,
            risk_reward_ratio=2.2,
        ),
        StrategyCandidate(
            name="Continuação SMA + RSI + fluxo",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["sma", "rsi", "volume_confirmation"],
                indicator_params={
                    "sma": {"sma_short": 10, "sma_long": 30},
                    "rsi": {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30},
                    "volume_confirmation": {
                        "volume_lookback": 48,
                        "min_relative_volume": 1.0,
                        "min_taker_imbalance": -0.05,
                    },
                },
                min_confirmations=2,
            ),
            signal_threshold=0.5,
            atr_stop_multiplier=2.0,
            risk_reward_ratio=2.0,
        ),
    ]
    breakout_candidates = [
        StrategyCandidate(
            name="Rompimento de compressão + EMA + fluxo",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["breakout", "ema", "volume_confirmation"],
                indicator_params={
                    "breakout": {"breakout_lookback": 20},
                    "ema": {"ema_short": 12, "ema_long": 26},
                    "volume_confirmation": {
                        "volume_lookback": 48,
                        "min_relative_volume": 1.1,
                        "min_taker_imbalance": 0.0,
                    },
                },
                min_confirmations=2,
            ),
            signal_threshold=0.5,
            atr_stop_multiplier=2.5,
            risk_reward_ratio=2.5,
        ),
    ]
    range_candidates = [
        StrategyCandidate(
            name="Reversão RSI + Bollinger + fluxo",
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["rsi", "bollinger", "volume_confirmation"],
                indicator_params={
                    "rsi": {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30},
                    "bollinger": {"bb_period": 20, "bb_std": 2},
                    "volume_confirmation": {
                        "volume_lookback": 48,
                        "min_relative_volume": 0.85,
                        "min_taker_imbalance": -0.1,
                    },
                },
                min_confirmations=2,
            ),
            signal_threshold=0.5,
            atr_stop_multiplier=1.8,
            risk_reward_ratio=1.8,
        ),
    ]
    if pattern == "COMPRESSION_BREAKOUT":
        return breakout_candidates
    if pattern == "MEAN_REVERSION":
        return range_candidates
    return trend_candidates


def _candidates_for_trend(trend: str) -> list[StrategyCandidate]:
    """Compatibility helper retained for existing research callers and tests."""
    if trend == "RANGE":
        return _candidates_for_pattern("MEAN_REVERSION") + _candidates_for_pattern(
            "TREND_CONTINUATION"
        )
    return _candidates_for_pattern(_normalize_pattern_name(trend))

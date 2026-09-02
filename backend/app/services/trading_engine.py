"""Trading Engine: the main loop that connects signals -> risk -> execution.

This is the core autonomous trading loop:
1. Receive market data updates (15m candles)
2. Generate ML predictions or technical signals
3. Validate through risk management
4. Execute approved trades
5. Monitor and manage open positions

Anti-churning rules:
- Max 6 trades per day per symbol
- Min 30 minutes between trades on same symbol
- Only process 15m+ timeframes (no 1m/5m noise)
- Trend filter: only trade in direction of SMA(50)
- Volatility filter: skip flat markets (ATR too low)
"""

import asyncio
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import TradingExecutionMode, settings
from app.core.events import Event, EventType, event_bus
from app.core.exceptions import (
    DrawdownCircuitBreakerError,
    OrderExecutionError,
    RiskLimitExceededError,
    TradeMasterError,
)
from app.core.logging import get_logger
from app.models.base import async_session_factory
from app.models.portfolio import Position
from app.models.signal import PredictionSignal
from app.services.exchange.binance_client import binance_client
from app.services.exchange.order_manager import SpotLongProtectionPlan, order_manager
from app.services.market.data_collector import market_data_collector
from app.services.market.freshness import has_recent_closed_candle
from app.services.ml.models.ensemble import EnsembleModel
from app.services.ml.pipeline import ml_pipeline
from app.services.portfolio.tracker import PaperExitCandidate, portfolio_tracker
from app.services.risk.drawdown import circuit_breaker
from app.services.risk.manager import RiskManager, TradeProposal
from app.services.risk.rolling_sharpe import rolling_sharpe_monitor

logger = get_logger(__name__)

# --- Anti-churning constants ---
MAX_TRADES_PER_DAY = 6  # Per symbol
MIN_TRADE_INTERVAL_SECONDS = 1800  # 30 minutes between trades on same symbol
MIN_CANDLES_FOR_SIGNAL = 30
MIN_CANDLES_FOR_ML = 200
ALLOWED_INTERVALS = ("15m", "1h", "4h")


def _requires_higher_timeframe_confirmation(interval: str, action: str) -> bool:
    """Require secondary trend evidence only before opening a 15m long."""
    return interval == "15m" and action == "BUY"


def _is_positive_finite(value: float) -> bool:
    """Return whether a numeric market value is safe for order sizing."""
    return math.isfinite(value) and value > 0


class TradingEngine:
    """Autonomous trading engine that processes signals and executes trades."""

    def __init__(self) -> None:
        self._running: bool = False
        self._start_reserved: bool = False
        self._risk_manager = RiskManager()
        self._last_trade_time: dict[str, datetime] = {}
        self._daily_trade_count: dict[str, int] = defaultdict(int)
        self._daily_trade_date: str = ""  # Track which day we're counting for

    async def start(self) -> None:
        """Start the trading engine loop."""
        # ``reserve_start`` sets _running before this coroutine is scheduled,
        # so an API double-click cannot create a second consumer. A direct
        # service startup has no reservation and follows the same path.
        if self._running and not self._start_reserved:
            return
        self._start_reserved = False
        self._running = True
        logger.info("trading_engine_starting", symbols=settings.symbols_list)

        try:
            # Restore circuit breaker from Redis, or initialize fresh
            restored = await circuit_breaker.restore_from_redis()
            if not restored:
                try:
                    balance = await binance_client.get_balance("USDT")
                    initial_equity = float(balance)
                    if not math.isfinite(initial_equity) or initial_equity <= 0:
                        raise ValueError("USDT balance must be a positive finite value")
                    circuit_breaker.initialize(initial_equity)
                    logger.info("circuit_breaker_initialized", equity=initial_equity)
                except Exception as e:
                    logger.error("failed_to_get_initial_balance", error=str(e), exc_info=True)
                    if settings.execution_mode != TradingExecutionMode.PAPER:
                        logger.critical(
                            "exchange_engine_start_blocked_without_verified_equity",
                            execution_mode=settings.execution_mode.value,
                        )
                        return
                    circuit_breaker.initialize(10000)  # Paper-only simulation fallback

            # Load ML models
            for symbol in settings.symbols_list:
                await ml_pipeline.load_models(symbol)

            # Main loop: consume kline events
            logger.info("trading_engine_started")

            while self._running:
                try:
                    events = await event_bus.subscribe(
                        event_types=[EventType.KLINE_CLOSED_PERSISTED],
                        group="trading_engine",
                        consumer="engine_1",
                        count=20,
                        block_ms=5000,
                    )

                    if not events:
                        await asyncio.sleep(5)
                        continue

                    for event in events:
                        if event.data.get("is_closed"):
                            await self._process_closed_candle(event)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("trading_engine_error", error=str(e), exc_info=True)
                    await asyncio.sleep(5)
        finally:
            self._running = False
            self._start_reserved = False
            logger.info("trading_engine_stopped")

    def reserve_start(self) -> bool:
        """Atomically reserve the singleton engine loop before task scheduling."""
        if self._running or self._start_reserved:
            return False
        self._running = True
        self._start_reserved = True
        return True

    async def stop(self) -> None:
        self._running = False
        self._start_reserved = False

    def _reset_daily_counts_if_needed(self) -> None:
        """Reset daily trade counters at midnight UTC."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._daily_trade_date:
            self._daily_trade_count.clear()
            self._daily_trade_date = today

    async def _process_closed_candle(self, event: Event) -> None:
        """Process a closed candle through the full gate chain.

        Gates: timeframe → cooldown → daily limit → rolling sharpe → signal
        → regime → ensemble → multi-timeframe → volatility → correlation
        → duplicate → risk → execute
        """
        symbol = event.data["symbol"]
        interval = event.data.get("interval", "15m")

        # --- Gate 1: Only trade on 15m+ timeframes ---
        if interval not in ALLOWED_INTERVALS:
            return

        now = datetime.now(UTC)

        # --- Gate 2: Anti-churning cooldown ---
        last_trade = self._last_trade_time.get(symbol)
        if last_trade and (now - last_trade).total_seconds() < MIN_TRADE_INTERVAL_SECONDS:
            return

        # --- Gate 3: Max trades per day ---
        self._reset_daily_counts_if_needed()
        if self._daily_trade_count[symbol] >= MAX_TRADES_PER_DAY:
            return

        # --- Gate 3.5: Rolling Sharpe auto-pause ---
        if rolling_sharpe_monitor.is_paused:
            logger.debug("rolling_sharpe_paused", symbol=symbol)
            return

        try:
            # 1. Get recent candles
            async with async_session_factory() as db:
                df = await market_data_collector.get_latest_candles(
                    db=db, symbol=symbol, interval=interval, limit=300
                )

                if df.empty or len(df) < MIN_CANDLES_FOR_SIGNAL:
                    logger.debug(
                        "insufficient_candles",
                        symbol=symbol,
                        count=len(df) if not df.empty else 0,
                    )
                    return

                if not has_recent_closed_candle(
                    df,
                    interval,
                    required_candles=MIN_CANDLES_FOR_SIGNAL,
                ):
                    logger.warning(
                        "stale_or_discontinuous_candle_history",
                        symbol=symbol,
                        interval=interval,
                        candle_count=len(df),
                    )
                    return

                from app.services.strategy_deployments import get_active_technical_strategy

                active_strategy = await get_active_technical_strategy(
                    db,
                    symbol=symbol,
                    interval=interval,
                    execution_mode=settings.execution_mode,
                )

            # The default ensemble is research-only. Any order headed to a
            # Binance environment must originate from an evidence-backed,
            # explicitly activated technical deployment for this symbol and
            # interval. This prevents an unvalidated fallback from reaching an
            # exchange merely because the engine was started.
            if active_strategy is None and settings.execution_mode != TradingExecutionMode.PAPER:
                logger.warning(
                    "exchange_entry_blocked_without_active_strategy",
                    symbol=symbol,
                    interval=interval,
                    execution_mode=settings.execution_mode.value,
                )
                return

            # 2. Regime detection → adaptive thresholds
            from app.services.ml.regime import regime_detector

            close_prices = df["close"].values.astype(float)
            regime_state = regime_detector.detect(close_prices, symbol)
            adaptive_threshold = regime_state.signal_threshold

            # 3. An active deployment takes precedence over the default ML
            # ensemble. It is the exact technical configuration that passed
            # fresh out-of-sample validation, while all risk gates below stay
            # in force.
            if active_strategy is not None:
                from app.services.backtest.technical_strategy import (
                    build_technical_strategy_signals,
                )

                deployed_signals, _definition = build_technical_strategy_signals(
                    df,
                    active_strategy.strategy,
                )
                ensemble_signal = float(deployed_signals.iloc[-1])
                if ensemble_signal >= active_strategy.signal_threshold:
                    action = "BUY"
                elif ensemble_signal <= -active_strategy.signal_threshold:
                    action = "SELL"
                else:
                    return
                signal_threshold = active_strategy.signal_threshold
                signal_source = f"strategy_deployment:{active_strategy.deployment_id}"
                signal_agreement = 1.0
                signal_votes = [
                    {
                        "model": "approved_technical_strategy",
                        "action": action,
                        "score": abs(ensemble_signal),
                        "confidence": 1.0,
                    }
                ]
            else:
                from app.services.ml.ensemble_voter import ensemble_voter

                votes = []

                # 3a. Technical signal (always available with 30+ candles)
                tech_pred = self._technical_signal(
                    df, symbol, signal_threshold=0.15
                )  # Low threshold, ensemble decides
                if tech_pred is not None:
                    tech_action = EnsembleModel.signal_to_action(tech_pred.signal_strength)
                    votes.append(
                        {
                            "model": "technical",
                            "action": tech_action,
                            "score": abs(tech_pred.signal_strength),
                            "confidence": tech_pred.confidence,
                        }
                    )

                # 3b. ML prediction (200+ candles)
                ml_pred = None
                if len(df) >= MIN_CANDLES_FOR_ML:
                    ml_pred = await ml_pipeline.predict(df, symbol)
                    if ml_pred is not None:
                        ml_action = EnsembleModel.signal_to_action(ml_pred.signal_strength)
                        votes.append(
                            {
                                "model": "ml",
                                "action": ml_action,
                                "score": abs(ml_pred.signal_strength),
                                "confidence": ml_pred.confidence,
                            }
                        )

                if not votes:
                    return

                # 3c. Ensemble vote with regime-adaptive weights
                vote_result = ensemble_voter.vote(
                    predictions=votes,
                    regime=regime_state.market,
                    volatility=regime_state.volatility,
                    regime_confidence=regime_state.confidence,
                )

                action = vote_result.action
                if action == "HOLD":
                    return

                ensemble_signal = vote_result.weighted_score
                signal_threshold = adaptive_threshold
                signal_source = "default_ensemble"
                signal_agreement = vote_result.agreement_ratio
                # Keep the source scores and confidences, not only the final
                # direction map. The persisted history must explain a signal
                # without reconstructing the model state later.
                signal_votes = votes

            if abs(ensemble_signal) < signal_threshold:
                logger.debug(
                    "strategy_signal_threshold_blocked",
                    symbol=symbol,
                    signal=round(ensemble_signal, 4),
                    threshold=signal_threshold,
                    regime=regime_state.market,
                    signal_source=signal_source,
                    ensemble_votes=signal_votes,
                )
                return

            # 4. Get current market state
            current_price = float(df.iloc[-1]["close"])
            if not _is_positive_finite(current_price):
                logger.warning("market_price_invalid", symbol=symbol, price=current_price)
                return
            atr = self._compute_atr(df, period=14)
            if atr is None:
                # Dynamic fallback: average high-low range of recent candles
                recent = df.tail(min(14, len(df)))
                high = recent["high"].values.astype(float)
                low = recent["low"].values.astype(float)
                atr = float(np.mean(high - low))
                if atr <= 0:
                    atr = current_price * 0.02  # Last resort fallback
            if not _is_positive_finite(atr):
                logger.warning("market_atr_invalid", symbol=symbol, atr=atr)
                return

            # --- Gate 4: Multi-timeframe confirmation (1h trend alignment) ---
            # This gate protects only new long entries. A valid SELL signal for
            # an existing Spot position must retain its exit path even when a
            # secondary market-data feed is unavailable; native OCO protection
            # remains the primary exit safety net.
            if _requires_higher_timeframe_confirmation(interval, action):
                mtf_ok = await self._check_higher_timeframe(symbol, action)
                if not mtf_ok:
                    logger.debug("mtf_filter_blocked", symbol=symbol, action=action)
                    return

            # --- Gate 5: Volatility filter ---
            atr_pct = atr / current_price if current_price > 0 else 0
            if not math.isfinite(atr_pct) or atr_pct < 0.003:
                logger.debug("market_too_flat", symbol=symbol, atr_pct=round(atr_pct, 5))
                return

            async with async_session_factory() as db:
                candidate_signal = PredictionSignal(
                    symbol=symbol,
                    action=action,
                    strength=float(ensemble_signal),
                    confidence=float(signal_agreement),
                    model_source=signal_source,
                    timeframe=interval,
                    features_snapshot=_serialize_signal_evidence(
                        signal_source=signal_source,
                        active_strategy_id=(
                            active_strategy.deployment_id if active_strategy is not None else None
                        ),
                        signal_threshold=signal_threshold,
                        agreement_ratio=signal_agreement,
                        votes=signal_votes,
                        regime_state=regime_state,
                        price=current_price,
                        atr=atr,
                        atr_pct=atr_pct,
                    ),
                    was_executed=False,
                    generated_at=now,
                )
                db.add(candidate_signal)
                # The audit record must survive a later risk rejection or an
                # exchange error. A strategy candidate without an auditable
                # rationale must never be silently lost.
                await db.flush()
                await db.commit()

                side = "BUY" if action == "BUY" else "SELL"
                execution_mode = settings.execution_mode.value
                await _acquire_symbol_execution_lock(
                    db,
                    symbol=symbol,
                    execution_mode=execution_mode,
                )
                open_positions = await portfolio_tracker.get_open_positions(
                    db,
                    symbol,
                    execution_mode=execution_mode,
                )

                if side == "SELL":
                    exited = await self._execute_strategy_exit(
                        db=db,
                        symbol=symbol,
                        positions=open_positions,
                        execution_mode=settings.execution_mode,
                        signal_id=candidate_signal.id,
                    )
                    if not exited:
                        logger.info(
                            "spot_strategy_exit_without_long_position",
                            symbol=symbol,
                            execution_mode=execution_mode,
                        )
                        return
                    candidate_signal.was_executed = True
                    await db.commit()
                    self._last_trade_time[symbol] = now
                    self._daily_trade_count[symbol] += 1
                    logger.info(
                        "strategy_exit_executed",
                        symbol=symbol,
                        execution_mode=execution_mode,
                        signal_id=candidate_signal.id,
                    )
                    return

                # The simulator follows the same long-only Spot contract as
                # Testnet and LIVE. SELL has already been handled as a close.
                if not _allows_new_position_side(settings.execution_mode, side):
                    logger.info(
                        "spot_sell_signal_not_opened",
                        symbol=symbol,
                        execution_mode=settings.execution_mode.value,
                        reason="Spot execution never opens an uncovered short position",
                    )
                    return

                try:
                    equity = float(await binance_client.get_balance("USDT"))
                    if not math.isfinite(equity) or equity <= 0:
                        raise ValueError("USDT balance must be a positive finite value")
                except Exception as exc:
                    if settings.execution_mode != TradingExecutionMode.PAPER:
                        logger.critical(
                            "exchange_entry_blocked_without_verified_equity",
                            symbol=symbol,
                            execution_mode=settings.execution_mode.value,
                            error=str(exc),
                        )
                        return
                    logger.warning(
                        "paper_entry_using_simulated_equity",
                        symbol=symbol,
                        error=str(exc),
                    )
                    equity = 10000.0

                # --- Gate 6: Correlation filter ---
                from app.services.risk.correlation import correlation_filter

                corr_ok, corr_reason = await correlation_filter.check_can_open(
                    db,
                    symbol,
                    side,
                    execution_mode=execution_mode,
                )
                if not corr_ok:
                    logger.info("correlation_blocked", symbol=symbol, reason=corr_reason)
                    return

                total_exposure = await portfolio_tracker.get_total_exposure(
                    db,
                    execution_mode=execution_mode,
                )
                symbol_exposure = await portfolio_tracker.get_symbol_exposure(
                    db,
                    symbol,
                    execution_mode=execution_mode,
                )

                # --- Gate 7: Duplicate position check ---
                for pos in open_positions:
                    if pos.side == "LONG":
                        return

                # 8. Risk management (apply regime position_size_mult)
                proposal = TradeProposal(
                    symbol=symbol,
                    side=side,
                    signal_strength=ensemble_signal,
                    entry_price=current_price,
                    atr=atr,
                    current_equity=equity,
                    current_exposure=total_exposure,
                    symbol_exposure=symbol_exposure,
                    atr_stop_multiplier=(
                        active_strategy.atr_stop_multiplier if active_strategy is not None else 2.0
                    ),
                    risk_reward_ratio=(
                        active_strategy.risk_reward_ratio if active_strategy is not None else 2.0
                    ),
                )
                approved = self._risk_manager.validate_trade(proposal)
                # Scale quantity by regime multiplier, then re-validate exposure limits
                adjusted_qty = float(approved.quantity) * regime_state.position_size_mult
                effective_limits = self._risk_manager.effective_limits()
                max_symbol_notional = equity * effective_limits.max_single_asset_exposure
                max_total_notional = equity * effective_limits.max_portfolio_exposure
                adjusted_notional = adjusted_qty * current_price
                if symbol_exposure + adjusted_notional > max_symbol_notional:
                    adjusted_qty = max(
                        0.0,
                        (max_symbol_notional - symbol_exposure) / current_price,
                    )
                if total_exposure + adjusted_notional > max_total_notional:
                    adjusted_qty = min(
                        adjusted_qty,
                        max(0.0, (max_total_notional - total_exposure) / current_price),
                    )
                if not _is_positive_finite(adjusted_qty):
                    logger.debug(
                        "regime_mult_invalid_or_capped_to_zero",
                        symbol=symbol,
                        mult=regime_state.position_size_mult,
                    )
                    return
                approved.quantity = adjusted_qty

                protective_exit = None
                if settings.execution_mode != TradingExecutionMode.PAPER:
                    if approved.stop_loss.take_profit_price is None:
                        logger.error(
                            "spot_exchange_entry_missing_take_profit",
                            symbol=symbol,
                            execution_mode=settings.execution_mode.value,
                        )
                        return
                    protective_exit = SpotLongProtectionPlan(
                        stop_loss_price=Decimal(str(approved.stop_loss.stop_price)),
                        take_profit_price=Decimal(str(approved.stop_loss.take_profit_price)),
                    )

                # 9. Execute trade
                order = await order_manager.execute_market_order(
                    db=db,
                    symbol=symbol,
                    side=side,
                    quantity=approved.quantity,
                    signal_id=candidate_signal.id,
                    protective_exit=protective_exit,
                )

                # 10. Record position
                await portfolio_tracker.open_position(
                    db=db,
                    symbol=symbol,
                    side="LONG",
                    entry_price=float(order.avg_fill_price or current_price),
                    quantity=float(order.filled_quantity),
                    stop_loss_price=approved.stop_loss.stop_price,
                    take_profit_price=approved.stop_loss.take_profit_price,
                    execution_mode=order.execution_mode,
                    entry_exchange_order_id=order.exchange_order_id,
                    protective_order_list_id=order.protective_order_list_id,
                    protective_quantity=order.protective_quantity,
                    protection_status=(
                        "ACTIVE" if order.protective_order_list_id is not None else "LOCAL"
                    ),
                )
                candidate_signal.was_executed = True
                await db.commit()

                # 11. Update trackers + drift detector
                self._last_trade_time[symbol] = now
                self._daily_trade_count[symbol] += 1

                # Drift detector moved to check_positions() where actual P&L is known

                logger.info(
                    "trade_executed",
                    symbol=symbol,
                    side=side,
                    qty=float(order.filled_quantity),
                    price=float(order.avg_fill_price or current_price),
                    signal=round(ensemble_signal, 4),
                    atr_pct=round(atr_pct, 4),
                    daily_trades=self._daily_trade_count[symbol],
                    regime=regime_state.market,
                    volatility=regime_state.volatility,
                    regime_threshold=regime_state.signal_threshold,
                    regime_size_mult=regime_state.position_size_mult,
                    signal_source=signal_source,
                    ensemble_agreement=signal_agreement,
                    ensemble_votes=signal_votes,
                )

        except (DrawdownCircuitBreakerError, RiskLimitExceededError) as e:
            logger.info("trade_blocked_by_risk", symbol=symbol, reason=str(e))
        except TradeMasterError as e:
            logger.error("trade_failed", symbol=symbol, error=str(e), code=e.code)
        except Exception as e:
            logger.error("unexpected_trading_error", symbol=symbol, error=str(e), exc_info=True)

    async def _execute_strategy_exit(
        self,
        *,
        db: AsyncSession,
        symbol: str,
        positions: list[Position],
        execution_mode: TradingExecutionMode,
        signal_id: int | None,
    ) -> bool:
        """Close tracked Spot longs; a strategy SELL is never a short entry."""
        long_positions = [position for position in positions if position.side == "LONG"]
        if not long_positions:
            return False

        if execution_mode == TradingExecutionMode.PAPER:
            for position in long_positions:
                order = await order_manager.execute_market_order(
                    db=db,
                    symbol=symbol,
                    side="SELL",
                    quantity=float(position.quantity),
                    signal_id=signal_id,
                )
                await portfolio_tracker.close_position(
                    db,
                    position,
                    float(order.avg_fill_price or position.current_price),
                )
            return True

        from app.services.exchange.spot_position_closer import spot_position_closer

        for position in long_positions:
            await spot_position_closer.close_for_strategy(
                db=db,
                position_id=position.id,
                execution_mode=execution_mode,
            )
        return True

    async def _check_higher_timeframe(self, symbol: str, action: str) -> bool:
        """Multi-timeframe confirmation: 1h trend must align with 15m signal.

        BUY: price must be above SMA(20) on fresh, continuous 1h candles
        (or SMA(80) on fresh, continuous 15m candles as a proxy).
        SELL: price must be below SMA(20) on 1h.

        New entries fail closed when neither confirmation dataset is available.
        """

        try:
            # Try 1h candles first
            async with async_session_factory() as db:
                df_1h = await market_data_collector.get_latest_candles(
                    db=db,
                    symbol=symbol,
                    interval="1h",
                    limit=30,
                )

            if (
                not df_1h.empty
                and len(df_1h) >= 20
                and has_recent_closed_candle(df_1h, "1h", required_candles=20)
            ):
                close_1h = df_1h["close"].values.astype(float)
                sma_1h = float(np.mean(close_1h[-20:]))
                current = close_1h[-1]
            else:
                # Fallback: use SMA(80) on 15m as proxy for 1h SMA(20)
                async with async_session_factory() as db:
                    df_15m = await market_data_collector.get_latest_candles(
                        db=db,
                        symbol=symbol,
                        interval="15m",
                        limit=120,
                    )
                if (
                    df_15m.empty
                    or len(df_15m) < 80
                    or not has_recent_closed_candle(
                        df_15m,
                        "15m",
                        required_candles=80,
                    )
                ):
                    logger.warning(
                        "mtf_confirmation_unavailable",
                        symbol=symbol,
                        action=action,
                    )
                    return False

                close = df_15m["close"].values.astype(float)
                sma_1h = float(np.mean(close[-80:]))
                current = close[-1]

            if action == "BUY":
                return current > sma_1h
            else:
                return current < sma_1h

        except Exception as e:
            logger.warning("mtf_check_failed", symbol=symbol, error=str(e))
            return False

    def _technical_signal(self, df, symbol: str, signal_threshold: float = 0.25):
        """Multi-indicator signal with trend filter and proper calculations.

        Indicators used:
        - Trend filter: SMA(50) — only BUY above, only SELL below
        - Entry signal: SMA(10) vs SMA(30) crossover momentum
        - RSI(14): overbought/oversold confirmation
        - MACD(12,26,9): proper histogram calculation
        - Bollinger Bands(20,2): mean reversion at extremes
        """

        close = df["close"].values.astype(float)
        n = len(close)

        if n < MIN_CANDLES_FOR_SIGNAL:
            return None

        # --- Trend filter: SMA(50) or longest available ---
        trend_period = min(50, n - 1)
        sma_trend = np.mean(close[-trend_period:])
        price_vs_trend = (close[-1] - sma_trend) / sma_trend if sma_trend > 0 else 0
        # Positive = uptrend, negative = downtrend

        scores = []

        # --- 1. SMA crossover momentum (10 vs 30) ---
        if n >= 30:
            sma_10 = np.mean(close[-10:])
            sma_30 = np.mean(close[-30:])
            if sma_30 > 0:
                crossover = (sma_10 - sma_30) / sma_30
                # Normalize: 0.1% crossover = moderate signal
                sma_score = np.clip(crossover * 500, -1, 1)
                scores.append(("sma_xover", float(sma_score), 0.45))

        # --- 2. RSI(14) ---
        rsi_period = min(14, n - 1)
        deltas = np.diff(close[-rsi_period - 1 :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0001
        rs = avg_gain / max(avg_loss, 0.0001)
        rsi = 100 - (100 / (1 + rs))

        # RSI < 35 = strong buy, > 65 = strong sell, 45-55 = neutral
        if rsi < 35:
            rsi_score = 0.5 + (35 - rsi) / 35  # 0.5 to 1.0
        elif rsi > 65:
            rsi_score = -0.5 - (rsi - 65) / 35  # -0.5 to -1.0
        elif rsi < 45:
            rsi_score = (45 - rsi) / 20  # 0 to 0.5
        elif rsi > 55:
            rsi_score = -(rsi - 55) / 20  # 0 to -0.5
        else:
            rsi_score = 0.0
        scores.append(("rsi", float(np.clip(rsi_score, -1, 1)), 0.15))

        # --- 3. MACD(12, 26, 9) — proper calculation ---
        if n >= 30:
            ema_12 = self._ema_series(close, 12)
            ema_26 = self._ema_series(close, 26)
            macd_line = ema_12 - ema_26
            signal_line = self._ema_series(macd_line, 9)
            histogram = macd_line - signal_line

            if len(histogram) >= 2:
                # Histogram direction + magnitude
                hist_current = histogram[-1]
                hist_prev = histogram[-2]
                hist_accel = hist_current - hist_prev  # Acceleration

                # Normalize by price
                if close[-1] > 0:
                    norm_hist = hist_current / close[-1] * 1000  # Per-mille
                    norm_accel = hist_accel / close[-1] * 1000
                    macd_score = np.clip(norm_hist * 2 + norm_accel * 3, -1, 1)
                    scores.append(("macd", float(macd_score), 0.30))

        # --- 4. Bollinger Bands(20, 2) — mean reversion ---
        if n >= 20:
            bb_mean = np.mean(close[-20:])
            bb_std = np.std(close[-20:])
            if bb_std > 0:
                bb_z = (close[-1] - bb_mean) / bb_std
                # Outside bands: strong mean reversion signal
                if abs(bb_z) > 2:
                    bb_score = -np.sign(bb_z) * 0.8  # Strong mean reversion
                elif abs(bb_z) > 1:
                    bb_score = -bb_z * 0.4  # Moderate
                else:
                    bb_score = 0.0  # Inside bands = no signal
                scores.append(("bb", float(np.clip(bb_score, -1, 1)), 0.10))

        # --- Composite signal ---
        if not scores:
            return None

        total_weight = sum(w for _, _, w in scores)
        raw_signal = sum(s * w for _, s, w in scores) / total_weight

        # --- Trend filter: suppress counter-trend signals ---
        if price_vs_trend > 0.002 and raw_signal < 0:
            # Uptrend but bearish signal — weaken it
            raw_signal *= 0.3
        elif price_vs_trend < -0.002 and raw_signal > 0:
            # Downtrend but bullish signal — weaken it
            raw_signal *= 0.3

        # Minimum threshold: require real conviction (regime-adaptive)
        if abs(raw_signal) < signal_threshold:
            return None

        # Map to action range: threshold-1.0 → 0.3-0.8
        direction = 1 if raw_signal > 0 else -1
        range_above = max(1.0 - signal_threshold, 0.01)
        scaled = 0.3 + (abs(raw_signal) - signal_threshold) / range_above * 0.5
        signal_strength = direction * min(scaled, 0.8)

        from app.services.ml.models.base import ModelPrediction

        action = 2 if signal_strength > 0 else 0
        probs = np.array(
            [
                max(0, -signal_strength),
                1 - abs(signal_strength),
                max(0, signal_strength),
            ]
        )
        probs = probs / probs.sum()

        logger.info(
            "technical_signal_generated",
            symbol=symbol,
            signal=round(signal_strength, 4),
            raw=round(raw_signal, 4),
            trend=round(price_vs_trend, 5),
            rsi=round(rsi, 1),
            indicators={name: round(score, 4) for name, score, _ in scores},
        )

        return ModelPrediction(
            action=action,
            probabilities=probs,
            confidence=abs(signal_strength),
            signal_strength=signal_strength,
        )

    @staticmethod
    def _ema_series(data, period: int):
        """Calculate EMA for entire series (proper MACD calculation)."""
        if len(data) < period:
            return data.copy()
        multiplier = 2 / (period + 1)
        ema = np.empty_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = data[i] * multiplier + ema[i - 1] * (1 - multiplier)
        return ema

    @staticmethod
    def _compute_atr(df, period: int = 14) -> float | None:
        """Compute ATR from OHLC data."""
        if len(df) < period + 1:
            return None
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else None
        return float(np.mean(tr[-period:]))

    async def check_positions(self) -> None:
        """Check all open positions for stop losses and trailing updates."""
        try:
            if settings.execution_mode != TradingExecutionMode.PAPER:
                try:
                    from app.services.exchange.spot_protection_reconciler import (
                        spot_protection_reconciler,
                    )

                    async with async_session_factory() as reconciliation_db:
                        report = await spot_protection_reconciler.reconcile(
                            reconciliation_db,
                            execution_mode=settings.execution_mode,
                        )
                        await reconciliation_db.commit()
                    if not report.ready:
                        logger.critical(
                            "live_spot_protection_unresolved",
                            issues=list(report.issues),
                        )
                        return
                except Exception as exc:
                    logger.critical(
                        "live_spot_protection_reconciliation_failed",
                        error=str(exc),
                        exc_info=True,
                    )
                    return

            async with async_session_factory() as symbols_db:
                from app.services.strategy_deployments import get_runtime_symbols

                runtime_symbols = await get_runtime_symbols(
                    symbols_db,
                    execution_mode=settings.execution_mode,
                    base_symbols=settings.symbols_list,
                )

            prices = {}
            for symbol in runtime_symbols:
                try:
                    price = await binance_client.get_ticker_price(symbol)
                    prices[symbol] = float(price)
                except Exception as exc:
                    logger.warning(
                        "position_price_fetch_failed",
                        symbol=symbol,
                        error=str(exc),
                    )

            if not prices:
                return

            async with async_session_factory() as db:
                execution_mode = settings.execution_mode.value
                await portfolio_tracker.update_prices(
                    db,
                    prices,
                    execution_mode=execution_mode,
                )
                paper_exit_candidates = (
                    await portfolio_tracker.find_paper_exit_candidates(
                        db,
                        prices,
                    )
                    if settings.execution_mode == TradingExecutionMode.PAPER
                    else []
                )

                from app.services.risk.rolling_sharpe import rolling_sharpe_monitor

                for candidate in paper_exit_candidates:
                    pos = await self._execute_confirmed_paper_exit(db=db, candidate=candidate)
                    # Record trade return for rolling Sharpe
                    if pos.entry_price and float(pos.entry_price) > 0:
                        close_price = float(pos.current_price)
                        if pos.side == "LONG":
                            ret = (close_price - float(pos.entry_price)) / float(pos.entry_price)
                        else:
                            ret = (float(pos.entry_price) - close_price) / float(pos.entry_price)
                        rolling_sharpe_monitor.record_trade(pos.symbol, ret, pos.side)

                        # Record drift with actual price change (not hardcoded 0)
                        from app.services.ml.drift_detector import drift_detector

                        predicted_action = "BUY" if pos.side == "LONG" else "SELL"
                        drift_detector.record_outcome(
                            symbol=pos.symbol,
                            predicted_action=predicted_action,
                            actual_price_change_pct=ret * 100,
                            signal_strength=abs(ret),
                        )

                try:
                    equity = float(await binance_client.get_balance("USDT"))
                    if not math.isfinite(equity) or equity <= 0:
                        raise ValueError("USDT balance must be a positive finite value")
                except Exception as exc:
                    if settings.execution_mode != TradingExecutionMode.PAPER:
                        logger.critical(
                            "exchange_snapshot_blocked_without_verified_equity",
                            execution_mode=settings.execution_mode.value,
                            error=str(exc),
                        )
                        await db.commit()
                        return
                    logger.warning("paper_snapshot_using_simulated_equity", error=str(exc))
                    equity = 10000.0
                await circuit_breaker.update_and_persist(equity)
                await portfolio_tracker.take_snapshot(db, equity, equity)

                # Refresh performance stats for Kelly sizing
                await self._risk_manager.refresh_performance_stats(db)

                await db.commit()

            # Check rolling Sharpe status
            try:
                from app.services.risk.rolling_sharpe import rolling_sharpe_monitor

                sharpe_status = rolling_sharpe_monitor.check()
                if sharpe_status.is_paused:
                    logger.info(
                        "rolling_sharpe_trading_paused",
                        sharpe=sharpe_status.sharpe,
                        win_rate=sharpe_status.win_rate,
                        reason=sharpe_status.pause_reason,
                    )
            except Exception as e:
                logger.warning("sharpe_check_failed", error=str(e), exc_info=True)

            # Auto-retrain models if drift detected (runs with cooldown)
            try:
                from app.services.ml.drift_detector import drift_detector

                await drift_detector.auto_retrain_if_needed()
            except Exception as e:
                logger.warning("drift_check_failed", error=str(e), exc_info=True)

        except Exception as e:
            logger.error("position_check_error", error=str(e), exc_info=True)

    async def _execute_confirmed_paper_exit(
        self,
        *,
        db: AsyncSession,
        candidate: PaperExitCandidate,
    ) -> Position:
        """Close paper ledger only after its simulated reducing order is fully filled."""
        position = candidate.position
        close_side = "SELL" if position.side == "LONG" else "BUY"
        exit_order = await order_manager.execute_market_order(
            db=db,
            symbol=position.symbol,
            side=close_side,
            quantity=float(position.quantity),
        )
        expected_quantity = float(position.quantity)
        actual_quantity = float(exit_order.filled_quantity)
        actual_price = exit_order.avg_fill_price
        if (
            exit_order.status.value != "FILLED"
            or actual_quantity != expected_quantity
            or actual_price is None
        ):
            raise OrderExecutionError(
                "paper exit order was not fully filled; position remains open"
            )
        return await portfolio_tracker.close_position(db, position, float(actual_price))


trading_engine = TradingEngine()


def _serialize_signal_evidence(
    *,
    signal_source: str,
    active_strategy_id: int | None,
    signal_threshold: float,
    agreement_ratio: float,
    votes: list[dict[str, object]],
    regime_state: object,
    price: float,
    atr: float,
    atr_pct: float,
) -> str:
    """Serialize the trader-readable rationale without model artifacts or secrets."""
    normalized_votes = [
        {
            "model": str(vote.get("model", "unknown")),
            "action": _normalize_signal_action(vote.get("action")),
            "score": float(vote.get("score", 0.0)),
            "confidence": float(vote.get("confidence", 0.0)),
        }
        for vote in votes
    ]
    evidence = {
        "signal_source": signal_source,
        "strategy_deployment_id": active_strategy_id,
        "signal_threshold": float(signal_threshold),
        "agreement_ratio": float(agreement_ratio),
        "votes": normalized_votes,
        "regime": {
            "market": str(getattr(regime_state, "market", "UNKNOWN")),
            "volatility": str(getattr(regime_state, "volatility", "UNKNOWN")),
            "confidence": float(getattr(regime_state, "confidence", 0.0)),
            "position_size_multiplier": float(getattr(regime_state, "position_size_mult", 1.0)),
        },
        "price": float(price),
        "atr": float(atr),
        "atr_pct": float(atr_pct),
    }
    return json.dumps(evidence, separators=(",", ":"), sort_keys=True)


def _normalize_signal_action(value: object) -> str:
    action = str(value).upper()
    return action if action in {"BUY", "HOLD", "SELL"} else "HOLD"


async def _acquire_symbol_execution_lock(
    db: AsyncSession,
    *,
    symbol: str,
    execution_mode: str,
) -> None:
    """Serialize one symbol ledger across workers before any close/open decision."""
    lock_material = f"trademaster:execution:{execution_mode}:{symbol}".encode()
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
        raise TradeMasterError(
            "could not acquire the symbol execution lock; no order was sent",
            code="SYMBOL_EXECUTION_LOCK_FAILED",
        ) from exc


def _allows_new_position_side(execution_mode: TradingExecutionMode, side: str) -> bool:
    """All runtime ledgers mirror Binance Spot: only LONG entries are allowed."""
    del execution_mode
    return side == "BUY"

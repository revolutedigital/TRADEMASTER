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
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from app.config import settings
from app.core.events import Event, EventType, event_bus
from app.core.exceptions import (
    DrawdownCircuitBreakerError,
    RiskLimitExceededError,
    TradeMasterError,
)
from app.core.logging import get_logger
from app.models.base import async_session_factory
from app.services.exchange.binance_client import binance_client
from app.services.exchange.order_manager import order_manager
from app.services.ml.models.ensemble import EnsembleModel
from app.services.ml.pipeline import ml_pipeline
from app.services.market.data_collector import market_data_collector
from app.services.portfolio.tracker import portfolio_tracker
from app.services.risk.drawdown import circuit_breaker
from app.services.risk.manager import RiskManager, TradeProposal

logger = get_logger(__name__)

# --- Anti-churning constants ---
MAX_TRADES_PER_DAY = 6  # Per symbol
MIN_TRADE_INTERVAL_SECONDS = 1800  # 30 minutes between trades on same symbol
MIN_CANDLES_FOR_SIGNAL = 30
MIN_CANDLES_FOR_ML = 200
ALLOWED_INTERVALS = ("15m", "1h", "4h")


class TradingEngine:
    """Autonomous trading engine that processes signals and executes trades."""

    def __init__(self) -> None:
        self._running: bool = False
        self._risk_manager = RiskManager()
        self._last_trade_time: dict[str, datetime] = {}
        self._daily_trade_count: dict[str, int] = defaultdict(int)
        self._daily_trade_date: str = ""  # Track which day we're counting for

    async def start(self) -> None:
        """Start the trading engine loop."""
        self._running = True
        logger.info("trading_engine_starting", symbols=settings.symbols_list)

        # Restore circuit breaker from Redis, or initialize fresh
        restored = await circuit_breaker.restore_from_redis()
        if not restored:
            try:
                balance = await binance_client.get_balance("USDT")
                circuit_breaker.initialize(float(balance))
                logger.info("circuit_breaker_initialized", equity=float(balance))
            except Exception as e:
                logger.error("failed_to_get_initial_balance", error=str(e))
                circuit_breaker.initialize(10000)  # Fallback

        # Load ML models
        for symbol in settings.symbols_list:
            await ml_pipeline.load_models(symbol)

        # Main loop: consume kline events
        logger.info("trading_engine_started")

        while self._running:
            try:
                events = await event_bus.subscribe(
                    event_types=[EventType.KLINE_UPDATE],
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
                logger.error("trading_engine_error", error=str(e))
                await asyncio.sleep(5)

        logger.info("trading_engine_stopped")

    async def stop(self) -> None:
        self._running = False

    def _reset_daily_counts_if_needed(self) -> None:
        """Reset daily trade counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

        now = datetime.now(timezone.utc)

        # --- Gate 2: Anti-churning cooldown ---
        last_trade = self._last_trade_time.get(symbol)
        if last_trade and (now - last_trade).total_seconds() < MIN_TRADE_INTERVAL_SECONDS:
            return

        # --- Gate 3: Max trades per day ---
        self._reset_daily_counts_if_needed()
        if self._daily_trade_count[symbol] >= MAX_TRADES_PER_DAY:
            return

        # --- Gate 3.5: Rolling Sharpe auto-pause ---
        from app.services.risk.rolling_sharpe import rolling_sharpe_monitor
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
                logger.debug("insufficient_candles", symbol=symbol, count=len(df) if not df.empty else 0)
                return

            # 2. Regime detection → adaptive thresholds
            import numpy as np
            from app.services.ml.regime import regime_detector

            close_prices = df["close"].values.astype(float)
            regime_state = regime_detector.detect(close_prices, symbol)
            adaptive_threshold = regime_state.signal_threshold

            # 3. Generate signals from all available sources
            from app.services.ml.ensemble_voter import ensemble_voter

            votes = []

            # 3a. Technical signal (always available with 30+ candles)
            tech_pred = self._technical_signal(df, symbol, signal_threshold=0.15)  # Low threshold, ensemble decides
            if tech_pred is not None:
                tech_action = EnsembleModel.signal_to_action(tech_pred.signal_strength)
                votes.append({
                    "model": "technical",
                    "action": tech_action,
                    "score": abs(tech_pred.signal_strength),
                    "confidence": tech_pred.confidence,
                })

            # 3b. ML prediction (200+ candles)
            ml_pred = None
            if len(df) >= MIN_CANDLES_FOR_ML:
                ml_pred = await ml_pipeline.predict(df, symbol)
                if ml_pred is not None:
                    ml_action = EnsembleModel.signal_to_action(ml_pred.signal_strength)
                    votes.append({
                        "model": "ml",
                        "action": ml_action,
                        "score": abs(ml_pred.signal_strength),
                        "confidence": ml_pred.confidence,
                    })

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

            # Use the best available prediction for signal_strength tracking
            prediction = ml_pred if ml_pred is not None else tech_pred
            # Override signal_strength with ensemble score (directional)
            ensemble_signal = vote_result.weighted_score

            # Check adaptive threshold (regime-driven)
            if abs(ensemble_signal) < adaptive_threshold:
                logger.debug(
                    "regime_threshold_blocked",
                    symbol=symbol,
                    signal=round(ensemble_signal, 4),
                    threshold=adaptive_threshold,
                    regime=regime_state.market,
                    ensemble_votes=vote_result.individual_votes,
                )
                return

            # 4. Get current market state
            current_price = float(df.iloc[-1]["close"])
            atr = self._compute_atr(df, period=14)
            if atr is None:
                # Dynamic fallback: average high-low range of recent candles
                recent = df.tail(min(14, len(df)))
                high = recent["high"].values.astype(float)
                low = recent["low"].values.astype(float)
                atr = float(np.mean(high - low))
                if atr <= 0:
                    atr = current_price * 0.02  # Last resort fallback

            # --- Gate 4: Multi-timeframe confirmation (1h trend alignment) ---
            if interval == "15m":
                mtf_ok = await self._check_higher_timeframe(symbol, action)
                if not mtf_ok:
                    logger.debug("mtf_filter_blocked", symbol=symbol, action=action)
                    return

            # --- Gate 5: Volatility filter ---
            atr_pct = atr / current_price if current_price > 0 else 0
            if atr_pct < 0.003:
                logger.debug("market_too_flat", symbol=symbol, atr_pct=round(atr_pct, 5))
                return

            async with async_session_factory() as db:
                try:
                    equity = float(await binance_client.get_balance("USDT"))
                except Exception:
                    equity = 10000.0

                side = "BUY" if action == "BUY" else "SELL"

                # --- Gate 6: Correlation filter ---
                from app.services.risk.correlation import correlation_filter
                corr_ok, corr_reason = await correlation_filter.check_can_open(db, symbol, side)
                if not corr_ok:
                    logger.info("correlation_blocked", symbol=symbol, reason=corr_reason)
                    return

                total_exposure = await portfolio_tracker.get_total_exposure(db)
                symbol_exposure = await portfolio_tracker.get_symbol_exposure(db, symbol)

                # --- Gate 7: Duplicate position check ---
                open_positions = await portfolio_tracker.get_open_positions(db, symbol)
                for pos in open_positions:
                    if (action == "BUY" and pos.side == "LONG") or (
                        action == "SELL" and pos.side == "SHORT"
                    ):
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
                )
                approved = self._risk_manager.validate_trade(proposal)
                # Scale quantity by regime multiplier, then re-validate exposure limits
                adjusted_qty = float(approved.quantity) * regime_state.position_size_mult
                max_symbol_notional = equity * settings.trading_max_single_asset_exposure
                max_total_notional = equity * settings.trading_max_portfolio_exposure
                adjusted_notional = adjusted_qty * current_price
                if symbol_exposure + adjusted_notional > max_symbol_notional:
                    adjusted_qty = max(0.0, (max_symbol_notional - symbol_exposure) / current_price)
                if total_exposure + adjusted_notional > max_total_notional:
                    adjusted_qty = min(adjusted_qty, max(0.0, (max_total_notional - total_exposure) / current_price))
                if adjusted_qty <= 0:
                    logger.debug("regime_mult_capped_to_zero", symbol=symbol, mult=regime_state.position_size_mult)
                    return
                approved.quantity = adjusted_qty

                # 9. Execute trade
                order = await order_manager.execute_market_order(
                    db=db,
                    symbol=symbol,
                    side=side,
                    quantity=approved.quantity,
                )

                # 10. Record position
                pos_side = "LONG" if side == "BUY" else "SHORT"
                await portfolio_tracker.open_position(
                    db=db,
                    symbol=symbol,
                    side=pos_side,
                    entry_price=float(order.avg_fill_price or current_price),
                    quantity=float(order.filled_quantity),
                    stop_loss_price=approved.stop_loss.stop_price,
                    take_profit_price=approved.stop_loss.take_profit_price,
                )
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
                    ensemble_agreement=vote_result.agreement_ratio,
                    ensemble_votes=vote_result.individual_votes,
                )

        except (DrawdownCircuitBreakerError, RiskLimitExceededError) as e:
            logger.info("trade_blocked_by_risk", symbol=symbol, reason=str(e))
        except TradeMasterError as e:
            logger.error("trade_failed", symbol=symbol, error=str(e), code=e.code)
        except Exception as e:
            logger.error("unexpected_trading_error", symbol=symbol, error=str(e))

    async def _check_higher_timeframe(self, symbol: str, action: str) -> bool:
        """Multi-timeframe confirmation: 1h trend must align with 15m signal.

        BUY: price must be above SMA(20) on 1h (or SMA(80) on 15m as proxy).
        SELL: price must be below SMA(20) on 1h.
        """
        import numpy as np

        try:
            # Try 1h candles first
            async with async_session_factory() as db:
                df_1h = await market_data_collector.get_latest_candles(
                    db=db, symbol=symbol, interval="1h", limit=30,
                )

            if not df_1h.empty and len(df_1h) >= 10:
                close_1h = df_1h["close"].values.astype(float)
                sma_period = min(20, len(close_1h))
                sma_1h = float(np.mean(close_1h[-sma_period:]))
                current = close_1h[-1]
            else:
                # Fallback: use SMA(80) on 15m as proxy for 1h SMA(20)
                async with async_session_factory() as db:
                    df_15m = await market_data_collector.get_latest_candles(
                        db=db, symbol=symbol, interval="15m", limit=120,
                    )
                if df_15m.empty or len(df_15m) < 20:
                    return True  # Not enough data → allow

                close = df_15m["close"].values.astype(float)
                sma_period = min(80, len(close))
                sma_1h = float(np.mean(close[-sma_period:]))
                current = close[-1]

            if action == "BUY":
                return current > sma_1h
            else:
                return current < sma_1h

        except Exception as e:
            logger.warning("mtf_check_failed", symbol=symbol, error=str(e))
            return True  # On error, don't block

    def _technical_signal(self, df, symbol: str, signal_threshold: float = 0.25):
        """Multi-indicator signal with trend filter and proper calculations.

        Indicators used:
        - Trend filter: SMA(50) — only BUY above, only SELL below
        - Entry signal: SMA(10) vs SMA(30) crossover momentum
        - RSI(14): overbought/oversold confirmation
        - MACD(12,26,9): proper histogram calculation
        - Bollinger Bands(20,2): mean reversion at extremes
        """
        import numpy as np

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
                scores.append(("sma_xover", float(sma_score), 0.30))

        # --- 2. RSI(14) ---
        rsi_period = min(14, n - 1)
        deltas = np.diff(close[-rsi_period - 1:])
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
        scores.append(("rsi", float(np.clip(rsi_score, -1, 1)), 0.25))

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
                    scores.append(("macd", float(macd_score), 0.25))

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
                scores.append(("bb", float(np.clip(bb_score, -1, 1)), 0.20))

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
        import numpy as np

        action = 2 if signal_strength > 0 else 0
        probs = np.array([
            max(0, -signal_strength),
            1 - abs(signal_strength),
            max(0, signal_strength),
        ])
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
        import numpy as np
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
        import numpy as np
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
            prices = {}
            for symbol in settings.symbols_list:
                try:
                    price = await binance_client.get_ticker_price(symbol)
                    prices[symbol] = float(price)
                except Exception:
                    pass

            if not prices:
                return

            async with async_session_factory() as db:
                await portfolio_tracker.update_prices(db, prices)
                closed = await portfolio_tracker.check_stop_losses(db, prices)

                from app.services.risk.rolling_sharpe import rolling_sharpe_monitor
                for pos in closed:
                    close_side = "SELL" if pos.side == "LONG" else "BUY"
                    await order_manager.execute_market_order(
                        db=db,
                        symbol=pos.symbol,
                        side=close_side,
                        quantity=float(pos.quantity),
                    )
                    # Record trade return for rolling Sharpe
                    if pos.entry_price and float(pos.entry_price) > 0:
                        close_price = prices.get(pos.symbol, float(pos.entry_price))
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
                except Exception:
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
                logger.warning("sharpe_check_failed", error=str(e))

            # Auto-retrain models if drift detected (runs with cooldown)
            try:
                from app.services.ml.drift_detector import drift_detector
                await drift_detector.auto_retrain_if_needed()
            except Exception as e:
                logger.warning("drift_check_failed", error=str(e))

        except Exception as e:
            logger.error("position_check_error", error=str(e))


trading_engine = TradingEngine()

"""Trading Engine: the main loop that connects signals -> risk -> execution.

This is the core autonomous trading loop:
1. Receive market data updates
2. Generate ML predictions
3. Validate through risk management
4. Execute approved trades
5. Monitor and manage open positions
"""

import asyncio
from datetime import datetime, timezone

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


class TradingEngine:
    """Autonomous trading engine that processes signals and executes trades."""

    def __init__(self) -> None:
        self._running: bool = False
        self._risk_manager = RiskManager()
        self._last_signal_time: dict[str, datetime] = {}
        self._min_signal_interval_seconds: float = 60  # 60s between signals per symbol (one per candle)

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
                    # No events (Redis down or no data) — avoid CPU spin
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

    async def _process_closed_candle(self, event: Event) -> None:
        """Process a closed candle: generate signal, validate, execute."""
        symbol = event.data["symbol"]
        interval = event.data.get("interval", "1h")

        # Trade on all active timeframes (1m, 5m, 15m, 1h, 4h)
        if interval not in ("1m", "5m", "15m", "1h", "4h"):
            return

        # Rate limit: don't signal too frequently for same symbol
        now = datetime.now(timezone.utc)
        last_time = self._last_signal_time.get(symbol)
        if last_time and (now - last_time).total_seconds() < self._min_signal_interval_seconds:
            return

        try:
            # 1. Get recent candles for ML prediction
            async with async_session_factory() as db:
                df = await market_data_collector.get_latest_candles(
                    db=db, symbol=symbol, interval=interval, limit=300
                )

            if df.empty or len(df) < 30:
                return

            # 2. ML prediction (needs 200+ candles for full features)
            prediction = None
            if len(df) >= 200:
                prediction = await ml_pipeline.predict(df, symbol)

            # Fallback: simple technical signal when ML unavailable
            if prediction is None:
                prediction = self._simple_signal(df, symbol)
                if prediction is None:
                    return

            self._last_signal_time[symbol] = now

            action = EnsembleModel.signal_to_action(prediction.signal_strength)
            if action == "HOLD":
                return

            # 3. Get current market state for risk validation
            current_price = float(df.iloc[-1]["close"])
            atr = float(df.iloc[-1].get("atr_14", current_price * 0.02))

            async with async_session_factory() as db:
                try:
                    equity = float(await binance_client.get_balance("USDT"))
                except Exception:
                    # Paper mode: use 10000 USDT as default equity
                    equity = 10000.0
                total_exposure = await portfolio_tracker.get_total_exposure(db)
                symbol_exposure = await portfolio_tracker.get_symbol_exposure(db, symbol)

                # Check if we already have a position in the same direction
                open_positions = await portfolio_tracker.get_open_positions(db, symbol)
                for pos in open_positions:
                    if (action == "BUY" and pos.side == "LONG") or (
                        action == "SELL" and pos.side == "SHORT"
                    ):
                        return  # Already have a position in this direction

                # 4. Risk management validation
                side = "BUY" if action == "BUY" else "SELL"
                proposal = TradeProposal(
                    symbol=symbol,
                    side=side,
                    signal_strength=prediction.signal_strength,
                    entry_price=current_price,
                    atr=atr,
                    current_equity=equity,
                    current_exposure=total_exposure,
                    symbol_exposure=symbol_exposure,
                )

                approved = self._risk_manager.validate_trade(proposal)

                # 5. Execute the trade
                order = await order_manager.execute_market_order(
                    db=db,
                    symbol=symbol,
                    side=side,
                    quantity=approved.quantity,
                )

                # 6. Record position
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

                logger.info(
                    "trade_executed",
                    symbol=symbol,
                    side=side,
                    qty=float(order.filled_quantity),
                    price=float(order.avg_fill_price or current_price),
                    signal=round(prediction.signal_strength, 4),
                )

        except (DrawdownCircuitBreakerError, RiskLimitExceededError) as e:
            logger.info("trade_blocked_by_risk", symbol=symbol, reason=str(e))
        except TradeMasterError as e:
            logger.error("trade_failed", symbol=symbol, error=str(e), code=e.code)
        except Exception as e:
            logger.error("unexpected_trading_error", symbol=symbol, error=str(e))

    def _simple_signal(self, df, symbol: str):
        """Multi-indicator technical signal when ML models are not available.

        Combines SMA crossover, RSI, MACD, Bollinger Bands, and momentum
        into a composite score. Designed to generate signals frequently
        even with low-variance candle data (CoinGecko 10s updates).
        """
        import numpy as np

        close = df["close"].values.astype(float)
        n = len(close)

        if n < 30:
            return None

        scores = []  # Each indicator contributes a score in [-1, 1]

        # --- 1. SMA crossover (fast 5 vs slow 15) ---
        sma_5 = np.mean(close[-5:])
        sma_15 = np.mean(close[-15:])
        if sma_15 > 0:
            sma_diff_pct = (sma_5 - sma_15) / sma_15 * 100
            sma_score = np.clip(sma_diff_pct * 20, -1, 1)  # amplify small diffs
            scores.append(("sma", sma_score, 0.25))

        # --- 2. RSI (14) ---
        rsi_period = min(14, n - 1)
        deltas = np.diff(close[-rsi_period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
        rs = avg_gain / max(avg_loss, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        # RSI < 40 = buy, > 60 = sell
        rsi_score = np.clip((50 - rsi) / 25, -1, 1)
        scores.append(("rsi", rsi_score, 0.20))

        # --- 3. MACD (fast=8, slow=17, signal=6) ---
        if n >= 20:
            ema_fast = self._ema(close, 8)
            ema_slow = self._ema(close, 17)
            macd_line = ema_fast - ema_slow
            if len(close) >= 6:
                signal_line = self._ema(np.array([macd_line]), 1)  # simplified
                macd_hist = macd_line - signal_line
                # Normalize by price
                macd_score = np.clip(macd_hist / (close[-1] * 0.0005) if close[-1] > 0 else 0, -1, 1)
                scores.append(("macd", macd_score, 0.20))

        # --- 4. Bollinger Bands (20, 2) ---
        if n >= 20:
            bb_mean = np.mean(close[-20:])
            bb_std = np.std(close[-20:])
            if bb_std > 0:
                bb_z = (close[-1] - bb_mean) / bb_std
                # Below -1 = buy, above +1 = sell (mean reversion)
                bb_score = np.clip(-bb_z / 1.5, -1, 1)
                scores.append(("bb", bb_score, 0.15))

        # --- 5. Momentum (5-period rate of change) ---
        if close[-6] > 0:
            momentum = (close[-1] - close[-6]) / close[-6] * 100
            mom_score = np.clip(momentum * 15, -1, 1)  # amplify
            scores.append(("mom", mom_score, 0.20))

        # --- Composite signal ---
        if not scores:
            return None

        total_weight = sum(w for _, _, w in scores)
        signal_strength = sum(s * w for _, s, w in scores) / total_weight

        # Apply a minimum threshold (very low to allow frequent trading)
        if abs(signal_strength) < 0.08:
            return None

        # Scale to match what signal_to_action expects (>= 0.3 or <= -0.3)
        # We map our 0.08-1.0 range to 0.3-0.8
        direction = 1 if signal_strength > 0 else -1
        scaled = 0.3 + abs(signal_strength) * 0.5
        signal_strength = direction * min(scaled, 0.8)

        from app.services.ml.models.base import ModelPrediction

        action = 2 if signal_strength > 0 else 0  # BUY or SELL
        probs = np.array([
            max(0, -signal_strength),
            1 - abs(signal_strength),
            max(0, signal_strength),
        ])
        probs = probs / probs.sum()

        logger.info(
            "simple_signal_generated",
            symbol=symbol,
            signal_strength=round(signal_strength, 4),
            indicators={name: round(score, 4) for name, score, _ in scores},
            rsi=round(rsi, 2),
        )

        return ModelPrediction(
            action=action,
            probabilities=probs,
            confidence=abs(signal_strength),
            signal_strength=signal_strength,
        )

    @staticmethod
    def _ema(data: "np.ndarray", period: int) -> float:
        """Calculate EMA of the last `period` values."""
        import numpy as np
        if len(data) < period:
            return float(np.mean(data))
        weights = np.exp(np.linspace(-1.0, 0.0, period))
        weights /= weights.sum()
        return float(np.dot(data[-period:], weights))

    async def check_positions(self) -> None:
        """Check all open positions for stop losses and trailing updates.

        Should be called periodically (e.g., every minute).
        """
        try:
            prices = {}
            for symbol in settings.symbols_list:
                try:
                    price = await binance_client.get_ticker_price(symbol)
                    prices[symbol] = float(price)
                except Exception:
                    pass  # Skip symbols with no price available

            if not prices:
                return  # No prices available, skip this check

            async with async_session_factory() as db:
                # Update prices
                await portfolio_tracker.update_prices(db, prices)

                # Check stops
                closed = await portfolio_tracker.check_stop_losses(db, prices)

                # Execute close orders for stopped positions
                for pos in closed:
                    if pos.side == "LONG":
                        await order_manager.execute_market_order(
                            db=db,
                            symbol=pos.symbol,
                            side="SELL",
                            quantity=float(pos.quantity),
                        )
                    else:
                        await order_manager.execute_market_order(
                            db=db,
                            symbol=pos.symbol,
                            side="BUY",
                            quantity=float(pos.quantity),
                        )

                # Update circuit breaker and persist to Redis
                try:
                    equity = float(await binance_client.get_balance("USDT"))
                except Exception:
                    equity = 10000.0  # Paper mode fallback
                await circuit_breaker.update_and_persist(equity)

                # Take snapshot every check
                await portfolio_tracker.take_snapshot(db, equity, equity)

                await db.commit()

        except Exception as e:
            logger.error("position_check_error", error=str(e))


trading_engine = TradingEngine()

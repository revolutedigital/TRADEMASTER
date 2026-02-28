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
        self._min_signal_interval_seconds: float = 10  # 10s between signals per symbol (ML inference takes ~1-3s)

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

            if df.empty or len(df) < 200:
                return

            # 2. ML prediction
            prediction = await ml_pipeline.predict(df, symbol)
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
                equity = float(await binance_client.get_balance("USDT"))
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

    async def check_positions(self) -> None:
        """Check all open positions for stop losses and trailing updates.

        Should be called periodically (e.g., every minute).
        """
        try:
            prices = {}
            for symbol in settings.symbols_list:
                price = await binance_client.get_ticker_price(symbol)
                prices[symbol] = float(price)

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
                equity = float(await binance_client.get_balance("USDT"))
                await circuit_breaker.update_and_persist(equity)

                # Take snapshot every check
                await portfolio_tracker.take_snapshot(db, equity, equity)

                await db.commit()

        except Exception as e:
            logger.error("position_check_error", error=str(e))


trading_engine = TradingEngine()

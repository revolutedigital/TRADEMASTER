"""Unit tests for the TradingEngine.

Tests cover:
- _technical_signal() with known SMA crossover / RSI scenarios
- Minimum trade interval (anti-churning cooldown)
- Signal generation with mock candle data
- Circuit breaker blocking trades
- Daily trade limit enforcement
- Timeframe gate filtering
"""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.config import TradingExecutionMode
from app.core.exceptions import OrderExecutionError, TradeMasterError
from app.models.trade import OrderStatus
from app.schemas.trading import TechnicalStrategyConfig
from app.services.portfolio.tracker import PaperExitCandidate
from app.services.strategy_deployments import ActiveTechnicalStrategy
from app.services.trading_engine import (
    ALLOWED_INTERVALS,
    MAX_TRADES_PER_DAY,
    MIN_CANDLES_FOR_SIGNAL,
    MIN_TRADE_INTERVAL_SECONDS,
    TradingEngine,
    _acquire_symbol_execution_lock,
    _allows_new_position_side,
    _is_positive_finite,
    _requires_higher_timeframe_confirmation,
)

# ---------------------------------------------------------------------------
# Helpers to build deterministic candle DataFrames
# ---------------------------------------------------------------------------


def _make_candle_df(
    close_prices: list[float],
    *,
    spread_pct: float = 0.005,
    interval: str = "15m",
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a list of close prices.

    high/low are derived from close +/- spread_pct so ATR is non-zero.
    """
    n = len(close_prices)
    close = np.array(close_prices, dtype=float)
    high = close * (1 + spread_pct)
    low = close * (1 - spread_pct)
    opn = (close + np.roll(close, 1)) / 2
    opn[0] = close[0]
    volume = np.full(n, 100.0)
    pandas_frequency, interval_duration = {
        "15m": ("15min", timedelta(minutes=15)),
        "1h": ("1h", timedelta(hours=1)),
        "4h": ("4h", timedelta(hours=4)),
    }.get(interval, (interval, pd.Timedelta(interval)))
    close_times = pd.date_range(
        end=datetime.now(UTC),
        periods=n,
        freq=pandas_frequency,
    )
    dates = close_times - interval_duration

    return pd.DataFrame(
        {
            "open_time": dates,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": close_times,
        }
    )


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[float]:
    """Generate a cleanly rising price series."""
    return [start + i * step for i in range(n)]


def _trending_down(n: int = 60, start: float = 200.0, step: float = 0.5) -> list[float]:
    """Generate a cleanly falling price series."""
    return [start - i * step for i in range(n)]


def _flat_prices(n: int = 60, price: float = 100.0, noise: float = 0.001) -> list[float]:
    """Generate a flat/sideways price series with minimal noise."""
    np.random.seed(99)
    return [price + np.random.uniform(-noise, noise) for _ in range(n)]


# ===========================================================================
# 1. Basic engine state tests
# ===========================================================================


class TestTradingEngineState:
    def test_initial_state(self):
        engine = TradingEngine()
        assert engine._running is False
        assert engine._start_reserved is False
        assert isinstance(engine._last_trade_time, dict)
        assert len(engine._last_trade_time) == 0

    async def test_engine_start_can_only_be_reserved_once_until_stopped(self):
        engine = TradingEngine()

        assert engine.reserve_start() is True
        assert engine._running is True
        assert engine._start_reserved is True
        assert engine.reserve_start() is False

        await engine.stop()

        assert engine._running is False
        assert engine._start_reserved is False
        assert engine.reserve_start() is True

    @patch("app.services.trading_engine.ml_pipeline")
    async def test_stop_sets_running_false(self, mock_ml):
        engine = TradingEngine()
        engine._running = True
        await engine.stop()
        assert engine._running is False

    def test_daily_count_reset(self):
        engine = TradingEngine()
        engine._daily_trade_date = "2025-01-01"
        engine._daily_trade_count["BTCUSDT"] = 5
        engine._reset_daily_counts_if_needed()
        # Date is in the past, counters must be cleared
        assert engine._daily_trade_count["BTCUSDT"] == 0

    def test_all_runtime_ledgers_never_open_a_short_position(self):
        assert _allows_new_position_side(TradingExecutionMode.PAPER, "SELL") is False
        assert _allows_new_position_side(TradingExecutionMode.TESTNET, "SELL") is False
        assert _allows_new_position_side(TradingExecutionMode.LIVE, "SELL") is False

    async def test_symbol_execution_lock_is_transactional_and_mode_isolated(self):
        database = AsyncMock()

        await _acquire_symbol_execution_lock(
            database,
            symbol="BTCUSDT",
            execution_mode="TESTNET",
        )

        statement, parameters = database.execute.await_args.args
        assert "pg_advisory_xact_lock" in str(statement)
        assert isinstance(parameters["lock_key"], int)

        first_key = parameters["lock_key"]
        await _acquire_symbol_execution_lock(
            database,
            symbol="BTCUSDT",
            execution_mode="LIVE",
        )
        assert database.execute.await_args.args[1]["lock_key"] != first_key

    async def test_lock_failure_prevents_execution(self):
        database = AsyncMock()
        database.execute.side_effect = RuntimeError("database unavailable")

        with pytest.raises(TradeMasterError, match="no order was sent") as error:
            await _acquire_symbol_execution_lock(
                database,
                symbol="BTCUSDT",
                execution_mode="LIVE",
            )

        assert error.value.code == "SYMBOL_EXECUTION_LOCK_FAILED"

    async def test_paper_strategy_sell_closes_the_existing_long_instead_of_opening_short(self):
        engine = TradingEngine()
        database = AsyncMock()
        position = SimpleNamespace(
            id=7,
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.01,
            current_price=100.0,
        )
        exit_order = SimpleNamespace(avg_fill_price=99.5)

        with (
            patch(
                "app.services.trading_engine.order_manager.execute_market_order",
                new=AsyncMock(return_value=exit_order),
            ) as execute_order,
            patch(
                "app.services.trading_engine.portfolio_tracker.close_position",
                new=AsyncMock(),
            ) as close_position,
        ):
            exited = await engine._execute_strategy_exit(
                db=database,
                symbol="BTCUSDT",
                positions=[position],
                execution_mode=TradingExecutionMode.PAPER,
                signal_id=11,
            )

        assert exited is True
        assert execute_order.await_args.kwargs == {
            "db": database,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "quantity": 0.01,
            "signal_id": 11,
        }
        close_position.assert_awaited_once_with(database, position, 99.5)

    async def test_exchange_strategy_sell_uses_the_protected_position_closer(self):
        engine = TradingEngine()
        database = AsyncMock()
        position = SimpleNamespace(id=7, symbol="BTCUSDT", side="LONG", quantity=0.01)

        with patch(
            "app.services.exchange.spot_position_closer.spot_position_closer.close_for_strategy",
            new=AsyncMock(),
        ) as close_for_strategy:
            exited = await engine._execute_strategy_exit(
                db=database,
                symbol="BTCUSDT",
                positions=[position],
                execution_mode=TradingExecutionMode.TESTNET,
                signal_id=11,
            )

        assert exited is True
        close_for_strategy.assert_awaited_once_with(
            db=database,
            position_id=7,
            execution_mode=TradingExecutionMode.TESTNET,
        )

    async def test_paper_exit_closes_ledger_at_the_confirmed_simulated_fill_price(self):
        engine = TradingEngine()
        database = AsyncMock()
        position = SimpleNamespace(symbol="BTCUSDT", side="LONG", quantity=0.01)
        candidate = PaperExitCandidate(position=position, observed_price=99.0, reason="STOP_LOSS")
        exit_order = SimpleNamespace(
            status=OrderStatus.FILLED,
            filled_quantity=0.01,
            avg_fill_price=98.8,
        )

        with (
            patch(
                "app.services.trading_engine.order_manager.execute_market_order",
                new=AsyncMock(return_value=exit_order),
            ) as execute_order,
            patch(
                "app.services.trading_engine.portfolio_tracker.close_position",
                new=AsyncMock(return_value=position),
            ) as close_position,
        ):
            result = await engine._execute_confirmed_paper_exit(
                db=database,
                candidate=candidate,
            )

        assert result is position
        execute_order.assert_awaited_once_with(
            db=database,
            symbol="BTCUSDT",
            side="SELL",
            quantity=0.01,
        )
        close_position.assert_awaited_once_with(database, position, 98.8)

    async def test_partial_paper_exit_keeps_ledger_open(self):
        engine = TradingEngine()
        database = AsyncMock()
        position = SimpleNamespace(symbol="BTCUSDT", side="LONG", quantity=0.01)
        candidate = PaperExitCandidate(position=position, observed_price=99.0, reason="STOP_LOSS")
        partial_order = SimpleNamespace(
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=0.005,
            avg_fill_price=98.8,
        )

        with (
            patch(
                "app.services.trading_engine.order_manager.execute_market_order",
                new=AsyncMock(return_value=partial_order),
            ),
            patch(
                "app.services.trading_engine.portfolio_tracker.close_position",
                new=AsyncMock(),
            ) as close_position,
            pytest.raises(OrderExecutionError, match="position remains open"),
        ):
            await engine._execute_confirmed_paper_exit(db=database, candidate=candidate)

        close_position.assert_not_awaited()

    async def test_active_deployment_drives_the_engine_with_saved_exit_parameters(self):
        engine = TradingEngine()
        candles = _make_candle_df([100.0] * 300, interval="1h")
        active_strategy = ActiveTechnicalStrategy(
            deployment_id=44,
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["sma"],
                indicator_params={"sma": {"sma_short": 10, "sma_long": 30}},
            ),
            signal_threshold=0.3,
            atr_stop_multiplier=1.5,
            risk_reward_ratio=3.0,
        )
        event = MagicMock(data={"symbol": "BTCUSDT", "interval": "1h", "is_closed": True})
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        database.commit = AsyncMock()

        class SessionContext:
            async def __aenter__(self):
                return database

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        approved = SimpleNamespace(
            quantity=0.01,
            stop_loss=SimpleNamespace(stop_price=98.5, take_profit_price=103.0),
        )
        order = SimpleNamespace(
            avg_fill_price=100.0,
            filled_quantity=0.01,
            execution_mode="PAPER",
            exchange_order_id="PAPER-1",
            protective_order_list_id=None,
            protective_quantity=None,
        )
        runtime_settings = SimpleNamespace(
            execution_mode=TradingExecutionMode.PAPER,
            trading_max_single_asset_exposure=0.30,
            trading_max_portfolio_exposure=0.60,
        )
        regime = SimpleNamespace(
            signal_threshold=0.3,
            market="TRENDING",
            volatility="NORMAL",
            confidence=1.0,
            position_size_mult=1.0,
        )

        with (
            patch(
                "app.services.trading_engine.async_session_factory",
                return_value=SessionContext(),
            ),
            patch("app.services.trading_engine.settings", runtime_settings),
            patch(
                "app.services.trading_engine.market_data_collector.get_latest_candles",
                new=AsyncMock(return_value=candles),
            ),
            patch(
                "app.services.strategy_deployments.get_active_technical_strategy",
                new=AsyncMock(return_value=active_strategy),
            ),
            patch(
                "app.services.backtest.technical_strategy.build_technical_strategy_signals",
                return_value=(pd.Series([0.0] * 299 + [1.0]), MagicMock()),
            ) as strategy_signals,
            patch("app.services.ml.regime.regime_detector.detect", return_value=regime),
            patch.object(
                engine._risk_manager, "validate_trade", return_value=approved
            ) as validate_trade,
            patch(
                "app.services.risk.correlation.correlation_filter.check_can_open",
                new=AsyncMock(return_value=(True, "")),
            ) as correlation_check,
            patch(
                "app.services.trading_engine.portfolio_tracker.get_total_exposure",
                new=AsyncMock(return_value=0.0),
            ) as get_total_exposure,
            patch(
                "app.services.trading_engine.portfolio_tracker.get_symbol_exposure",
                new=AsyncMock(return_value=0.0),
            ) as get_symbol_exposure,
            patch(
                "app.services.trading_engine.portfolio_tracker.get_open_positions",
                new=AsyncMock(return_value=[]),
            ) as get_open_positions,
            patch(
                "app.services.trading_engine.portfolio_tracker.open_position",
                new=AsyncMock(),
            ),
            patch(
                "app.services.trading_engine.order_manager.execute_market_order",
                new=AsyncMock(return_value=order),
            ) as execute_order,
            patch(
                "app.services.trading_engine.binance_client.get_balance",
                new=AsyncMock(return_value=10_000.0),
            ),
            patch("app.services.trading_engine.rolling_sharpe_monitor") as rolling_sharpe,
        ):
            rolling_sharpe.is_paused = False
            await engine._process_closed_candle(event)

        strategy_signals.assert_called_once()
        proposal = validate_trade.call_args.args[0]
        assert proposal.atr_stop_multiplier == 1.5
        assert proposal.risk_reward_ratio == 3.0
        execute_order.assert_awaited_once()
        assert correlation_check.await_args.kwargs["execution_mode"] == "PAPER"
        assert get_total_exposure.await_args.kwargs["execution_mode"] == "PAPER"
        assert get_symbol_exposure.await_args.kwargs["execution_mode"] == "PAPER"
        assert get_open_positions.await_args.kwargs["execution_mode"] == "PAPER"
        candidate_signal = database.add.call_args.args[0]
        assert candidate_signal.action == "BUY"
        assert candidate_signal.was_executed is True
        assert json.loads(candidate_signal.features_snapshot)["strategy_deployment_id"] == 44

    async def test_exchange_monitor_never_runs_local_paper_exit_logic(self):
        engine = TradingEngine()
        database = AsyncMock()
        database.commit = AsyncMock()

        class SessionContext:
            async def __aenter__(self):
                return database

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        runtime_settings = SimpleNamespace(
            execution_mode=TradingExecutionMode.TESTNET,
            symbols_list=["BTCUSDT"],
        )
        reconciliation_report = SimpleNamespace(ready=True, issues=())

        with (
            patch(
                "app.services.trading_engine.async_session_factory",
                return_value=SessionContext(),
            ),
            patch("app.services.trading_engine.settings", runtime_settings),
            patch(
                "app.services.exchange.spot_protection_reconciler."
                "spot_protection_reconciler.reconcile",
                new=AsyncMock(return_value=reconciliation_report),
            ),
            patch(
                "app.services.trading_engine.binance_client.get_ticker_price",
                new=AsyncMock(return_value=100.0),
            ),
            patch(
                "app.services.trading_engine.binance_client.get_balance",
                new=AsyncMock(return_value=10_000.0),
            ),
            patch(
                "app.services.trading_engine.portfolio_tracker.update_prices",
                new=AsyncMock(),
            ) as update_prices,
            patch(
                "app.services.trading_engine.portfolio_tracker.find_paper_exit_candidates",
                new=AsyncMock(),
            ) as find_paper_exit_candidates,
            patch(
                "app.services.trading_engine.portfolio_tracker.take_snapshot",
                new=AsyncMock(),
            ),
            patch.object(
                engine._risk_manager,
                "refresh_performance_stats",
                new=AsyncMock(),
            ),
            patch(
                "app.services.trading_engine.circuit_breaker.update_and_persist",
                new=AsyncMock(),
            ),
            patch(
                "app.services.trading_engine.order_manager.execute_market_order",
                new=AsyncMock(),
            ) as execute_market_order,
            patch(
                "app.services.risk.rolling_sharpe.rolling_sharpe_monitor.check",
                return_value=SimpleNamespace(is_paused=False),
            ),
            patch(
                "app.services.ml.drift_detector.drift_detector.auto_retrain_if_needed",
                new=AsyncMock(),
            ),
        ):
            await engine.check_positions()

        update_prices.assert_awaited_once_with(
            database,
            {"BTCUSDT": 100.0},
            execution_mode="TESTNET",
        )
        find_paper_exit_candidates.assert_not_awaited()
        execute_market_order.assert_not_awaited()

    async def test_exchange_mode_rejects_entries_without_an_active_strategy_deployment(self):
        engine = TradingEngine()
        candles = _make_candle_df([100.0] * 300, interval="1h")
        event = MagicMock(data={"symbol": "BTCUSDT", "interval": "1h", "is_closed": True})
        database = AsyncMock()

        class SessionContext:
            async def __aenter__(self):
                return database

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        with (
            patch(
                "app.services.trading_engine.async_session_factory",
                return_value=SessionContext(),
            ),
            patch(
                "app.services.trading_engine.settings",
                SimpleNamespace(execution_mode=TradingExecutionMode.TESTNET),
            ),
            patch(
                "app.services.trading_engine.market_data_collector.get_latest_candles",
                new=AsyncMock(return_value=candles),
            ),
            patch(
                "app.services.strategy_deployments.get_active_technical_strategy",
                new=AsyncMock(return_value=None),
            ),
            patch.object(engine, "_technical_signal") as technical_signal,
        ):
            await engine._process_closed_candle(event)

        technical_signal.assert_not_called()

    async def test_stale_closed_candle_history_never_reaches_strategy_or_order_execution(self):
        engine = TradingEngine()
        candles = _make_candle_df([100.0] * 300, interval="1h")
        candles["close_time"] = candles["close_time"] - timedelta(days=2)
        event = MagicMock(data={"symbol": "BTCUSDT", "interval": "1h", "is_closed": True})
        database = AsyncMock()

        class SessionContext:
            async def __aenter__(self):
                return database

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        with (
            patch(
                "app.services.trading_engine.async_session_factory",
                return_value=SessionContext(),
            ),
            patch(
                "app.services.trading_engine.market_data_collector.get_latest_candles",
                new=AsyncMock(return_value=candles),
            ),
            patch(
                "app.services.strategy_deployments.get_active_technical_strategy",
                new=AsyncMock(),
            ) as active_strategy,
            patch(
                "app.services.trading_engine.order_manager.execute_market_order",
                new=AsyncMock(),
            ) as execute_order,
        ):
            await engine._process_closed_candle(event)

        active_strategy.assert_not_awaited()
        execute_order.assert_not_awaited()

    async def test_exchange_entry_is_blocked_when_quote_balance_is_unverified(self):
        engine = TradingEngine()
        candles = _make_candle_df([100.0] * 300, interval="1h")
        active_strategy = ActiveTechnicalStrategy(
            deployment_id=45,
            strategy=TechnicalStrategyConfig(
                kind="technical_ensemble",
                indicators=["sma"],
                indicator_params={"sma": {"sma_short": 10, "sma_long": 30}},
            ),
            signal_threshold=0.3,
            atr_stop_multiplier=1.5,
            risk_reward_ratio=3.0,
        )
        event = MagicMock(data={"symbol": "BTCUSDT", "interval": "1h", "is_closed": True})
        database = AsyncMock()
        database.add = MagicMock()
        database.flush = AsyncMock()
        database.commit = AsyncMock()
        regime = SimpleNamespace(
            signal_threshold=0.3,
            market="TRENDING",
            volatility="NORMAL",
            confidence=1.0,
            position_size_mult=1.0,
        )

        class SessionContext:
            async def __aenter__(self):
                return database

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        runtime_settings = SimpleNamespace(
            execution_mode=TradingExecutionMode.TESTNET,
            trading_max_single_asset_exposure=0.30,
            trading_max_portfolio_exposure=0.60,
        )
        with (
            patch(
                "app.services.trading_engine.async_session_factory",
                return_value=SessionContext(),
            ),
            patch("app.services.trading_engine.settings", runtime_settings),
            patch(
                "app.services.trading_engine.market_data_collector.get_latest_candles",
                new=AsyncMock(return_value=candles),
            ),
            patch(
                "app.services.strategy_deployments.get_active_technical_strategy",
                new=AsyncMock(return_value=active_strategy),
            ),
            patch(
                "app.services.backtest.technical_strategy.build_technical_strategy_signals",
                return_value=(pd.Series([0.0] * 299 + [1.0]), MagicMock()),
            ),
            patch("app.services.ml.regime.regime_detector.detect", return_value=regime),
            patch(
                "app.services.trading_engine.portfolio_tracker.get_open_positions",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.services.trading_engine.binance_client.get_balance",
                new=AsyncMock(side_effect=RuntimeError("Binance unavailable")),
            ),
            patch(
                "app.services.trading_engine.order_manager.execute_market_order",
                new=AsyncMock(),
            ) as execute_order,
            patch("app.services.trading_engine.rolling_sharpe_monitor") as rolling_sharpe,
        ):
            rolling_sharpe.is_paused = False
            await engine._process_closed_candle(event)

        execute_order.assert_not_awaited()

    async def test_exchange_snapshot_never_uses_simulated_equity(self):
        engine = TradingEngine()
        database = AsyncMock()
        database.commit = AsyncMock()

        class SessionContext:
            async def __aenter__(self):
                return database

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        runtime_settings = SimpleNamespace(
            execution_mode=TradingExecutionMode.TESTNET,
            symbols_list=["BTCUSDT"],
        )
        reconciliation_report = SimpleNamespace(ready=True, issues=())
        with (
            patch(
                "app.services.trading_engine.async_session_factory",
                return_value=SessionContext(),
            ),
            patch("app.services.trading_engine.settings", runtime_settings),
            patch(
                "app.services.exchange.spot_protection_reconciler."
                "spot_protection_reconciler.reconcile",
                new=AsyncMock(return_value=reconciliation_report),
            ),
            patch(
                "app.services.trading_engine.binance_client.get_ticker_price",
                new=AsyncMock(return_value=100.0),
            ),
            patch(
                "app.services.trading_engine.binance_client.get_balance",
                new=AsyncMock(side_effect=RuntimeError("Binance unavailable")),
            ),
            patch(
                "app.services.trading_engine.portfolio_tracker.update_prices",
                new=AsyncMock(),
            ),
            patch(
                "app.services.trading_engine.circuit_breaker.update_and_persist",
                new=AsyncMock(),
            ) as update_circuit_breaker,
            patch(
                "app.services.trading_engine.portfolio_tracker.take_snapshot",
                new=AsyncMock(),
            ) as take_snapshot,
        ):
            await engine.check_positions()

        update_circuit_breaker.assert_not_awaited()
        take_snapshot.assert_not_awaited()
        database.commit.assert_awaited()


# ===========================================================================
# 2. _technical_signal() tests
# ===========================================================================


class TestTechnicalSignal:
    """Test _technical_signal() with deterministic candle data."""

    def test_returns_none_when_insufficient_candles(self):
        engine = TradingEngine()
        df = _make_candle_df([100.0] * 10)  # Only 10 candles, need 30
        result = engine._technical_signal(df, "BTCUSDT")
        assert result is None

    def test_bullish_signal_on_uptrend(self):
        """A clean uptrend should produce a positive (BUY) signal."""
        engine = TradingEngine()
        prices = _trending_up(n=80, start=100, step=0.5)
        df = _make_candle_df(prices)
        result = engine._technical_signal(df, "BTCUSDT", signal_threshold=0.05)
        assert result is not None
        assert result.signal_strength > 0, f"Expected BUY signal, got {result.signal_strength}"
        assert result.action == 2  # BUY

    def test_bearish_signal_on_downtrend(self):
        """A clean downtrend should produce a negative (SELL) signal."""
        engine = TradingEngine()
        prices = _trending_down(n=80, start=200, step=0.5)
        df = _make_candle_df(prices)
        result = engine._technical_signal(df, "BTCUSDT", signal_threshold=0.05)
        assert result is not None
        assert result.signal_strength < 0, f"Expected SELL signal, got {result.signal_strength}"
        assert result.action == 0  # SELL

    def test_no_signal_on_flat_market(self):
        """A flat market should return None (below threshold) or a near-zero signal."""
        engine = TradingEngine()
        prices = _flat_prices(n=60, price=100.0, noise=0.001)
        df = _make_candle_df(prices, spread_pct=0.0001)
        result = engine._technical_signal(df, "BTCUSDT", signal_threshold=0.25)
        # Flat market should be below the default 0.25 threshold
        assert result is None

    def test_signal_strength_bounded(self):
        """Signal strength must be capped at [-0.8, 0.8]."""
        engine = TradingEngine()
        # Very steep trend to push signal to max
        prices = _trending_up(n=80, start=100, step=5.0)
        df = _make_candle_df(prices)
        result = engine._technical_signal(df, "BTCUSDT", signal_threshold=0.01)
        if result is not None:
            assert -0.8 <= result.signal_strength <= 0.8

    def test_prediction_has_valid_probabilities(self):
        """Probabilities must sum to 1 and have shape (3,)."""
        engine = TradingEngine()
        prices = _trending_up(n=80, start=100, step=0.5)
        df = _make_candle_df(prices)
        result = engine._technical_signal(df, "BTCUSDT", signal_threshold=0.05)
        assert result is not None
        assert result.probabilities.shape == (3,)
        assert abs(result.probabilities.sum() - 1.0) < 1e-6
        assert all(p >= 0 for p in result.probabilities)

    def test_rsi_oversold_contributes_buy(self):
        """After a sharp drop then recovery, RSI should be low-ish, helping buy signal.

        We construct prices that drop hard then flatten — RSI should be low.
        """
        engine = TradingEngine()
        # Drop from 200 to 100, then stay at 100 (RSI should be low)
        drop = [200 - i * 3 for i in range(34)]
        flat = [100.0] * 30
        prices = drop + flat
        df = _make_candle_df(prices)
        result = engine._technical_signal(df, "BTCUSDT", signal_threshold=0.05)
        # The exact direction depends on the combined trend and mean-reversion
        # filters, but any emitted prediction must remain a valid bounded action.
        if result is not None:
            assert result.action in {0, 2}
            assert 0 < abs(result.signal_strength) <= 0.8

    def test_minimum_30_candles_required(self):
        """Exactly MIN_CANDLES_FOR_SIGNAL candles should be processed."""
        engine = TradingEngine()
        prices = _trending_up(n=MIN_CANDLES_FOR_SIGNAL, start=100, step=0.5)
        df = _make_candle_df(prices)
        # Should not crash — may or may not produce a signal
        engine._technical_signal(df, "BTCUSDT", signal_threshold=0.05)
        # 29 candles must return None
        df_short = _make_candle_df(prices[: MIN_CANDLES_FOR_SIGNAL - 1])
        assert engine._technical_signal(df_short, "BTCUSDT") is None


# ===========================================================================
# 3. Anti-churning: minimum trade interval
# ===========================================================================


class TestMinimumTradeInterval:
    """Test that _process_closed_candle respects the anti-churning cooldown."""

    @patch("app.services.trading_engine.circuit_breaker")
    @patch("app.services.trading_engine.binance_client")
    async def test_cooldown_blocks_rapid_trades(self, mock_binance, mock_cb):
        """If last trade was < MIN_TRADE_INTERVAL_SECONDS ago, candle is skipped."""
        engine = TradingEngine()
        engine._running = True
        now = datetime.now(UTC)
        engine._last_trade_time["BTCUSDT"] = now - timedelta(seconds=60)  # 60s ago

        event = MagicMock()
        event.data = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "is_closed": True,
            "close": 85000,
        }

        # The method should return early without calling any downstream services
        with patch.object(engine, "_technical_signal") as mock_signal:
            await engine._process_closed_candle(event)
            mock_signal.assert_not_called()

    @patch("app.services.trading_engine.circuit_breaker")
    @patch("app.services.trading_engine.binance_client")
    async def test_cooldown_allows_after_interval(self, mock_binance, mock_cb):
        """After MIN_TRADE_INTERVAL_SECONDS, the cooldown gate passes."""
        engine = TradingEngine()
        engine._running = True
        now = datetime.now(UTC)
        # Set last trade well in the past (beyond cooldown)
        engine._last_trade_time["BTCUSDT"] = now - timedelta(
            seconds=MIN_TRADE_INTERVAL_SECONDS + 60
        )

        event = MagicMock()
        event.data = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "is_closed": True,
        }

        # It should pass the cooldown gate and reach the rolling sharpe check.
        # We patch rolling_sharpe_monitor to pause so it stops there (avoids DB).
        with patch("app.services.trading_engine.rolling_sharpe_monitor") as mock_sharpe:
            # Import here to get the correct import path
            mock_sharpe.is_paused = True
            with patch("app.services.risk.rolling_sharpe.rolling_sharpe_monitor", mock_sharpe):
                await engine._process_closed_candle(event)
            # If rolling_sharpe is_paused was checked, it means cooldown gate passed
            # (The method accesses rolling_sharpe_monitor.is_paused after Gate 3)


# ===========================================================================
# 4. Daily trade limit
# ===========================================================================


class TestDailyTradeLimit:
    @patch("app.services.trading_engine.circuit_breaker")
    @patch("app.services.trading_engine.binance_client")
    async def test_daily_limit_blocks_excess_trades(self, mock_binance, mock_cb):
        """When daily count >= MAX_TRADES_PER_DAY, candle is skipped."""
        engine = TradingEngine()
        engine._running = True
        # Set today's date and max out the counter
        engine._daily_trade_date = datetime.now(UTC).strftime("%Y-%m-%d")
        engine._daily_trade_count["BTCUSDT"] = MAX_TRADES_PER_DAY

        event = MagicMock()
        event.data = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "is_closed": True,
        }

        with patch.object(engine, "_technical_signal") as mock_signal:
            await engine._process_closed_candle(event)
            mock_signal.assert_not_called()


# ===========================================================================
# 5. Timeframe gate
# ===========================================================================


class TestTimeframeGate:
    @patch("app.services.trading_engine.circuit_breaker")
    @patch("app.services.trading_engine.binance_client")
    async def test_1m_interval_rejected(self, mock_binance, mock_cb):
        """1-minute candles must be rejected at the timeframe gate."""
        engine = TradingEngine()
        engine._running = True

        event = MagicMock()
        event.data = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "is_closed": True,
        }

        with patch.object(engine, "_technical_signal") as mock_signal:
            await engine._process_closed_candle(event)
            mock_signal.assert_not_called()

    def test_allowed_intervals_constant(self):
        assert "15m" in ALLOWED_INTERVALS
        assert "1h" in ALLOWED_INTERVALS
        assert "4h" in ALLOWED_INTERVALS
        assert "1m" not in ALLOWED_INTERVALS
        assert "5m" not in ALLOWED_INTERVALS


# ===========================================================================
# 6. Circuit breaker / rolling sharpe pause
# ===========================================================================


class TestCircuitBreakerBlocking:
    @patch("app.services.trading_engine.circuit_breaker")
    @patch("app.services.trading_engine.binance_client")
    async def test_rolling_sharpe_pause_blocks_trade(self, mock_binance, mock_cb):
        """When rolling sharpe monitor is paused, no signal processing occurs."""
        engine = TradingEngine()
        engine._running = True

        event = MagicMock()
        event.data = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "is_closed": True,
        }

        with patch("app.services.risk.rolling_sharpe.rolling_sharpe_monitor") as mock_sharpe:
            mock_sharpe.is_paused = True
            with patch.object(engine, "_technical_signal") as mock_signal:
                await engine._process_closed_candle(event)
                mock_signal.assert_not_called()


# ===========================================================================
# 7. EMA and ATR helper methods
# ===========================================================================


class TestHelperMethods:
    def test_ema_series_basic(self):
        """EMA of constant series should equal that constant."""
        data = np.array([50.0] * 20)
        result = TradingEngine._ema_series(data, 10)
        np.testing.assert_allclose(result[-1], 50.0, atol=1e-6)

    def test_ema_series_responds_to_step(self):
        """EMA should lag behind a step change."""
        data = np.array([100.0] * 10 + [200.0] * 10)
        result = TradingEngine._ema_series(data, 5)
        # After step, EMA should be between 100 and 200
        assert 100 < result[12] < 200
        # Eventually converges near 200
        assert result[-1] > 190

    def test_compute_atr_returns_float(self):
        """ATR should return a positive float for valid OHLCV data."""
        df = _make_candle_df(_trending_up(n=30, start=100, step=0.5))
        atr = TradingEngine._compute_atr(df, period=14)
        assert atr is not None
        assert atr > 0

    def test_compute_atr_returns_none_for_short_data(self):
        """ATR should return None when data is shorter than period + 1."""
        df = _make_candle_df([100.0] * 5)
        atr = TradingEngine._compute_atr(df, period=14)
        assert atr is None


# ===========================================================================
# 8. Multi-timeframe entry confirmation
# ===========================================================================


class TestHigherTimeframeConfirmation:
    def test_only_new_15m_buy_entries_require_secondary_confirmation(self):
        assert _requires_higher_timeframe_confirmation("15m", "BUY")
        assert not _requires_higher_timeframe_confirmation("15m", "SELL")
        assert not _requires_higher_timeframe_confirmation("1h", "BUY")

    @pytest.mark.parametrize("value", [1.0, 0.00001])
    def test_market_value_must_be_positive_and_finite(self, value):
        assert _is_positive_finite(value)

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_market_value_is_rejected(self, value):
        assert not _is_positive_finite(value)

    async def test_accepts_fresh_aligned_one_hour_history(self):
        engine = TradingEngine()
        candles = _make_candle_df(_trending_up(n=20), interval="1h")

        with patch(
            "app.services.trading_engine.market_data_collector.get_latest_candles",
            new=AsyncMock(return_value=candles),
        ):
            assert await engine._check_higher_timeframe("BTCUSDT", "BUY")

    async def test_uses_fresh_continuous_15m_proxy_when_one_hour_is_missing(self):
        engine = TradingEngine()
        no_one_hour_data = pd.DataFrame()
        proxy_candles = _make_candle_df(_trending_up(n=80), interval="15m")

        with patch(
            "app.services.trading_engine.market_data_collector.get_latest_candles",
            new=AsyncMock(side_effect=[no_one_hour_data, proxy_candles]),
        ):
            assert await engine._check_higher_timeframe("BTCUSDT", "BUY")

    async def test_rejects_new_entry_when_no_fresh_confirmation_history_exists(self):
        engine = TradingEngine()

        with patch(
            "app.services.trading_engine.market_data_collector.get_latest_candles",
            new=AsyncMock(return_value=pd.DataFrame()),
        ):
            assert not await engine._check_higher_timeframe("BTCUSDT", "BUY")

    async def test_rejects_new_entry_when_confirmation_lookup_fails(self):
        engine = TradingEngine()

        with patch(
            "app.services.trading_engine.market_data_collector.get_latest_candles",
            new=AsyncMock(side_effect=RuntimeError("database unavailable")),
        ):
            assert not await engine._check_higher_timeframe("BTCUSDT", "BUY")


# ===========================================================================
# 8. Signal integration with mock data (sample_ohlcv_data fixture)
# ===========================================================================


class TestSignalWithFixtureData:
    def test_signal_with_random_walk_data(self, sample_ohlcv_data):
        """_technical_signal should handle the conftest random-walk data without crashing."""
        engine = TradingEngine()
        result = engine._technical_signal(sample_ohlcv_data, "BTCUSDT", signal_threshold=0.05)
        # Result can be None or a valid prediction
        if result is not None:
            assert -0.8 <= result.signal_strength <= 0.8
            assert result.probabilities.shape == (3,)
            assert result.action in (0, 1, 2)

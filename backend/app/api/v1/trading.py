"""Trading API endpoints: orders, engine control, manual actions, paper trading."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.dependencies import (
    get_db,
    get_order_repository,
    get_position_repository,
    get_market_repository,
    get_trading_engine,
    require_auth,
)
from app.models.trade import Order
from app.services.exchange.binance_client import binance_client
from app.models.portfolio import Position
from app.repositories.order_repo import OrderRepository
from app.repositories.position_repo import PositionRepository
from app.repositories.market_repo import MarketDataRepository
from app.schemas.trading import OrderResponse

logger = get_logger(__name__)

router = APIRouter()


class PaperOrderRequest(BaseModel):
    symbol: str = "BTCUSDT"
    side: str = "BUY"  # BUY or SELL
    quantity: float = 0.001  # Amount of the asset
    stop_loss_pct: float | None = 0.02  # 2% stop loss
    take_profit_pct: float | None = 0.04  # 4% take profit
    price: float | None = None  # Live price from frontend (Binance WS)


@router.get("/orders", response_model=list[OrderResponse])
async def get_orders(
    symbol: str | None = None,
    side: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
    repo: OrderRepository = Depends(get_order_repository),
):
    """Get recent orders with filters. Requires authentication."""
    orders = await repo.list_filtered(
        db,
        symbol=symbol,
        side=side,
        status=status,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    return orders


@router.post("/paper-order")
async def create_paper_order(
    req: PaperOrderRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Execute a simulated paper trade using live Binance prices. Requires authentication."""
    symbol = req.symbol.upper()
    side = req.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(400, "side must be BUY or SELL")

    # Use frontend live price (from Binance WebSocket) or fetch from Binance API
    if req.price and req.price > 0:
        price = req.price
    else:
        try:
            price = float(await binance_client.get_ticker_price(symbol))
        except Exception as e:
            logger.error("live_price_fetch_failed", symbol=symbol, error=str(e))
            raise HTTPException(503, f"Não foi possível obter preço em tempo real para {symbol}. Tente novamente.")
    now = datetime.now(timezone.utc)
    commission = price * req.quantity * 0.001  # 0.1% fee

    # Create order record (instantly filled)
    order = Order(
        exchange_order_id=f"PAPER-{int(now.timestamp() * 1000)}",
        symbol=symbol,
        side=side,
        order_type="MARKET",
        status="FILLED",
        quantity=req.quantity,
        price=price,
        filled_quantity=req.quantity,
        avg_fill_price=price,
        commission=commission,
        notes="Paper trade (simulated)",
    )
    db.add(order)

    # Handle position logic
    position_side = "LONG" if side == "BUY" else "SHORT"

    # Check for existing open position on opposite side (close it)
    opposite_side = "SHORT" if side == "BUY" else "LONG"
    existing = await db.execute(
        select(Position).where(
            Position.symbol == symbol,
            Position.side == opposite_side,
            Position.is_open == True,
        )
    )
    existing_pos = existing.scalar_one_or_none()

    if existing_pos:
        # Close opposite position
        if existing_pos.side == "LONG":
            pnl = (price - float(existing_pos.entry_price)) * float(existing_pos.quantity)
        else:
            pnl = (float(existing_pos.entry_price) - price) * float(existing_pos.quantity)
        pnl -= commission

        existing_pos.current_price = price
        existing_pos.realized_pnl = pnl
        existing_pos.unrealized_pnl = 0
        existing_pos.is_open = False
        existing_pos.closed_at = now
        logger.info("paper_position_closed", symbol=symbol, side=existing_pos.side, pnl=round(pnl, 2))

        await db.commit()
        await db.refresh(order)

        return {
            "status": "position_closed",
            "order_id": order.id,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": req.quantity,
            "closed_position_id": existing_pos.id,
            "realized_pnl": round(pnl, 2),
        }

    # Check for existing open position on same side (add to it)
    same_result = await db.execute(
        select(Position).where(
            Position.symbol == symbol,
            Position.side == position_side,
            Position.is_open == True,
        )
    )
    same_pos = same_result.scalar_one_or_none()

    if same_pos:
        # Average into position
        old_qty = float(same_pos.quantity)
        old_price = float(same_pos.entry_price)
        new_qty = old_qty + req.quantity
        avg_price = (old_price * old_qty + price * req.quantity) / new_qty
        same_pos.entry_price = avg_price
        same_pos.quantity = new_qty
        same_pos.current_price = price
        if position_side == "LONG":
            same_pos.unrealized_pnl = (price - avg_price) * new_qty
        else:
            same_pos.unrealized_pnl = (avg_price - price) * new_qty
        # Update stops
        if req.stop_loss_pct:
            sl = price * (1 - req.stop_loss_pct) if position_side == "LONG" else price * (1 + req.stop_loss_pct)
            same_pos.stop_loss_price = sl
        if req.take_profit_pct:
            tp = price * (1 + req.take_profit_pct) if position_side == "LONG" else price * (1 - req.take_profit_pct)
            same_pos.take_profit_price = tp

        await db.commit()
        await db.refresh(order)

        return {
            "status": "position_increased",
            "order_id": order.id,
            "symbol": symbol,
            "side": position_side,
            "price": price,
            "quantity": req.quantity,
            "position_id": same_pos.id,
            "total_quantity": new_qty,
            "avg_entry": round(avg_price, 2),
        }

    # New position
    stop_loss = None
    take_profit = None
    if req.stop_loss_pct:
        stop_loss = price * (1 - req.stop_loss_pct) if position_side == "LONG" else price * (1 + req.stop_loss_pct)
    if req.take_profit_pct:
        take_profit = price * (1 + req.take_profit_pct) if position_side == "LONG" else price * (1 - req.take_profit_pct)

    position = Position(
        symbol=symbol,
        side=position_side,
        entry_price=price,
        quantity=req.quantity,
        current_price=price,
        unrealized_pnl=0,
        realized_pnl=0,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        is_open=True,
        opened_at=now,
    )
    db.add(position)
    await db.commit()
    await db.refresh(order)
    await db.refresh(position)

    logger.info(
        "paper_position_opened",
        symbol=symbol,
        side=position_side,
        price=price,
        qty=req.quantity,
        sl=round(stop_loss, 2) if stop_loss else None,
        tp=round(take_profit, 2) if take_profit else None,
    )

    return {
        "status": "position_opened",
        "order_id": order.id,
        "position_id": position.id,
        "symbol": symbol,
        "side": position_side,
        "entry_price": price,
        "quantity": req.quantity,
        "stop_loss": round(stop_loss, 2) if stop_loss else None,
        "take_profit": round(take_profit, 2) if take_profit else None,
    }


class ClosePositionRequest(BaseModel):
    price: float | None = None  # Live price from frontend (Binance WS)


@router.post("/close-position/{position_id}")
async def close_position_manually(
    position_id: int,
    req: ClosePositionRequest | None = None,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
):
    """Manually close a position at current live price. Requires authentication."""
    result = await db.execute(
        select(Position).where(Position.id == position_id, Position.is_open == True)
    )
    position = result.scalar_one_or_none()
    if not position:
        raise HTTPException(status_code=404, detail="Open position not found")

    # Use frontend-provided live price first, then try Binance API
    exit_price = None
    if req and req.price and req.price > 0:
        exit_price = req.price
    else:
        try:
            exit_price = float(await binance_client.get_ticker_price(position.symbol))
        except Exception as e:
            logger.error("live_price_fetch_failed", symbol=position.symbol, error=str(e))
            raise HTTPException(503, f"Não foi possível obter preço em tempo real. Tente novamente.")

    # Calculate P&L
    if position.side == "LONG":
        pnl = (exit_price - float(position.entry_price)) * float(position.quantity)
    else:
        pnl = (float(position.entry_price) - exit_price) * float(position.quantity)

    commission = exit_price * float(position.quantity) * 0.001
    pnl -= commission

    position.current_price = exit_price
    position.realized_pnl = pnl
    position.unrealized_pnl = 0
    position.is_open = False
    position.closed_at = datetime.now(timezone.utc)

    # Create closing order record
    close_side = "SELL" if position.side == "LONG" else "BUY"
    order = Order(
        exchange_order_id=f"PAPER-CLOSE-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        symbol=position.symbol,
        side=close_side,
        order_type="MARKET",
        status="FILLED",
        quantity=float(position.quantity),
        price=exit_price,
        filled_quantity=float(position.quantity),
        avg_fill_price=exit_price,
        commission=commission,
        notes=f"Paper close position #{position_id}",
    )
    db.add(order)
    await db.commit()

    logger.info("paper_position_closed", position_id=position_id, exit_price=exit_price, pnl=round(pnl, 2))

    return {
        "status": "closed",
        "position_id": position_id,
        "exit_price": exit_price,
        "pnl": round(pnl, 2),
    }


@router.post("/engine/start")
async def start_engine(
    _user: dict = Depends(require_auth),
    engine=Depends(get_trading_engine),
):
    """Start the trading engine. Requires authentication."""
    if engine._running:
        return {"status": "already_running"}

    import asyncio

    # Also start price fetcher + synthetic kline generator if not running
    from app.services.market.price_fetcher import price_fetcher
    if not price_fetcher._running:
        asyncio.create_task(price_fetcher.start(), name="price_fetcher")

    from app.services.market.synthetic_kline_generator import synthetic_kline_generator
    if not synthetic_kline_generator._running:
        asyncio.create_task(synthetic_kline_generator.start(), name="synthetic_kline_generator")

    # Start stream processor if not running
    from app.services.market.stream_processor import market_stream_processor
    if not market_stream_processor._running:
        asyncio.create_task(market_stream_processor.start(), name="market_stream_processor")

    asyncio.create_task(engine.start(), name="trading_engine")
    return {"status": "started"}


@router.post("/engine/stop")
async def stop_engine(
    _user: dict = Depends(require_auth),
    engine=Depends(get_trading_engine),
):
    """Pause the trading engine. Requires authentication."""
    if not engine._running:
        return {"status": "already_stopped"}

    await engine.stop()
    return {"status": "stopped"}


@router.get("/engine/status")
async def engine_status(_user: dict = Depends(require_auth)):
    """Get trading engine status. Requires authentication."""
    from app.dependencies import get_trading_engine, get_circuit_breaker
    from app.services.scheduler import scheduler
    from app.services.exchange.binance_ws import binance_ws_manager
    from app.services.market.synthetic_kline_generator import synthetic_kline_generator
    from app.services.market.price_fetcher import price_fetcher
    from sqlalchemy import func

    engine = get_trading_engine()
    cb = get_circuit_breaker()

    # Count candles per symbol for monitoring
    candle_counts = {}
    try:
        from app.models.market import OHLCV
        from app.models.base import async_session_factory as db_session_maker
        async with db_session_maker() as sdb:
            for sym in ["BTCUSDT", "ETHUSDT"]:
                result = await sdb.execute(
                    select(func.count()).select_from(OHLCV).where(
                        OHLCV.symbol == sym,
                        OHLCV.interval == synthetic_kline_generator._interval_label,
                    )
                )
                candle_counts[sym] = result.scalar() or 0
    except Exception:
        pass

    return {
        "engine_running": engine._running,
        "price_fetcher_active": price_fetcher._running,
        "price_fetcher_source": price_fetcher._source,
        "candle_interval": synthetic_kline_generator._interval_label,
        "candle_counts": candle_counts,
        "min_candles_for_signal": 30,
        "websocket_streams": len(binance_ws_manager._tasks),
        "websocket_active": binance_ws_manager._running,
        "synthetic_kline_active": synthetic_kline_generator._running,
        "synthetic_candles_tracking": list(synthetic_kline_generator._candles.keys()),
        "daily_trade_count": dict(engine._daily_trade_count),
        "max_trades_per_day": 6,
        "circuit_breaker": cb.get_status(),
        "scheduled_tasks": scheduler.get_status(),
    }


@router.post("/engine/train-model")
async def train_bootstrap_model(
    _user: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Train a bootstrap XGBoost model using available historical data."""
    from pathlib import Path
    import numpy as np
    from app.services.market.data_collector import market_data_collector
    from app.services.ml.features import feature_engineer
    from app.services.ml.models.xgboost_model import XGBoostTradingModel
    from app.services.ml.preprocessor import Preprocessor
    from app.services.ml.pipeline import ml_pipeline

    MODELS_DIR = Path("ml_artifacts/models")
    SCALERS_DIR = Path("ml_artifacts/scalers")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    preprocessor = Preprocessor(threshold=0.005)

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        symbol_lower = symbol.lower()

        # Get historical candles
        df = await market_data_collector.get_latest_candles(
            db=db, symbol=symbol, interval="1m", limit=10000
        )

        if df.empty or len(df) < 300:
            # Try 1h interval
            df = await market_data_collector.get_latest_candles(
                db=db, symbol=symbol, interval="1h", limit=10000
            )

        if df.empty or len(df) < 300:
            results[symbol] = {"status": "skipped", "reason": f"Only {len(df)} candles available (need 300+)"}
            continue

        # Feature engineering
        df_features = feature_engineer.build_features(df)
        if df_features.empty:
            results[symbol] = {"status": "skipped", "reason": "Feature engineering failed"}
            continue

        feature_cols = feature_engineer.get_feature_columns(df_features)

        # Create targets and split
        try:
            df_features = preprocessor.create_target(df_features, horizon=5)
            split = preprocessor.prepare_tabular(df_features, feature_cols)
        except Exception as e:
            results[symbol] = {"status": "error", "reason": str(e)}
            continue

        # Train XGBoost
        model = XGBoostTradingModel()
        training_result = model.train(
            split.X_train, split.y_train,
            split.X_val, split.y_val,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            feature_names=split.feature_names,
        )

        # Save model and scaler
        model.save(MODELS_DIR / f"xgboost_{symbol_lower}.json")
        Preprocessor.save_scaler(split.scaler, SCALERS_DIR / f"scaler_{symbol_lower}.joblib")

        results[symbol] = {
            "status": "trained",
            "rows": len(df),
            "features": len(feature_cols),
            "train_accuracy": round(training_result.accuracy, 4),
            "val_accuracy": round(training_result.val_accuracy, 4),
        }

    # Reload models in the pipeline
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        await ml_pipeline.load_models(symbol)

    return {"results": results, "models_reloaded": True}


@router.get("/engine/drift-status")
async def get_drift_status(_user: dict = Depends(require_auth)):
    """Get model drift detection status for all symbols."""
    from app.services.ml.drift_detector import drift_detector
    return drift_detector.get_status()


@router.post("/engine/retrain-if-drifted")
async def trigger_drift_retrain(_user: dict = Depends(require_auth)):
    """Manually trigger drift check and retrain if needed."""
    from app.services.ml.drift_detector import drift_detector
    retrained = await drift_detector.auto_retrain_if_needed()
    return {"retrained": retrained}


@router.get("/engine/execution-analytics")
async def get_execution_analytics(
    symbol: str | None = None,
    _user: dict = Depends(require_auth),
):
    """Get trade execution quality metrics (slippage, latency, fill rate)."""
    from app.services.exchange.execution_analytics import execution_analytics
    return execution_analytics.get_best_execution_report()


@router.get("/engine/regime-status")
async def get_regime_status(_user: dict = Depends(require_auth)):
    """Get current market regime detection for all tracked symbols."""
    from app.services.ml.regime import regime_detector
    return {
        "regimes": regime_detector.get_all(),
        "description": "Adaptive regime: bull/bear/sideways × low/normal/high volatility",
    }


@router.get("/engine/rolling-sharpe")
async def get_rolling_sharpe(_user: dict = Depends(require_auth)):
    """Get rolling Sharpe ratio monitor status (auto-pause indicator)."""
    from app.services.risk.rolling_sharpe import rolling_sharpe_monitor
    return rolling_sharpe_monitor.get_status()


@router.post("/engine/rolling-sharpe/resume")
async def force_resume_sharpe(_user: dict = Depends(require_auth)):
    """Manually resume trading after rolling Sharpe auto-pause."""
    from app.services.risk.rolling_sharpe import rolling_sharpe_monitor
    rolling_sharpe_monitor.force_resume()
    return {"status": "resumed"}


@router.get("/engine/ensemble-status")
async def get_ensemble_status(_user: dict = Depends(require_auth)):
    """Get ensemble voting configuration and regime-adaptive weight info."""
    from app.services.ml.ensemble_voter import _REGIME_WEIGHTS, _VOL_ADJUSTMENTS, _REGIME_BIAS
    return {
        "regime_weights": {k: {"technical": v[0], "ml": v[1], "regime": v[2]} for k, v in _REGIME_WEIGHTS.items()},
        "volatility_adjustments": {k: {"tech": v[0], "ml": v[1], "regime": v[2]} for k, v in _VOL_ADJUSTMENTS.items()},
        "regime_bias": _REGIME_BIAS,
    }

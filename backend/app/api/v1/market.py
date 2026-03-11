"""Market data endpoints."""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger
from app.dependencies import get_db, require_auth
from app.models.market import OHLCV
from app.schemas.market import OHLCVResponse

logger = get_logger(__name__)

router = APIRouter()


class PriceFeedRequest(BaseModel):
    symbol: str
    price: float
    timestamp: float | None = None


@router.post("/price-feed")
async def receive_price_feed(
    req: PriceFeedRequest,
    _user: dict = Depends(require_auth),
):
    """Receive live price from frontend WebSocket and store in Redis.

    Used when backend can't reach Binance directly (geo-restriction).
    The frontend's browser connects to Binance WS and relays prices here.
    """
    from app.core.events import event_bus

    symbol = req.symbol.upper()
    price = req.price

    # Store latest price in Redis for other services to read
    if event_bus._redis:
        await event_bus._redis.set(f"price:{symbol}", str(price), ex=30)

    return {"status": "ok", "symbol": symbol, "price": price}


@router.get("/klines/{symbol}", response_model=list[OHLCVResponse])
async def get_klines(
    symbol: str,
    interval: str = Query(default="1h", pattern="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """Get historical kline/candlestick data for a symbol."""
    result = await db.execute(
        select(OHLCV)
        .where(OHLCV.symbol == symbol.upper(), OHLCV.interval == interval)
        .order_by(OHLCV.open_time.desc())
        .limit(limit)
    )
    candles = result.scalars().all()
    return list(reversed(candles))


@router.get("/symbols")
async def get_symbols():
    """Get list of supported trading symbols."""
    return {"symbols": settings.symbols_list}


@router.get("/tickers")
async def get_tickers(db: AsyncSession = Depends(get_db)):
    """Get current ticker prices with 24h stats for all configured symbols."""
    tickers = []

    for symbol in settings.symbols_list:
        ticker = {
            "symbol": symbol,
            "price": 0.0,
            "change_24h": 0.0,
            "volume_24h": 0.0,
            "high_24h": 0.0,
            "low_24h": 0.0,
        }

        # Try live Binance price first
        try:
            from app.services.exchange.binance_client import binance_client
            price = await binance_client.get_ticker_price(symbol)
            ticker["price"] = float(price)
        except Exception:
            pass

        # Get 24h stats from DB (last 24 1h candles)
        try:
            result = await db.execute(
                select(OHLCV)
                .where(OHLCV.symbol == symbol.upper(), OHLCV.interval == "1h")
                .order_by(OHLCV.open_time.desc())
                .limit(24)
            )
            candles = list(result.scalars().all())
            if candles:
                # Use latest close as price fallback
                if ticker["price"] == 0.0:
                    ticker["price"] = float(candles[0].close)

                highs = [float(c.high) for c in candles]
                lows = [float(c.low) for c in candles]
                volumes = [float(c.volume) for c in candles]

                ticker["high_24h"] = max(highs)
                ticker["low_24h"] = min(lows)
                ticker["volume_24h"] = sum(volumes)

                # 24h change: (latest close - oldest open) / oldest open
                oldest = candles[-1]
                open_price = float(oldest.open)
                if open_price > 0:
                    ticker["change_24h"] = (ticker["price"] - open_price) / open_price
        except Exception as e:
            logger.warning("ticker_db_stats_failed", symbol=symbol, error=str(e))

        tickers.append(ticker)

    return tickers


@router.get("/depth/{symbol}")
async def get_order_book_depth(symbol: str, limit: int = Query(default=25, ge=5, le=100)):
    """Get order book depth (bids and asks) for a symbol."""
    try:
        from app.services.exchange.binance_client import binance_client
        depth = await binance_client.get_order_book(symbol.upper(), limit=limit)
        return depth
    except Exception as e:
        logger.warning("depth_fetch_failed", symbol=symbol, error=str(e))
        return {"bids": [], "asks": [], "symbol": symbol.upper()}


@router.get("/sentiment")
async def get_market_sentiment():
    """Get market sentiment indicators."""
    try:
        from app.services.ml.sentiment import sentiment_analyzer
        result = await sentiment_analyzer.get_composite_sentiment()
        return result
    except Exception as e:
        logger.warning("sentiment_fetch_failed", error=str(e))
        return {
            "fear_greed_index": 50,
            "fear_greed_label": "Neutral",
            "funding_rates": {},
            "long_short_ratio": {},
            "open_interest": {},
        }

"""Market data endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.logging import get_logger
from app.dependencies import get_db
from app.models.market import OHLCV
from app.schemas.market import OHLCVResponse

logger = get_logger(__name__)

router = APIRouter()


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

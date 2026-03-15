"""Bootstrap historical candles from CryptoCompare on startup.

Fetches 500+ candles of 15m data so the engine can start trading
immediately after deploy instead of waiting 7.5 hours.
CryptoCompare histominute is free and NOT geo-blocked.
"""

import asyncio
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.core.events import event_bus
from app.core.logging import get_logger
from app.models.base import async_session_factory
from app.models.market import OHLCV
from sqlalchemy import select

logger = get_logger(__name__)

_CC_SYMBOLS = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "BNBUSDT": "BNB",
    "SOLUSDT": "SOL",
    "XRPUSDT": "XRP",
    "DOGEUSDT": "DOGE",
    "ADAUSDT": "ADA",
    "AVAXUSDT": "AVAX",
    "DOTUSDT": "DOT",
    "LINKUSDT": "LINK",
}


async def bootstrap_historical_candles() -> dict[str, int]:
    """Fetch historical 15m candles from CryptoCompare and store in DB.

    Returns dict of {symbol: candles_inserted}.
    """
    results = {}

    for symbol in settings.symbols_list:
        cc_sym = _CC_SYMBOLS.get(symbol)
        if not cc_sym:
            results[symbol] = 0
            continue

        try:
            # Check how many candles we already have
            async with async_session_factory() as db:
                count_result = await db.execute(
                    select(OHLCV)
                    .where(OHLCV.symbol == symbol, OHLCV.interval == "15m")
                    .limit(1)
                )
                existing = count_result.scalar_one_or_none()

            if existing:
                # Already have data, check count
                async with async_session_factory() as db:
                    from sqlalchemy import func
                    count_result = await db.execute(
                        select(func.count())
                        .select_from(OHLCV)
                        .where(OHLCV.symbol == symbol, OHLCV.interval == "15m")
                    )
                    count = count_result.scalar() or 0

                if count >= 200:
                    logger.info("bootstrap_skipped", symbol=symbol, existing_candles=count)
                    results[symbol] = 0
                    continue
                else:
                    logger.info("bootstrap_supplementing", symbol=symbol, existing_candles=count)

            inserted = await _fetch_and_store(symbol, cc_sym)
            results[symbol] = inserted

        except Exception as e:
            logger.error("bootstrap_failed", symbol=symbol, error=str(e))
            results[symbol] = 0

    logger.info("bootstrap_complete", results=results)
    return results


async def _fetch_and_store(symbol: str, cc_sym: str) -> int:
    """Fetch 15m candles by aggregating 15x 1-minute candles from CryptoCompare."""
    async with httpx.AsyncClient(timeout=30) as client:
        # CryptoCompare histominute: get 2000 1-minute candles (≈33 hours)
        # We'll aggregate them into 15-minute candles (≈133 candles)
        resp = await client.get(
            "https://min-api.cryptocompare.com/data/v2/histominute",
            params={
                "fsym": cc_sym,
                "tsym": "USD",
                "limit": 2000,
            },
        )

        if resp.status_code != 200:
            logger.warning("cryptocompare_fetch_failed", status=resp.status_code)
            return 0

        data = resp.json()
        if data.get("Response") != "Success":
            logger.warning("cryptocompare_api_error", message=data.get("Message", ""))
            return 0

        candles_1m = data.get("Data", {}).get("Data", [])
        if not candles_1m:
            return 0

        # Aggregate 1m candles into 15m candles
        candles_15m = _aggregate_candles(candles_1m, interval_minutes=15)

        # Store in database
        inserted = 0
        async with async_session_factory() as db:
            for c in candles_15m:
                open_time = datetime.fromtimestamp(c["time"], tz=timezone.utc).replace(tzinfo=None)

                # Dedup check
                existing = await db.execute(
                    select(OHLCV).where(
                        OHLCV.symbol == symbol,
                        OHLCV.interval == "15m",
                        OHLCV.open_time == open_time,
                    ).limit(1)
                )
                if existing.scalar_one_or_none():
                    continue

                close_time = datetime.fromtimestamp(
                    c["time"] + 15 * 60 - 1, tz=timezone.utc
                ).replace(tzinfo=None)

                ohlcv = OHLCV(
                    symbol=symbol,
                    interval="15m",
                    open_time=open_time,
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"],
                    close_time=close_time,
                    quote_volume=c.get("quote_volume", 0),
                    trade_count=0,
                )
                db.add(ohlcv)
                inserted += 1

            await db.commit()

        logger.info("bootstrap_stored", symbol=symbol, candles=inserted)
        return inserted


def _aggregate_candles(candles_1m: list[dict], interval_minutes: int = 15) -> list[dict]:
    """Aggregate 1-minute candles into N-minute candles."""
    if not candles_1m:
        return []

    interval_seconds = interval_minutes * 60
    aggregated = []
    current_group: list[dict] = []
    current_boundary = None

    for candle in candles_1m:
        ts = candle["time"]
        boundary = (ts // interval_seconds) * interval_seconds

        if current_boundary is None:
            current_boundary = boundary

        if boundary != current_boundary:
            # Close current group
            if current_group:
                aggregated.append(_merge_group(current_group, current_boundary))
            current_group = [candle]
            current_boundary = boundary
        else:
            current_group.append(candle)

    # Don't add the last incomplete group
    return aggregated


def _merge_group(group: list[dict], boundary_ts: int) -> dict:
    """Merge a list of 1m candles into one aggregated candle."""
    return {
        "time": boundary_ts,
        "open": group[0]["open"],
        "high": max(c["high"] for c in group),
        "low": min(c["low"] for c in group),
        "close": group[-1]["close"],
        "volume": sum(c.get("volumefrom", 0) for c in group),
        "quote_volume": sum(c.get("volumeto", 0) for c in group),
    }

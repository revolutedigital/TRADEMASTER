"""Script to download and store 2 years of historical OHLCV data from Binance."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.core.logging import setup_logging, get_logger
from app.models.base import async_session_factory, engine, Base
from app.services.exchange.binance_client import binance_client
from app.services.market.data_collector import market_data_collector

logger = get_logger(__name__)

INTERVALS = ["1h", "4h"]
DAYS_BACK = {
    "1h": 90,     # 3 months (trading engine uses 1h and 4h)
    "4h": 180,    # 6 months
}


async def main():
    setup_logging()
    logger.info("seed_historical_start", symbols=settings.symbols_list)

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Connect to Binance
    await binance_client.connect()

    try:
        for symbol in settings.symbols_list:
            for interval in INTERVALS:
                days = DAYS_BACK[interval]
                logger.info(
                    "seeding",
                    symbol=symbol,
                    interval=interval,
                    days_back=days,
                )
                async with async_session_factory() as db:
                    count = await market_data_collector.seed_historical(
                        db=db,
                        symbol=symbol,
                        interval=interval,
                        days_back=days,
                    )
                    logger.info(
                        "seeded",
                        symbol=symbol,
                        interval=interval,
                        candles=count,
                    )
    finally:
        await binance_client.disconnect()

    logger.info("seed_historical_complete")


if __name__ == "__main__":
    asyncio.run(main())

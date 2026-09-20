"""Time-boxed probe of NautilusTrader's backtest speed on one month of M1 quotes.

Run with a Python that has `nautilus_trader` installed (kept out of the project's requirements).
It feeds bid/ask quote ticks and a strategy that flips its position every N quotes, so that the
order path is exercised, and reports events per second. It does not evaluate a strategy.
"""

from __future__ import annotations

import sys
import time

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Money, Quantity
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy


class FlipConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    every: int = 500


class Flip(Strategy):
    def __init__(self, config: FlipConfig) -> None:
        super().__init__(config)
        self.count = 0
        self.long = False

    def on_start(self) -> None:
        self.subscribe_quote_ticks(self.config.instrument_id)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        self.count += 1
        if self.count % self.config.every:
            return
        instrument = self.cache.instrument(self.config.instrument_id)
        side = OrderSide.SELL if self.long else OrderSide.BUY
        self.long = not self.long
        self.submit_order(
            self.order_factory.market(self.config.instrument_id, side, Quantity.from_int(10_000)),
        )
        del instrument


def main(parquet_path: str) -> int:
    frame = pd.read_parquet(parquet_path)
    quotes = pd.DataFrame(
        {
            "bid_price": frame["bid_open"].to_numpy(),
            "ask_price": frame["ask_open"].to_numpy(),
            "bid_size": 1_000_000.0,
            "ask_size": 1_000_000.0,
        },
        index=frame.index,
    )
    venue = Venue("SIM")
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD", venue)
    ticks = QuoteTickDataWrangler(instrument).process(quotes)

    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
    )
    engine.add_instrument(instrument)
    engine.add_data(ticks)
    engine.add_strategy(Flip(FlipConfig(instrument_id=instrument.id)))

    started = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - started
    sys.stdout.write(
        f"{len(ticks):,} quote ticks in {elapsed:.2f}s = {len(ticks) / elapsed / 1e3:.0f} thousand quotes/s\n"
    )
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

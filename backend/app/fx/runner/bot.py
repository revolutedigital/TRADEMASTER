"""One bot: a strategy fed with live bars whose decisions go through the executor.

It runs exactly the compiled `step` that the lab ran, through the same `StreamingRunner`, on bars
built by the same rules, and a decision is taken at the close of a bar like in the simulator. What
changes is only who fills the order: the broker, through the risk-checked executor.
"""

from __future__ import annotations

import numpy as np

from app.fx import strategy as fx
from app.fx.runner.executor import Executor
from app.fx.runner.feed import BarBuilder
from app.fx.runner.venue import Position, Quote, Venue
from app.fx.strategies.common import FAR_TARGET


class Bot:
    def __init__(self, *, key: str, symbol: str, seconds: int, step, init, state_size: int,
                 params: np.ndarray, executor: Executor, venue: Venue) -> None:
        self.key, self.symbol, self.executor, self.venue = key, symbol, executor, venue
        self.builder = BarBuilder(seconds)
        self.strategy = fx.StreamingRunner(step, init, params, state_size)

    async def on_quote(self, quote: Quote) -> None:
        bar = self.builder.on_quote(quote.time, quote.bid, quote.ask)
        if bar is not None:
            await self.on_bar(bar)

    async def on_clock(self, now: float) -> None:
        bar = self.builder.close_due(now)
        if bar is not None:
            await self.on_bar(bar)

    async def _position(self) -> Position | None:
        mine = [p for p in await self.venue.positions() if p.symbol == self.symbol]
        return mine[0] if mine else None

    async def on_bar(self, bar: np.ndarray) -> None:
        if self.executor.guard.killed:
            await self.executor.flatten_all(f"kill switch: {self.executor.guard.kill_reason}")
            return
        position = await self._position()
        intent, stop, target = self.strategy.on_bar(bar, position.side if position else fx.FLAT)
        if intent == fx.HOLD:
            return
        if intent == fx.EXIT:
            if position is not None:
                await self.executor.close(position.id, f"{self.key} exit")
            return
        side = fx.LONG if intent == fx.ENTER_LONG else fx.SHORT
        if position is not None and position.side == side:
            return
        if position is not None:
            await self.executor.close(position.id, f"{self.key} reversal")
        await self.executor.enter(
            symbol=self.symbol, side=side, stop_distance=stop,
            target_distance=None if target >= FAR_TARGET else target,
            client_order_id=f"{self.key}-{self.symbol}-{int(bar[fx.BAR_TIME])}-{side}",
        )

"""The demo runner's warm-up and loop, against the fake broker (no docker, no network)."""

import json

import numpy as np
import pytest

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates
from app.fx.runner.fake import FakeVenue
from scripts.research import fx_demo_runner as runner

RATES = ConversionRates({"EURUSD": 1.1})


class Transport:
    def __init__(self, candles: list[dict], bid=1.1000, ask=1.1001) -> None:
        self.candles, self.bid, self.ask = candles, bid, ask
        self.log: list[str] = []

    def send(self, command: str, timeout: float = 30.0) -> str:
        self.log.append(command)
        if command.startswith("price"):
            return json.dumps({"bid": self.bid, "ask": self.ask})
        if command.startswith("candles"):
            return json.dumps({"bars": self.candles})
        raise AssertionError(command)


def candles(count: int, start: float = 1_715_000_000.0, step: int = 60) -> list[dict]:
    from datetime import UTC, datetime

    out = []
    for i in range(count):
        opened = start + i * step
        price = 1.1 + 0.00001 * (i % 7)
        out.append({"timestamp": datetime.fromtimestamp(opened, UTC).isoformat().replace("+00:00", "Z"),
                    "open": price, "high": price + 0.00004, "low": price - 0.00004, "close": price + 0.00001})
    return out


def test_the_warm_up_feeds_completed_bars_only_and_builds_ask_from_bid_plus_the_spread(tmp_path) -> None:
    transport = Transport(candles(30))
    bot, *_ = runner.build("C1", transport, tmp_path)
    now = 1_715_000_000.0 + 29 * 60 + 30  # the 30th candle (index 29) is still forming

    fed = runner.warm_up(bot, transport, now)

    assert fed == 29 and any(call.startswith("candles EURUSD m1") for call in transport.log)


def test_the_strategy_is_built_from_the_pre_registered_configuration(tmp_path) -> None:
    transport = Transport([])
    c1, *_ = runner.build("C1", transport, tmp_path)
    c2, *_ = runner.build("C2", transport, tmp_path)

    assert (c1.builder.seconds, c2.builder.seconds) == (60, 300)
    assert runner.LIMITS.risk_fraction * 1000 > 0.3  # enough budget for the smallest lot at a 3 pip stop


async def test_the_loop_stops_on_the_stop_file_and_flattens(tmp_path) -> None:
    venue = FakeVenue(1000.0, RATES)
    venue.set_quote("EURUSD", 1.1000, 1.10002)
    bot, cli_venue, guard, journal, executor = runner.build("C1", Transport([]), tmp_path)
    executor.venue = venue
    bot.venue = venue
    stop = tmp_path / "STOP"
    stop.write_text("")

    reason = await runner.loop(bot, venue, guard, executor, stop, iterations=5, sleep=lambda s: _noop())

    assert reason == "stop file"


async def test_the_loop_feeds_quotes_and_heartbeats_without_error(tmp_path) -> None:
    venue = FakeVenue(1000.0, RATES)
    venue.set_quote("EURUSD", 1.1000, 1.10002)
    bot, _, guard, journal, executor = runner.build("C1", Transport([]), tmp_path)
    executor.venue = venue
    bot.venue = venue
    now = [1_715_000_000.0]

    def clock() -> float:
        now[0] += 20
        return now[0]

    reason = await runner.loop(bot, venue, guard, executor, tmp_path / "none", clock=clock, iterations=6,
                               sleep=lambda s: _noop())

    assert reason == "iterations done" and guard.is_stale(now[0] + 1000) and not guard.killed


async def _noop() -> None:
    return None

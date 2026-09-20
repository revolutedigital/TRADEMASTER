"""Live bars equal the lab's bars, and a bar's decision becomes a protected order or a close."""

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from app.fx.bars import aggregate_minutes
from app.fx.instruments import ConversionRates
from app.fx.runner import executor as ex
from app.fx.runner.bot import Bot
from app.fx.runner.executor import Executor
from app.fx.runner.fake import FakeVenue
from app.fx.runner.feed import BarBuilder
from app.fx.runner.journal import Journal
from app.fx.runner.risk import RiskGuard, RiskLimits
from app.fx.runner.venue import Quote
from app.fx.strategies import spike_fade as sf
from tests.unit.fx_strategy_checks import PIP, synthetic_frame
from tests.unit.test_fx_spike_fade import frame_from_steps, hand_built

RATES = ConversionRates({"EURUSD": 1.1})


def ticks_of(matrix: np.ndarray):
    """A quote stream with the same open, high, low and close of every minute, in time order."""
    for row in matrix:
        start = row[fx.BAR_TIME]
        for offset, (bid, ask) in zip(
            (0, 15, 30, 45),
            ((row[fx.BID_OPEN], row[fx.ASK_OPEN]), (row[fx.BID_HIGH], row[fx.ASK_HIGH]),
             (row[fx.BID_LOW], row[fx.ASK_LOW]), (row[fx.BID_CLOSE], row[fx.ASK_CLOSE])),
            strict=True,
        ):
            yield start + offset, bid, ask


@pytest.mark.parametrize("seconds", [60, 300, 900, 3600])
def test_live_bars_are_the_bars_the_lab_builds_from_the_same_minutes(seconds: int) -> None:
    frame = synthetic_frame(start="2024-05-13", weeks=1, bar_seconds=60, seed=3)
    frame = frame.drop(frame.index[500:620])  # a two-hour hole must stay a hole
    matrix = fx.bars_to_matrix(frame)
    builder = BarBuilder(seconds)

    live = [bar for t, bid, ask in ticks_of(matrix) if (bar := builder.on_quote(t, bid, ask)) is not None]
    last = builder.close_due(matrix[-1, fx.BAR_TIME] + seconds)
    live.append(last)

    assert np.array_equal(np.array(live), aggregate_minutes(matrix, seconds))


def test_a_bar_closes_by_the_clock_and_a_quiet_bucket_makes_no_bar() -> None:
    builder = BarBuilder(60)
    assert builder.on_quote(10.0, 1.1, 1.1001) is None
    assert builder.close_due(59.0) is None
    closed = builder.close_due(60.0)

    assert closed[fx.BAR_TIME] == 0.0 and closed[fx.BID_CLOSE] == 1.1
    assert builder.close_due(600.0) is None  # nothing was quoted since, so nothing to emit


def test_the_builder_rejects_out_of_order_and_crossed_quotes() -> None:
    builder = BarBuilder(60)
    builder.on_quote(130.0, 1.1, 1.1001)

    with pytest.raises(ValueError, match="time order"):
        builder.on_quote(10.0, 1.1, 1.1001)
    with pytest.raises(ValueError, match="crossed"):
        builder.on_quote(131.0, 1.1002, 1.1001)
    with pytest.raises(ValueError, match="whole number"):
        BarBuilder(90)


@pytest.fixture(autouse=True)
def no_retry_pause(monkeypatch):
    monkeypatch.setattr(ex, "RETRY_PAUSE_SECONDS", 0.0)


def make_bot(tmp_path, venue):
    guard = RiskGuard(RiskLimits(max_spread_pips=5.0), tmp_path / "risk.json")
    executor = Executor(venue, guard, Journal(tmp_path / "j.jsonl"), RATES, clock=lambda: 1_715_000_000.0)
    params = sf.spike_fade_params(threshold=4.0, max_bars=12)
    bot = Bot(key="F3a", symbol="EURUSD", seconds=300, step=sf.spike_fade_step, init=sf.spike_fade_init,
              state_size=sf.SPIKE_FADE_STATE_SIZE, params=params, executor=executor, venue=venue)
    return bot, executor


async def feed(bot: Bot, venue: FakeVenue, frame: pd.DataFrame, until: int) -> None:
    matrix = fx.bars_to_matrix(frame)
    for row in matrix[:until]:
        venue.set_quote("EURUSD", row[fx.BID_CLOSE], row[fx.ASK_CLOSE], row[fx.BAR_TIME])
        await bot.on_bar(row)


async def test_a_spike_becomes_a_protected_order_with_the_bar_in_its_client_id(tmp_path) -> None:
    frame, at = hand_built(+10.0)
    venue = FakeVenue(500.0, RATES)
    bot, executor = make_bot(tmp_path, venue)

    await feed(bot, venue, frame, at + 1)

    (position,) = await venue.positions()
    assert position.side == fx.SHORT and position.stop_price > position.entry_price
    assert position.target_price < position.entry_price
    assert position.client_order_id == f"F3a-EURUSD-{int(frame.index[at].timestamp())}--1"


async def test_the_time_exit_closes_the_position_after_twelve_bars(tmp_path) -> None:
    frame, at = hand_built(+10.0)
    venue = FakeVenue(500.0, RATES)
    bot, _ = make_bot(tmp_path, venue)

    await feed(bot, venue, frame, at + 1)
    assert len(await venue.positions()) == 1
    await feed(bot, venue, frame.iloc[at + 1 :], 11)
    assert len(await venue.positions()) == 1
    await feed(bot, venue, frame.iloc[at + 12 :], 1)

    assert await venue.positions() == []


async def test_a_killed_guard_flattens_and_stops_trading(tmp_path) -> None:
    frame, at = hand_built(+10.0)
    venue = FakeVenue(500.0, RATES)
    bot, executor = make_bot(tmp_path, venue)
    await feed(bot, venue, frame, at + 1)
    assert len(await venue.positions()) == 1

    executor.guard.kill("operator")
    await feed(bot, venue, frame.iloc[at + 1 :], 1)

    assert await venue.positions() == []
    later, _ = hand_built(+12.0, spike_at=130)
    await feed(bot, venue, later, 140)
    assert await venue.positions() == []


async def test_a_strategy_without_a_target_sends_none_to_the_broker(tmp_path) -> None:
    from app.fx.runner import bot as bot_module
    from app.fx.strategies import fixing_flow as ff

    frame = synthetic_frame(start="2024-05-13", weeks=1, bar_seconds=300, seed=2, sigma_pips=1.0)
    venue = FakeVenue(500.0, RATES)
    bot, _ = make_bot(tmp_path, venue)
    bot.strategy = fx.StreamingRunner(
        ff.fixing_flow_step, ff.fixing_flow_init, ff.pre_fixing_params("EURUSD"), ff.FIXING_FLOW_STATE_SIZE
    )
    matrix = fx.bars_to_matrix(frame)

    for row in matrix:
        venue.set_quote("EURUSD", row[fx.BID_CLOSE], row[fx.ASK_CLOSE], row[fx.BAR_TIME])
        await bot.on_bar(row)
        if await venue.positions():
            break

    (position,) = await venue.positions()
    assert bot_module.FAR_TARGET >= 1e5 and position.target_price is None and position.stop_price is not None

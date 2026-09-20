"""The runner core against a fake broker: sizing, server-side stops, limits, kill switch, restart."""

from pathlib import Path

import pytest

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates
from app.fx.runner import executor as ex
from app.fx.runner.executor import Executor, ProtectionError
from app.fx.runner.fake import FakeVenue
from app.fx.runner.journal import Journal
from app.fx.runner.reconcile import reconcile
from app.fx.runner.risk import RiskGuard, RiskLimits
from app.fx.runner.venue import VenueUnavailable

RATES = ConversionRates({"EURUSD": 1.1, "GBPUSD": 1.3, "USDJPY": 150.0})
NOW = 1_715_000_000.0
DAY = 86_400.0


class Bench:
    def __init__(self, tmp_path: Path, venue=None, limits=RiskLimits(), clock=lambda: NOW) -> None:
        self.venue = venue or FakeVenue(500.0, RATES)
        self.venue.set_quote("EURUSD", 1.10000, 1.10008)
        self.state = tmp_path / "risk.json"
        self.guard = RiskGuard(limits, self.state)
        self.journal = Journal(tmp_path / "journal.jsonl")
        self.executor = Executor(self.venue, self.guard, self.journal, RATES, clock=clock)

    async def buy(self, order_id="o1", stop=0.0010, target=0.0015):
        return await self.executor.enter(symbol="EURUSD", side=fx.LONG, stop_distance=stop,
                                         target_distance=target, client_order_id=order_id)


@pytest.fixture(autouse=True)
def no_retry_pause(monkeypatch):
    monkeypatch.setattr(ex, "RETRY_PAUSE_SECONDS", 0.0)


async def test_an_entry_is_sized_by_risk_and_carries_a_server_side_stop_and_target(tmp_path) -> None:
    bench = Bench(tmp_path)

    position = await bench.buy()

    assert position.units == 1000  # 10 pips at 0.01 lot risks $1.00, inside 0.25% of $500
    assert position.stop_price == pytest.approx(1.10008 - 0.0010)
    assert position.target_price == pytest.approx(1.10008 + 0.0015)
    events = [e["event"] for e in bench.journal.events()]
    assert events == ["order_intent", "order_filled"]


async def test_a_stop_too_wide_for_the_smallest_lot_is_refused_without_sending_anything(tmp_path) -> None:
    bench = Bench(tmp_path)

    position = await bench.buy(stop=0.0015)  # 15 pips: the smallest lot would risk $1.50 of a $1.25 budget

    assert position is None and bench.venue.orders_sent == 0
    assert "smallest lot" in bench.journal.events()[-1]["reason"]


async def test_the_position_cap_and_the_spread_cap_refuse_entries(tmp_path) -> None:
    bench = Bench(tmp_path)
    await bench.buy("o1")

    second = await bench.buy("o2")
    assert second is None and "position cap" in bench.journal.events()[-1]["reason"]

    bench.venue.set_quote("EURUSD", 1.10000, 1.10030)
    await bench.executor.flatten_all("test")
    wide = await bench.buy("o3")
    assert wide is None and "spread" in bench.journal.events()[-1]["reason"]


async def test_a_lost_response_is_retried_without_opening_a_second_position(tmp_path) -> None:
    class LostResponse(FakeVenue):
        lost = 1

        async def market_order(self, **kwargs):
            position = await super().market_order(**kwargs)
            if self.lost:
                self.lost -= 1
                raise VenueUnavailable("the answer never arrived")
            return position

    bench = Bench(tmp_path, venue=LostResponse(500.0, RATES))

    position = await bench.buy()

    assert position is not None
    assert bench.venue.orders_sent == 1 and len(await bench.venue.positions()) == 1


async def test_the_same_order_id_is_never_sent_twice(tmp_path) -> None:
    bench = Bench(tmp_path)
    await bench.buy("same")
    await bench.executor.flatten_all("test")

    again = await bench.buy("same")

    assert again is None and bench.venue.orders_sent == 1


async def test_an_outage_that_outlasts_the_retries_is_raised(tmp_path) -> None:
    bench = Bench(tmp_path)
    bench.venue.unavailable_calls = 10

    with pytest.raises(VenueUnavailable):
        await bench.buy()


async def test_a_broker_that_drops_the_stop_gets_the_position_closed_and_the_bot_stopped(tmp_path) -> None:
    bench = Bench(tmp_path)
    bench.venue.ignores_protection = True

    with pytest.raises(ProtectionError):
        await bench.buy()

    assert await bench.venue.positions() == []
    assert bench.guard.killed and "protective stop" in bench.guard.kill_reason
    assert bench.journal.events()[-1]["event"] == "closed"


async def test_the_daily_loss_cap_trips_the_kill_switch_and_survives_a_restart(tmp_path) -> None:
    bench = Bench(tmp_path)
    bench.guard.observe(NOW, 500.0)

    bench.guard.observe(NOW + 60, 494.9)  # $5.10 lost against a $5.00 cap
    assert bench.guard.killed and "daily loss" in bench.guard.kill_reason

    assert await bench.buy() is None and "kill switch" in bench.journal.events()[-1]["reason"]
    assert RiskGuard(RiskLimits(), bench.state).killed
    bench.guard.reset()
    assert not RiskGuard(RiskLimits(), bench.state).killed


async def test_an_entry_whose_stop_would_break_the_daily_cap_is_refused(tmp_path) -> None:
    bench = Bench(tmp_path)
    bench.guard.observe(NOW, 500.0)
    bench.venue.balance = 495.5  # $4.50 already lost today, $0.50 left, and the stop would risk $1.00

    position = await bench.buy()

    assert position is None and "loss cap" in bench.journal.events()[-1]["reason"]
    assert not bench.guard.killed


async def test_a_new_fx_day_restarts_the_allowance_and_a_restart_mid_day_does_not(tmp_path) -> None:
    bench = Bench(tmp_path)
    bench.guard.observe(NOW, 500.0)

    bench.guard.observe(NOW + 60, 497.0)
    restarted = RiskGuard(RiskLimits(), bench.state)
    restarted.observe(NOW + 120, 497.0)
    assert restarted.day_loss(497.0) == pytest.approx(3.0)  # still counted from 500, not from 497

    restarted.observe(NOW + 2 * DAY, 490.0)
    assert restarted.day_loss(490.0) == 0.0 and not restarted.killed


async def test_flatten_all_closes_every_position(tmp_path) -> None:
    bench = Bench(tmp_path, limits=RiskLimits(max_open_positions=3))
    await bench.buy("a")
    await bench.buy("b")

    closed = await bench.executor.flatten_all("kill switch")

    assert closed == 2 and await bench.venue.positions() == []


def test_a_silent_process_is_detected_by_its_stale_heartbeat(tmp_path) -> None:
    guard = RiskGuard(RiskLimits(heartbeat_seconds=60), tmp_path / "s.json")
    assert not guard.is_stale(NOW)
    guard.heartbeat(NOW)

    assert not guard.is_stale(NOW + 59) and guard.is_stale(NOW + 61)


async def test_the_server_side_stop_fires_while_the_bot_is_down_and_reconcile_records_it(tmp_path) -> None:
    bench = Bench(tmp_path)
    position = await bench.buy()

    bench.venue.set_quote("EURUSD", 1.09850, 1.09858)  # the market falls through the stop with no bot alive
    report = await reconcile(bench.venue, bench.journal)

    assert await bench.venue.positions() == []
    assert bench.venue.balance < 500.0
    assert report.closed_while_down == [position.id]
    assert bench.journal.open_position_ids() == set()


async def test_reconcile_adopts_a_protected_orphan_and_closes_an_unprotected_one(tmp_path) -> None:
    bench = Bench(tmp_path, limits=RiskLimits(max_open_positions=3))
    protected = await bench.venue.market_order(symbol="EURUSD", side=fx.LONG, units=1000, stop_price=1.099,
                                               target_price=None, client_order_id="manual-1")
    bench.venue.ignores_protection = True
    naked = await bench.venue.market_order(symbol="EURUSD", side=fx.LONG, units=1000, stop_price=1.099,
                                           target_price=None, client_order_id="manual-2")

    report = await reconcile(bench.venue, bench.journal)

    assert report.adopted == [protected.id] and report.closed_orphans == [naked.id]
    assert [p.id for p in await bench.venue.positions()] == [protected.id]
    assert bench.journal.open_position_ids() == {protected.id}


async def test_reconcile_restores_a_lost_stop_from_the_journal(tmp_path) -> None:
    bench = Bench(tmp_path)
    position = await bench.buy()
    bench.venue._positions[position.id] = type(position)(**{**position.__dict__, "stop_price": None})

    report = await reconcile(bench.venue, bench.journal)

    assert report.restored_stops == [position.id]
    assert (await bench.venue.positions())[0].stop_price == pytest.approx(position.stop_price)


async def test_reconcile_is_idempotent(tmp_path) -> None:
    bench = Bench(tmp_path)
    await bench.buy()

    first = await reconcile(bench.venue, bench.journal)
    second = await reconcile(bench.venue, bench.journal)

    assert first.adopted == first.closed_orphans == second.adopted == second.closed_while_down == []


async def test_levels_anchor_to_the_reference_price_and_a_crossed_market_is_refused(tmp_path) -> None:
    bench = Bench(tmp_path)

    position = await bench.executor.enter(symbol="EURUSD", side=fx.LONG, stop_distance=0.0010, target_distance=0.0015,
                                          client_order_id="ref", reference_price=1.10004)
    assert position.stop_price == pytest.approx(1.10004 - 0.0010)
    assert position.target_price == pytest.approx(1.10004 + 0.0015)

    await bench.executor.flatten_all("test")
    bench.venue.set_quote("EURUSD", 1.0990, 1.09908)  # the market fell below the long's stop level
    crossed = await bench.executor.enter(symbol="EURUSD", side=fx.LONG, stop_distance=0.0005, target_distance=0.0015,
                                         client_order_id="late", reference_price=1.1000)
    assert crossed is None and "crossed" in bench.journal.events()[-1]["reason"]


async def test_the_journal_becomes_closed_trades_with_r_from_the_risk_taken(tmp_path) -> None:
    bench = Bench(tmp_path)
    position = await bench.buy()
    bench.venue.set_quote("EURUSD", 1.10100, 1.10108)
    result = await bench.executor.close(position.id, "strategy exit")

    from app.fx.runner.journal import closed_trades

    (trade,) = closed_trades(bench.journal.events())

    assert trade.symbol == "EURUSD" and trade.side == fx.LONG and trade.pnl == pytest.approx(result)
    assert trade.r_multiple == pytest.approx(result / 1.0)  # 1000 units, 10 pips: $1.00 at the stop
    assert trade.exit_time >= trade.entry_time and trade.exit_reason == "strategy exit"


async def test_a_trade_still_open_or_closed_without_a_known_result_is_not_a_closed_trade(tmp_path) -> None:
    from app.fx.runner.journal import closed_trades

    bench = Bench(tmp_path)
    await bench.buy()
    assert closed_trades(bench.journal.events()) == []
    bench.venue.set_quote("EURUSD", 1.09850, 1.09858)
    await reconcile(bench.venue, bench.journal)  # the server stop fired while the bot was away: no result known

    assert closed_trades(bench.journal.events()) == []


class RefusesAmend(FakeVenue):
    async def amend_protection(self, position_id, *, stop_price, target_price):
        from app.fx.runner.venue import OrderRejected

        raise OrderRejected("INVALID_STOP")


async def test_a_refused_stop_amendment_leaves_the_position_closed_and_the_bot_stopped(tmp_path) -> None:
    venue = RefusesAmend(500.0, RATES)
    venue.ignores_protection = True
    bench = Bench(tmp_path, venue=venue)

    with pytest.raises(ProtectionError):
        await bench.buy()

    assert await venue.positions() == [] and bench.guard.killed


async def test_reconcile_closes_an_unprotected_position_whose_stop_the_broker_refuses_to_restore(tmp_path) -> None:
    venue = RefusesAmend(500.0, RATES)
    bench = Bench(tmp_path, venue=venue)
    position = await bench.buy()
    venue._positions[position.id] = type(position)(**{**position.__dict__, "stop_price": None})

    report = await reconcile(venue, bench.journal)

    assert report.closed_orphans == [position.id] and await venue.positions() == []

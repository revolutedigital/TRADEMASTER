"""The cTrader `Venue`: orders that survive an outage, exact server-side stops, money and prices."""

import asyncio
import contextlib
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates
from app.fx.runner import executor as executor_module
from app.fx.runner.ctrader.client import CTraderClient, payload_type_of
from app.fx.runner.ctrader.proto import OpenApiMessages_pb2 as messages
from app.fx.runner.ctrader.proto import OpenApiModelMessages_pb2 as model
from app.fx.runner.ctrader.venue import CTraderConfigError, CTraderProtocolError, CTraderVenue
from app.fx.runner.executor import Executor, ProtectionError
from app.fx.runner.journal import Journal
from app.fx.runner.reconcile import reconcile
from app.fx.runner.risk import RiskGuard, RiskLimits
from app.fx.runner.venue import OrderRejected, VenueUnavailable
from scripts.research import ctrader_probe as probe_module
from tests.unit.fx_ctrader_server import SYMBOLS, FakeCTraderServer, client_for, eventually

NEW_ORDER_REQ = payload_type_of(messages.ProtoOANewOrderReq())
CLOSE_REQ = payload_type_of(messages.ProtoOAClosePositionReq())
RATES = ConversionRates({"EURUSD": 1.1, "GBPUSD": 1.3, "USDJPY": 150.0})
ORDER_ID = "F1a-EURUSD-1750000000-1"
STOP, TARGET = 1.09908, 1.10158  # 10 and 15 pips from the 1.10008 ask
QUOTES = {
    "EURUSD": (1.10000, 1.10008),
    "GBPUSD": (1.30000, 1.30010),
    "USDJPY": (150.000, 150.010),
    "EURJPY": (163.100, 163.120),
}


@dataclass
class Rig:
    server: FakeCTraderServer
    client: CTraderClient
    venue: CTraderVenue


@contextlib.asynccontextmanager
async def rigged(server, symbols=("EURUSD", "GBPUSD", "USDJPY"), **client_settings):
    for symbol in symbols:
        if SYMBOLS[symbol][0] not in server.quotes:
            server.set_quote(symbol, *QUOTES[symbol])
    client = client_for(server, **client_settings)
    venue = CTraderVenue(
        client, symbols=symbols, trade_timeout=0.3, confirm_pause=0.01, max_quote_age=60.0
    )
    async with client:
        await venue.start()
        await eventually(lambda: set(symbols) <= set(venue.conversion_rates().mids))
        yield Rig(server, client, venue)


@pytest.fixture
async def server():
    async with FakeCTraderServer() as running:
        yield running


@pytest.fixture
async def rig(server):
    async with rigged(server) as ready:
        yield ready


async def buy(rig: Rig, order_id: str = ORDER_ID, units: int = 1000, **overrides):
    order = {
        "symbol": "EURUSD",
        "side": fx.LONG,
        "units": units,
        "stop_price": STOP,
        "target_price": TARGET,
        "client_order_id": order_id,
    }
    return await rig.venue.market_order(**{**order, **overrides})


# prices and the feed


async def test_start_loads_the_symbols_and_the_first_prices_arrive_with_the_last_quote(rig) -> None:
    quote = await rig.venue.quote("EURUSD")

    assert (quote.symbol, quote.bid, quote.ask) == ("EURUSD", 1.10000, 1.10008)
    assert abs(quote.time - time.time()) < 5
    [subscription] = rig.server.requests_of(messages.ProtoOASubscribeSpotsReq)
    assert list(subscription.symbolId) == [1, 2, 3]


async def test_the_feed_yields_each_price_update_in_order(rig) -> None:
    updates = rig.venue.spots()
    for _ in range(3):  # the last price of each subscribed symbol, sent on subscription
        await asyncio.wait_for(anext(updates), 1)

    await rig.server.push_spot("EURUSD", 1.1001, 1.10018)
    await rig.server.push_spot("EURUSD", 1.1002, 1.10028)

    first = await asyncio.wait_for(anext(updates), 1)
    assert (first.symbol, first.bid, first.ask) == ("EURUSD", 1.1001, 1.10018)
    assert (await asyncio.wait_for(anext(updates), 1)).bid == 1.1002


async def test_a_half_update_is_merged_and_a_transient_cross_is_not_published(rig) -> None:
    await rig.server.push_spot("EURUSD", bid=1.10010)  # bid above the old ask: crossed for now
    await asyncio.sleep(0.05)
    assert (await rig.venue.quote("EURUSD")).bid == 1.10000

    await rig.server.push_spot("EURUSD", ask=1.10018)
    await eventually(lambda: rig.venue.conversion_rates().mids["EURUSD"] > 1.1001)

    quote = await rig.venue.quote("EURUSD")
    assert (quote.bid, quote.ask) == (1.10010, 1.10018)


async def test_an_old_or_missing_price_is_venue_unavailable_never_a_guess(server) -> None:
    server.quotes.clear()
    async with rigged(server, symbols=("EURUSD", "USDJPY")) as ready:
        await ready.server.push_spot("EURUSD", 1.1, 1.10008, at_ms=int((time.time() - 600) * 1000))

        await eventually(lambda: ready.venue._quotes["EURUSD"].time < time.time() - 500)

        with pytest.raises(VenueUnavailable, match="s old"):
            await ready.venue.quote("EURUSD")
        with pytest.raises(VenueUnavailable, match="no price"):
            await ready.venue.quote("GBPUSD")  # never subscribed


async def test_a_timestamp_that_is_not_plausible_is_replaced_by_the_local_clock(rig) -> None:
    await rig.server.push_spot("EURUSD", 1.1005, 1.10058, at_ms=1_700_000)  # 1970, i.e. seconds

    await eventually(lambda: rig.venue.conversion_rates().mids["EURUSD"] > 1.1004)

    assert abs((await rig.venue.quote("EURUSD")).time - time.time()) < 5


async def test_symbol_names_written_with_a_slash_are_understood() -> None:
    async with FakeCTraderServer(slash_names=True) as server:
        async with rigged(server) as ready:
            position = await buy(ready)

    assert position.symbol == "EURUSD"


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda s: None, "no symbol AUDUSD"),
        (lambda s: setattr(s, "currency", "EUR"), "account is in EUR"),
        (lambda s: setattr(s, "account_type", model.NETTED), "hedging"),
        (lambda s: s.pip_positions.update(EURUSD=5), "pip"),
    ],
)
async def test_an_account_that_does_not_match_the_runner_is_refused_at_start(
    setup, message
) -> None:
    async with FakeCTraderServer() as server:
        setup(server)
        symbols = ["EURUSD", "AUDUSD"] if "AUDUSD" in message else ["EURUSD"]
        client = client_for(server)
        venue = CTraderVenue(client, symbols=symbols)
        async with client:
            with pytest.raises(CTraderConfigError, match=message):
                await venue.start()


# market orders


async def test_a_market_order_goes_out_relative_and_is_amended_to_the_exact_levels(rig) -> None:
    rig.server.fill_slippage = 0.00002  # the fill lands 0.2 pip away from the quote

    position = await buy(rig)

    order = rig.server.orders_received[0]
    assert order.orderType == model.MARKET and order.tradeSide == model.BUY
    assert not order.HasField("stopLoss") and not order.HasField("takeProfit")
    assert (order.relativeStopLoss, order.relativeTakeProfit) == (100, 150)
    assert order.label == order.clientOrderId == ORDER_ID
    assert order.volume == 100_000  # 1000 units in 1/100
    amendment = rig.server.requests_of(messages.ProtoOAAmendPositionSLTPReq)
    assert [(a.stopLoss, a.takeProfit) for a in amendment] == [(STOP, TARGET)]
    assert (position.stop_price, position.target_price) == (STOP, TARGET)
    assert position.entry_price == pytest.approx(1.10010)
    assert (position.symbol, position.side, position.units) == ("EURUSD", fx.LONG, 1000)
    assert position.client_order_id == ORDER_ID
    assert [p.client_order_id for p in await rig.venue.positions()] == [ORDER_ID]


async def test_a_sell_measures_its_relative_levels_from_the_bid(rig) -> None:
    await buy(rig, "s1", side=fx.SHORT, stop_price=1.10100, target_price=1.09850)

    order = rig.server.orders_received[0]
    assert order.tradeSide == model.SELL
    assert (order.relativeStopLoss, order.relativeTakeProfit) == (100, 150)


async def test_no_amendment_is_sent_when_the_fill_already_sits_on_the_levels(rig) -> None:
    position = await buy(rig)

    assert rig.server.requests_of(messages.ProtoOAAmendPositionSLTPReq) == []
    assert (position.stop_price, position.target_price) == (STOP, TARGET)


async def test_the_same_client_order_id_twice_is_one_position(rig) -> None:
    first = await buy(rig)
    second = await buy(rig)

    assert first == second
    assert len(rig.server.orders_received) == 1 and len(rig.server.positions) == 1


async def test_a_retry_after_the_link_died_mid_order_finds_the_position_the_first_one_opened(
    server,
) -> None:
    server.drop_after_order = True
    settings = {"backoff_initial": 0.4, "backoff_max": 0.4}
    async with rigged(server, **settings) as rig:
        with pytest.raises(VenueUnavailable):
            await buy(rig)
        assert len(server.orders_received) == 1 and len(server.positions) == 1

        server.drop_after_order = False
        await eventually(lambda: rig.client.connected)
        position = await buy(rig)

    assert len(server.orders_received) == 1 and len(server.positions) == 1
    assert position.client_order_id == ORDER_ID and position.stop_price == STOP


async def test_an_order_that_never_reached_the_broker_is_sent_again_on_retry(rig) -> None:
    rig.server.silent.add(NEW_ORDER_REQ)
    with pytest.raises(VenueUnavailable, match="unknown"):
        await buy(rig)
    assert rig.server.positions == {}

    rig.server.silent.clear()
    position = await buy(rig)

    assert len(rig.server.positions) == 1 and position.stop_price == STOP
    assert len(rig.server.requests_of(messages.ProtoOANewOrderReq)) == 2  # the first was swallowed


async def test_without_any_trading_event_the_reconcile_still_finds_the_position(server) -> None:
    server.emit_events = False
    async with rigged(server) as rig:
        position = await buy(rig)

    assert position.client_order_id == ORDER_ID and position.stop_price == STOP


@pytest.mark.parametrize("style", ["order_error", "execution", "error_res"])
async def test_a_refused_order_is_order_rejected_with_the_broker_code(rig, style) -> None:
    rig.server.reject_next_order("NOT_ENOUGH_MONEY", style)

    with pytest.raises(OrderRejected, match="NOT_ENOUGH_MONEY"):
        await buy(rig)

    assert rig.server.positions == {}


async def test_a_refusal_that_only_says_not_now_is_venue_unavailable(rig) -> None:
    rig.server.reject_next_order("BLOCKED_PAYLOAD_TYPE")

    with pytest.raises(VenueUnavailable, match="BLOCKED_PAYLOAD_TYPE"):
        await buy(rig)


async def test_a_fill_whose_label_the_broker_forgets_halts_trading_instead_of_doubling_up(
    server,
) -> None:
    server.strip_labels = True
    async with rigged(server) as rig:
        with pytest.raises(CTraderProtocolError, match="idempotency"):
            await buy(rig)
        with pytest.raises(CTraderProtocolError):
            await buy(rig, "another-id")

    assert len(server.orders_received) == 1


async def test_what_the_broker_would_refuse_is_refused_before_anything_is_sent(rig) -> None:
    with pytest.raises(OrderRejected, match="TRADING_BAD_STOPS"):
        await buy(rig, stop_price=1.10100)  # a long stop above the entry
    with pytest.raises(OrderRejected, match="TRADING_BAD_VOLUME"):
        await buy(rig, units=1500)  # off the step
    with pytest.raises(OrderRejected, match="SYMBOL_NOT_FOUND"):
        await buy(rig, symbol="EURGBP")
    for bad in ({"client_order_id": ""}, {"client_order_id": "x" * 101}, {"side": 0}, {"units": 0}):
        with pytest.raises(ValueError, match="must"):
            await buy(rig, **bad)

    assert rig.server.orders_received == []


async def test_an_id_too_long_for_client_order_id_still_travels_in_the_label(rig) -> None:
    long_id = "k" * 70

    position = await buy(rig, long_id)

    order = rig.server.orders_received[0]
    assert order.label == long_id and not order.HasField("clientOrderId")
    assert position.client_order_id == long_id


async def test_the_server_refuses_an_absolute_stop_on_a_market_order_as_the_proto_says(rig) -> None:
    order = messages.ProtoOANewOrderReq(
        ctidTraderAccountId=1001,
        symbolId=1,
        orderType=model.MARKET,
        tradeSide=model.BUY,
        volume=100_000,
        stopLoss=1.099,
    )
    await rig.client.submit(order, "raw-1")
    await eventually(lambda: len(rig.server.orders_received) == 1)

    assert rig.server.positions == {}


# protection


async def test_a_refused_amendment_leaves_the_position_with_the_relative_stop(rig) -> None:
    rig.server.fill_slippage = 0.00002
    rig.server.amend_errors.append("PROTECTION_IS_TOO_CLOSE_TO_MARKET")

    position = await buy(rig)

    assert position.stop_price == pytest.approx(1.10010 - 0.00100)
    assert len(rig.server.positions) == 1


async def test_a_broker_that_drops_every_stop_returns_a_position_with_none_for_the_executor(
    server,
) -> None:
    server.ignore_relative_sltp = server.ignore_amend = True
    async with rigged(server) as rig:
        position = await buy(rig)

    assert position.stop_price is None and position.target_price is None


async def test_amend_protection_moves_the_stop_and_the_target(rig) -> None:
    position = await buy(rig)

    moved = await rig.venue.amend_protection(position.id, stop_price=1.09950, target_price=1.10300)

    assert (moved.stop_price, moved.target_price) == (1.09950, 1.10300)
    stored = next(iter(rig.server.positions.values()))
    assert (stored.stop, stored.target) == (1.09950, 1.10300)


async def test_a_refused_amend_is_reported_as_the_position_stands_and_the_code_is_logged(
    rig, caplog
) -> None:
    position = await buy(rig)

    unchanged = await rig.venue.amend_protection(
        position.id, stop_price=1.10000, target_price=TARGET
    )

    assert unchanged.stop_price == STOP and next(iter(rig.server.positions.values())).stop == STOP
    assert "PROTECTION_IS_TOO_CLOSE_TO_MARKET" in caplog.text
    with pytest.raises(OrderRejected, match="POSITION_NOT_FOUND"):
        await rig.venue.amend_protection("123", stop_price=STOP, target_price=None)


async def test_an_amendment_the_broker_accepts_and_ignores_is_reported_as_it_is(rig) -> None:
    position = await buy(rig)
    rig.server.ignore_amend = True

    unchanged = await rig.venue.amend_protection(
        position.id, stop_price=1.09950, target_price=TARGET
    )

    assert unchanged.stop_price == STOP


# closing


async def test_close_returns_the_realized_result_from_the_closing_deal(rig) -> None:
    position = await buy(rig)
    rig.server.set_quote("EURUSD", 1.10108, 1.10116)  # +10 pips for a long

    result = await rig.venue.close(position.id)

    assert result == pytest.approx(1000 * 0.00100 - 0.08)  # gross minus the commission
    assert rig.server.positions == {} and rig.server.balance == pytest.approx(500.92)
    with pytest.raises(OrderRejected, match="POSITION_NOT_FOUND"):
        await rig.venue.close(position.id)


async def test_without_the_closing_event_the_result_is_the_confirmed_balance_change(server) -> None:
    server.emit_close_event = False
    async with rigged(server) as rig:
        position = await buy(rig)
        rig.server.set_quote("EURUSD", 1.10108, 1.10116)

        result = await rig.venue.close(position.id)

    assert result == pytest.approx(0.92)


async def test_a_close_retried_after_the_link_died_returns_the_result_without_closing_twice(
    server,
) -> None:
    server.drop_on_close = True
    async with rigged(server, backoff_initial=0.4, backoff_max=0.4) as rig:
        position = await buy(rig)
        server.set_quote("EURUSD", 1.10108, 1.10116)
        with pytest.raises(VenueUnavailable):
            await rig.venue.close(position.id)

        await eventually(lambda: rig.client.connected)
        result = await rig.venue.close(position.id)

    assert result == pytest.approx(0.92)
    assert len(server.requests_of(messages.ProtoOAClosePositionReq)) == 1


async def test_a_close_nobody_confirms_is_venue_unavailable_and_the_retry_settles_it(rig) -> None:
    position = await buy(rig)
    rig.server.silent.add(CLOSE_REQ)

    with pytest.raises(VenueUnavailable, match="not confirmed"):
        await rig.venue.close(position.id)
    assert len(rig.server.positions) == 1

    rig.server.silent.clear()
    assert await rig.venue.close(position.id) == pytest.approx(
        -0.16
    )  # -0.08 gross, -0.08 commission


async def test_a_refused_close_keeps_the_position_and_names_the_code(rig) -> None:
    position = await buy(rig)
    rig.server.close_errors += ["MARKET_CLOSED", "PENDING_EXECUTION"]

    with pytest.raises(OrderRejected, match="MARKET_CLOSED"):
        await rig.venue.close(position.id)
    with pytest.raises(VenueUnavailable, match="PENDING_EXECUTION"):
        await rig.venue.close(position.id)

    assert len(rig.server.positions) == 1
    assert await rig.venue.close(position.id) == pytest.approx(-0.08 + 1000 * (1.10000 - 1.10008))


# account


async def test_equity_is_the_balance_plus_the_unrealized_pnl_at_the_cached_prices(rig) -> None:
    await buy(rig)  # long 1000 EURUSD at 1.10008
    rig.server.add_position("GBPUSD", model.SELL, 2000, 1.30100)  # short, entered above the ask
    await rig.server.push_spot("EURUSD", 1.10108, 1.10116)
    await eventually(lambda: rig.venue.conversion_rates().mids["EURUSD"] > 1.1010)

    account = await rig.venue.account()

    long_pnl = 1000 * (1.10108 - 1.10008)
    short_pnl = 2000 * (1.30100 - 1.30010)  # closed at the ask
    assert account.balance == 500.0 and account.currency == "USD"
    assert account.equity == pytest.approx(500.0 + long_pnl + short_pnl)


async def test_a_position_in_a_pair_quoted_in_yen_is_valued_through_usdjpy(rig) -> None:
    rig.server.add_position("USDJPY", model.BUY, 1000, 150.000)
    await rig.server.push_spot("USDJPY", 150.100, 150.110)
    await eventually(lambda: rig.venue.conversion_rates().mids["USDJPY"] > 150.05)

    account = await rig.venue.account()

    assert account.equity == pytest.approx(500.0 + 1000 * 0.100 / 150.105)


async def test_equity_without_a_price_or_a_conversion_rate_is_venue_unavailable(server) -> None:
    async with rigged(server, symbols=("EURUSD", "EURJPY")) as rig:
        unsubscribed = rig.server.add_position("GBPUSD", model.BUY, 1000, 1.3)
        with pytest.raises(VenueUnavailable, match="no price for GBPUSD"):
            await rig.venue.account()

        del rig.server.positions[unsubscribed.position_id]
        rig.server.add_position("EURJPY", model.BUY, 1000, 163.000)
        with pytest.raises(VenueUnavailable, match="cannot value"):  # USDJPY is not subscribed
            await rig.venue.account()


async def test_positions_carry_the_client_order_id_from_the_label(rig) -> None:
    rig.server.add_position("GBPUSD", model.SELL, 3000, 1.3, label="manual-7")

    [position] = await rig.venue.positions()

    assert (position.symbol, position.side, position.units) == ("GBPUSD", fx.SHORT, 3000)
    assert (position.client_order_id, position.entry_price) == ("manual-7", 1.3)
    assert position.stop_price is None and position.target_price is None


async def test_text_from_the_broker_is_scrubbed_of_the_credentials(server) -> None:
    server.leak_token_in_rejections = True
    async with rigged(server) as rig:
        rig.server.reject_next_order("NOT_ENOUGH_MONEY", "error_res")

        with pytest.raises(OrderRejected) as caught:
            await buy(rig)

    assert server.access_token not in str(caught.value) and "***" in str(caught.value)


async def test_a_slow_consumer_loses_the_oldest_prices_and_the_venue_counts_them(rig) -> None:
    rig.venue._quote_queue = asyncio.Queue(2)
    for step in range(4):
        await rig.server.push_spot("EURUSD", 1.1 + step / 10_000, 1.10008 + step / 10_000)
    await eventually(lambda: rig.venue.dropped_quotes >= 2)

    remaining = [rig.venue._quote_queue.get_nowait().bid for _ in range(2)]

    assert remaining == [pytest.approx(1.1002), pytest.approx(1.1003)]


# the real executor against the same server


def executor_for(rig: Rig, tmp_path: Path) -> tuple[Executor, Journal, RiskGuard]:
    journal = Journal(tmp_path / "journal.jsonl")
    guard = RiskGuard(RiskLimits(), tmp_path / "risk.json")
    return Executor(rig.venue, guard, journal, RATES), journal, guard


@pytest.fixture(autouse=True)
def short_retry_pause(monkeypatch):
    monkeypatch.setattr(executor_module, "RETRY_PAUSE_SECONDS", 0.2)


async def test_the_executor_enters_sized_by_risk_and_the_stop_is_confirmed_on_the_server(
    rig, tmp_path
) -> None:
    executor, journal, _ = executor_for(rig, tmp_path)

    position = await executor.enter(
        symbol="EURUSD",
        side=fx.LONG,
        stop_distance=0.0010,
        target_distance=0.0015,
        client_order_id="o1",
    )

    assert position.units == 1000  # 10 pips at 0.01 lot risk $1.00, inside 0.25% of $500
    stored = next(iter(rig.server.positions.values()))
    assert (stored.stop, stored.target) == (pytest.approx(1.09908), pytest.approx(1.10158))
    assert [e["event"] for e in journal.events()] == ["order_intent", "order_filled"]
    assert (
        await executor.enter(
            symbol="EURUSD",
            side=fx.LONG,
            stop_distance=0.0010,
            target_distance=0.0015,
            client_order_id="o1",
        )
        is None
    )
    assert len(rig.server.orders_received) == 1


async def test_the_executor_survives_an_outage_in_the_middle_of_its_entry(server, tmp_path) -> None:
    server.drop_after_order = True
    async with rigged(server, backoff_initial=0.05, backoff_max=0.05) as rig:
        rig.venue._trade_timeout = 0.05
        executor, _, _ = executor_for(rig, tmp_path)

        position = await executor.enter(
            symbol="EURUSD",
            side=fx.LONG,
            stop_distance=0.0010,
            target_distance=0.0015,
            client_order_id="o1",
        )

    assert position.stop_price == pytest.approx(1.09908)
    assert len(server.orders_received) == 1 and len(server.positions) == 1


async def test_the_executor_closes_a_position_the_broker_left_without_a_stop(
    server, tmp_path
) -> None:
    server.ignore_relative_sltp = server.ignore_amend = True
    async with rigged(server) as rig:
        executor, journal, guard = executor_for(rig, tmp_path)

        with pytest.raises(ProtectionError):
            await executor.enter(
                symbol="EURUSD",
                side=fx.LONG,
                stop_distance=0.0010,
                target_distance=0.0015,
                client_order_id="o1",
            )

    assert server.positions == {} and guard.killed
    assert [e["event"] for e in journal.events()] == ["order_intent", "closed"]


async def test_a_refused_amend_on_a_stopless_position_still_ends_in_a_close(
    server, tmp_path
) -> None:
    server.ignore_relative_sltp = True
    server.amend_errors += ["PROTECTION_IS_TOO_CLOSE_TO_MARKET"] * 2
    async with rigged(server) as rig:
        executor, _, guard = executor_for(rig, tmp_path)

        with pytest.raises(ProtectionError):
            await executor.enter(
                symbol="EURUSD",
                side=fx.LONG,
                stop_distance=0.0010,
                target_distance=0.0015,
                client_order_id="o1",
            )

    assert server.positions == {} and guard.killed


async def test_the_executor_closes_and_flattens_through_the_venue(rig, tmp_path) -> None:
    executor, journal, _ = executor_for(rig, tmp_path)
    position = await executor.enter(
        symbol="EURUSD",
        side=fx.LONG,
        stop_distance=0.0010,
        target_distance=0.0015,
        client_order_id="o1",
    )
    rig.server.set_quote("EURUSD", 1.10108, 1.10116)

    result = await executor.close(position.id, "exit")

    assert result == pytest.approx(0.92) and rig.server.positions == {}
    assert journal.events()[-1]["event"] == "closed"


async def test_boot_reconciliation_adopts_a_protected_position_and_closes_a_naked_one(
    rig, tmp_path
) -> None:
    protected = rig.server.add_position("EURUSD", model.BUY, 1000, 1.10008, label="kept")
    protected.stop = 1.0990
    naked = rig.server.add_position("GBPUSD", model.BUY, 1000, 1.3001, label="naked")
    _, journal, _ = executor_for(rig, tmp_path)

    report = await reconcile(rig.venue, journal)

    assert report.adopted == [str(protected.position_id)]
    assert report.closed_orphans == [str(naked.position_id)]
    assert list(rig.server.positions) == [protected.position_id]


# the demo probe, its mechanics checked against the same fake server (what the real broker does
# is what the probe is for; the fake's answers for MARKET_RANGE are invented)


def probe_for(rig: Rig, **settings) -> probe_module.Probe:
    def make_client(credentials):
        return client_for(rig.server, credentials)

    return probe_module.Probe(
        rig.client, rig.venue, make_client, rig.server.credentials, idle_seconds=0.0, **settings
    )


async def test_the_probe_runs_the_matrix_and_leaves_the_account_flat(rig, capsys) -> None:
    async def keep_the_market_ticking() -> None:
        while True:
            await rig.server.push_spot("EURUSD", 1.10000, 1.10008)
            await asyncio.sleep(0.05)

    ticking = asyncio.create_task(keep_the_market_ticking())
    try:
        outcomes = await probe_module.run_items(probe_for(rig), probe_module.DEFAULT_ITEMS)
    finally:
        ticking.cancel()

    statuses = {outcome.item: outcome.status for outcome in outcomes}
    assert statuses == {
        "pre": "PASS",
        "A": "PASS",
        "1": "PASS",
        "2": "PASS",
        "3": "PASS",
        "5": "PASS",
        "6": "SKIPPED",
        "7": "PASS",
        "8": "PASS",
        "9": "OBSERVATION",
    }
    assert rig.server.positions == {}
    printed = capsys.readouterr().out
    assert "[PASS] 1: market order with relative stop and target" in printed
    assert rig.server.access_token not in printed and rig.server.client_secret not in printed
    assert probe_module.summary(outcomes) == "summary: 8 pass, 0 fail, 1 observation, 1 skipped"


async def test_the_probe_reports_an_item_that_breaks_and_goes_on(rig) -> None:
    rig.server.ignore_relative_sltp = True  # item 1 must notice that no stop was applied

    outcomes = await probe_module.run_items(probe_for(rig), ["1", "6"])

    assert [(o.item, o.status) for o in outcomes] == [("1", "FAIL"), ("6", "SKIPPED")]
    assert rig.server.positions == {}  # the item closed what it opened


async def test_item_4_leaves_a_position_and_item_5_finds_it_again_by_label(rig) -> None:
    probe = probe_for(rig, kill=True)

    [outcome] = await probe_module.run_items(probe, ["4"])
    label = next(iter(rig.server.positions.values())).label

    assert probe.kill_when_done and outcome.status == "OBSERVATION" and label in outcome.detail
    await probe.cleanup()
    assert len(rig.server.positions) == 1  # the kill position survives the cleanup on purpose

    [resumed] = await probe_module.run_items(probe_for(rig, resume_label=label), ["5"])

    assert resumed.status == "PASS" and len(rig.server.orders_received) == 1


def test_the_probe_refuses_anything_but_the_demo_host() -> None:
    probe_module.check_host("demo.ctraderapi.com")
    probe_module.check_host("127.0.0.1")
    for host in ("live.ctraderapi.com", "ctraderapi.com", "demo.ctraderapi.com.evil.example"):
        with pytest.raises(probe_module.ProbeConfigError, match="refusing"):
            probe_module.check_host(host)


def test_missing_credentials_are_named_and_never_quoted(capsys) -> None:
    env = {"CTRADER_CLIENT_ID": "id-value", "CTRADER_ACCOUNT_ID": "not-a-number"}

    assert probe_module.main([], env=env) == 2
    printed = capsys.readouterr().out

    assert "CTRADER_CLIENT_SECRET" in printed and "CTRADER_ACCESS_TOKEN" in printed
    assert "id-value" not in printed
    with pytest.raises(probe_module.ProbeConfigError, match="numeric"):
        probe_module.credentials_from_env({name: "x" for name in probe_module.ENV_NAMES})


def test_the_probe_command_refuses_the_live_host_before_connecting(capsys, monkeypatch) -> None:
    env = {name: "x" for name in probe_module.ENV_NAMES} | {"CTRADER_ACCOUNT_ID": "7"}
    monkeypatch.setattr(probe_module.asyncio, "run", lambda *a, **k: pytest.fail("connected"))

    assert probe_module.main(["--host", "live.ctraderapi.com"], env=env) == 2
    assert "refusing" in capsys.readouterr().out


async def test_the_probe_command_end_to_end_and_a_refused_login_is_a_clean_failure(
    server, capsys
) -> None:
    env = {
        "CTRADER_CLIENT_ID": server.client_id,
        "CTRADER_CLIENT_SECRET": server.client_secret,
        "CTRADER_ACCESS_TOKEN": server.access_token,
        "CTRADER_ACCOUNT_ID": str(server.account_id),
    }
    argv = ["--host", "127.0.0.1", "--port", str(server.port), "--items", "6"]

    ok = await asyncio.to_thread(probe_module.main, argv, env)
    printed = capsys.readouterr().out
    assert ok == 0 and "[SKIPPED] 6:" in printed and "summary: 0 pass, 0 fail" in printed

    refused = await asyncio.to_thread(
        probe_module.main, argv, env | {"CTRADER_ACCESS_TOKEN": "a-token-the-server-refuses"}
    )
    printed = capsys.readouterr().out
    assert refused == 1 and "[FAIL] pre: could not start" in printed
    assert "a-token-the-server-refuses" not in printed

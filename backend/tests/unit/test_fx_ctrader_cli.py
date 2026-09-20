"""The persistent CLI session and the venue built on it, against a fake CLI (no docker, no broker)."""

import json
import sys

import pytest

from app.fx import strategy as fx
from app.fx.runner.ctrader_cli import CliVenue, PtySession, json_of
from app.fx.runner.venue import OrderRejected, VenueUnavailable

FAKE_CLI = r"""
import json, sys
print("Connected as fake.")
sys.stdout.write("\x1b[1;32m> \x1b[0m"); sys.stdout.flush()
for line in sys.stdin:
    command = line.strip()
    if command == "quit":
        break
    if command == "price EURUSD":
        body = {"symbolName": "EURUSD", "bid": 1.1, "ask": 1.10002, "spread": 0.2}
    elif command == "slow":
        import time; time.sleep(5); body = {}
    else:
        body = {"echo": command}
    print(f"\x1b[2m[t] {command}\x1b[0m")
    print(json.dumps(body, indent=2))
    sys.stdout.write("\n\x1b[1;32m> \x1b[0m"); sys.stdout.flush()
"""


@pytest.fixture
def session():
    started = PtySession([sys.executable, "-c", FAKE_CLI], login_timeout=10)
    started.start()
    yield started
    started.close()


def test_a_command_gets_its_answer_and_the_prompt_is_not_part_of_the_data(session) -> None:
    answer = json_of(session.send("price EURUSD", timeout=5))

    assert answer == {"symbolName": "EURUSD", "bid": 1.1, "ask": 1.10002, "spread": 0.2}
    assert json_of(session.send("something else", timeout=5)) == {"echo": "something else"}


def test_a_command_that_never_answers_raises_and_a_dead_session_raises(session) -> None:
    with pytest.raises(VenueUnavailable, match="did not answer"):
        session.send("slow", timeout=0.5)
    session.close()
    with pytest.raises(VenueUnavailable):
        session.send("price EURUSD", timeout=1)


def test_an_answer_without_json_is_a_rejection_carrying_the_cli_words() -> None:
    with pytest.raises(OrderRejected, match="Unknown symbol"):
        json_of("order place-market FOO\nError: Unknown symbol FOO\n> ")


class ScriptedTransport:
    """Answers venue commands from a table; `positions` reflects the orders and closes it has seen."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.open: dict[int, dict] = {}
        self.next_id = 100
        self.lose_order_answer = False

    def send(self, command: str, timeout: float = 30.0) -> str:
        self.log.append(command)
        if command.startswith("account"):
            return json.dumps({"balance": 999.5, "equity": 999.3, "depositAsset": "USD"})
        if command.startswith("price"):
            return json.dumps({"bid": 1.14850, "ask": 1.14856})
        if command == "positions":
            return json.dumps({"positions": list(self.open.values())})
        if command.startswith("order place-market"):
            _, _, symbol, side, units, *levels, _ = command.split()
            self.next_id += 1
            self.open[self.next_id] = {"id": self.next_id, "symbolName": symbol, "tradeSide": side.capitalize(),
                                       "volume": int(units), "entryPrice": 1.14856,
                                       "stopLoss": float(levels[0]) - 0.0001, "takeProfit": float(levels[1]) + 0.0001}
            if self.lose_order_answer:
                self.lose_order_answer = False
                raise VenueUnavailable("the answer was lost")
            return json.dumps({"positionId": self.next_id, "status": "opened"})
        if command.startswith("position modify"):
            _, _, identifier, stop, target, _ = command.split()
            self.open[int(identifier)].update(stopLoss=float(stop), takeProfit=float(target))
            return json.dumps({"positionId": int(identifier), "status": "modified"})
        if command.startswith("position close"):
            identifier = int(command.split()[2])
            self.open.pop(identifier)
            return json.dumps({"positionId": identifier, "status": "closed"})
        if command.startswith("deals"):
            return json.dumps({"deals": [{"positionId": 101, "netProfit": -0.12}]})
        raise AssertionError(command)


async def test_the_venue_reads_account_quote_and_positions() -> None:
    transport = ScriptedTransport()
    venue = CliVenue(transport, 10139135, clock=lambda: 42.0)

    account, quote = await venue.account(), await venue.quote("EURUSD")

    assert (account.balance, account.equity, account.currency) == (999.5, 999.3, "USD")
    assert (quote.bid, quote.ask, quote.time) == (1.14850, 1.14856, 42.0)
    assert await venue.positions() == []


async def test_an_order_opens_a_position_that_the_venue_can_amend_and_close() -> None:
    transport = ScriptedTransport()
    venue = CliVenue(transport, 10139135)

    position = await venue.market_order(symbol="EURUSD", side=fx.LONG, units=1000, stop_price=1.14556,
                                        target_price=1.14906, client_order_id="F-1")
    assert (position.side, position.units, position.client_order_id) == (fx.LONG, 1000, "F-1")
    assert position.stop_price == pytest.approx(1.14546)  # what the CLI made of the relative stop

    amended = await venue.amend_protection(position.id, stop_price=1.14556, target_price=1.14906)
    assert (amended.stop_price, amended.target_price) == (1.14556, 1.14906)

    assert await venue.close(position.id) == -0.12
    assert await venue.positions() == []


async def test_a_retry_of_an_accepted_order_returns_the_same_position_and_a_lost_answer_is_recovered() -> None:
    transport = ScriptedTransport()
    venue = CliVenue(transport, 10139135)
    first = await venue.market_order(symbol="EURUSD", side=fx.SHORT, units=1000, stop_price=1.1490,
                                     target_price=1.1470, client_order_id="A")
    again = await venue.market_order(symbol="EURUSD", side=fx.SHORT, units=1000, stop_price=1.1490,
                                     target_price=1.1470, client_order_id="A")
    assert again.id == first.id and len(transport.open) == 1

    transport.lose_order_answer = True
    recovered = await venue.market_order(symbol="EURUSD", side=fx.LONG, units=1000, stop_price=1.14556,
                                         target_price=1.14906, client_order_id="B")
    assert recovered.side == fx.LONG and len(transport.open) == 2  # the position that the lost answer had opened


async def test_a_rejected_order_carries_the_cli_words() -> None:
    class Rejecting(ScriptedTransport):
        def send(self, command, timeout=30.0):
            if command.startswith("order place-market"):
                return "Error: TRADING_BAD_STOPS: stop too close to the market"
            return super().send(command, timeout)

    venue = CliVenue(Rejecting(), 10139135)

    with pytest.raises(OrderRejected, match="TRADING_BAD_STOPS"):
        await venue.market_order(symbol="EURUSD", side=fx.LONG, units=1000, stop_price=1.1485,
                                 target_price=1.1486, client_order_id="C")

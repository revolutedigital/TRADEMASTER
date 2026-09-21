"""The persistent CLI session and the venue built on it, against a fake CLI (no docker, no broker)."""

import json
import sys
import time
from datetime import UTC, datetime

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


FAKE_LOGIN_CLI = r"""
import os, sys
sys.stdout.write("cTrader CLI\r\nPassword: "); sys.stdout.flush()
if sys.stdin.readline().strip() != os.environ["EXPECTED_PASSWORD"]:
    print("Authentication failed"); sys.exit(1)
sys.stdout.write("> "); sys.stdout.flush()
for line in sys.stdin:
    if line.strip() == "quit":
        break
    print('{"echo": "%s"}' % line.strip())
    sys.stdout.write("\n> "); sys.stdout.flush()
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


def test_the_password_is_typed_at_the_prompt_and_never_given_in_the_arguments(monkeypatch) -> None:
    monkeypatch.setenv("EXPECTED_PASSWORD", "s3cret#")
    argv = [sys.executable, "-c", FAKE_LOGIN_CLI]
    logged_in = PtySession(argv, login_timeout=10, password="s3cret#")
    logged_in.start()
    try:
        assert json_of(logged_in.send("hello", timeout=5)) == {"echo": "hello"}
    finally:
        logged_in.close()
    assert not any("s3cret#" in part for part in argv)


RACY_CLI = r"""
import json, sys, time
sys.stdout.write("> "); sys.stdout.flush()
for line in sys.stdin:
    command = line.strip()
    if command == "quit":
        break
    sys.stdout.write(command + "\r\n> "); sys.stdout.flush()  # the echo and a redrawn prompt arrive first...
    time.sleep(0.25)
    if command.startswith("bad"):
        sys.stdout.write("\r\nError: unknown command\r\n> "); sys.stdout.flush()
    else:
        sys.stdout.write("\r\n" + json.dumps({"answer": command}, indent=2) + "\r\n> "); sys.stdout.flush()
"""


def test_an_answer_that_comes_after_a_redrawn_prompt_is_not_cut_short_or_shifted_to_the_next_command() -> None:
    racy = PtySession([sys.executable, "-c", RACY_CLI], login_timeout=10)
    racy.start()
    try:
        answers = [json_of(racy.send(command, timeout=5)) for command in ("price EURUSD", "positions", "account 1")]
    finally:
        racy.close()

    assert answers == [{"answer": "price EURUSD"}, {"answer": "positions"}, {"answer": "account 1"}]


def test_an_error_text_without_json_comes_back_after_a_short_silence_with_the_cli_words(monkeypatch) -> None:
    monkeypatch.setattr("app.fx.runner.ctrader_cli.QUIET_SECONDS", 0.5)
    racy = PtySession([sys.executable, "-c", RACY_CLI], login_timeout=10)
    racy.start()
    try:
        with pytest.raises(OrderRejected, match="unknown command"):
            json_of(racy.send("bad thing", timeout=5))
    finally:
        racy.close()


def test_a_complete_json_object_is_recognised_with_braces_inside_strings() -> None:
    from app.fx.runner.ctrader_cli import has_complete_json

    assert not has_complete_json("> price\n{\n  \"a\": 1")
    assert has_complete_json("noise {\"a\": {\"b\": \"}{\"}} > ")
    assert not has_complete_json("no json here >")


STUBBORN_CLI = r"""
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)  # like the docker client, which catches SIGTERM
sys.stdout.write("> "); sys.stdout.flush()
while True:
    time.sleep(1)
"""


def test_closing_a_session_that_ignores_sigterm_does_not_hang(monkeypatch) -> None:
    monkeypatch.setattr("app.fx.runner.ctrader_cli.TERMINATE_GRACE_SECONDS", 0.3)
    stubborn = PtySession([sys.executable, "-c", STUBBORN_CLI], login_timeout=10)
    stubborn.start()
    started = time.monotonic()

    stubborn.close()

    assert time.monotonic() - started < 5
    with pytest.raises(VenueUnavailable):
        stubborn.send("price EURUSD", timeout=1)


def test_a_rejected_password_fails_the_start_instead_of_hanging(monkeypatch) -> None:
    monkeypatch.setenv("EXPECTED_PASSWORD", "s3cret#")
    rejected = PtySession([sys.executable, "-c", FAKE_LOGIN_CLI], login_timeout=10, password="wrong")
    try:
        with pytest.raises(VenueUnavailable):
            rejected.start()
    finally:
        rejected.close()


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
            return json.dumps({"deals": [
                {"positionId": 101, "dealType": "Market", "executionPrice": 1.1484, "netProfit": -0.12,
                 "time": "2026-09-20T23:01:55.706Z"},
                {"positionId": 205, "dealType": "Limit", "executionPrice": 1.14775, "netProfit": 0.24,
                 "time": "2026-09-21T00:02:49.456Z"},
                {"positionId": 206, "dealType": "Stop", "executionPrice": 1.14835, "netProfit": -0.36,
                 "time": "2026-09-21T00:20:00.000Z"},
            ]})
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


async def test_the_closing_deal_says_how_a_position_ended_and_an_unknown_position_has_none() -> None:
    venue = CliVenue(ScriptedTransport(), 10139135)

    target = await venue.exit_of("205", "EURUSD")
    assert (target.reason, target.result, target.price) == ("target", 0.24, 1.14775)
    assert target.time == datetime(2026, 9, 21, 0, 2, 49, 456000, tzinfo=UTC).timestamp()
    assert (await venue.exit_of("206", "EURUSD")).reason == "stop"
    assert (await venue.exit_of("101", "EURUSD")).reason == "market"
    assert await venue.exit_of("999", "EURUSD") is None


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

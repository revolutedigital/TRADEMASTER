"""The demo soak: one measured cycle, the scalp exits, the guards that stop it, and its error handling."""

import json
from datetime import UTC, datetime

import pytest

from scripts.research import fx_soak as soak


def block(payload: dict) -> str:
    return "noise\n" + json.dumps(payload, indent=2) + "\n"


class FakeCli:
    """Answers like the CLI. `server_closes_after_polls` closes the position on the server (target or stop)."""

    def __init__(self, fail_orders: bool = False, server_closes_after_polls: int | None = None,
                 exit_price: float = 1.1485) -> None:
        self.calls: list[str] = []
        self.open = False
        self.fail_orders = fail_orders
        self.polls_after_order = 0
        self.server_closes_after_polls = server_closes_after_polls
        self.exit_price = exit_price

    def __call__(self, *commands: str) -> str:
        command = commands[0]
        self.calls.append(command)
        if command.startswith("price"):
            return block({"bid": 1.14850, "ask": 1.14856, "spread": 0.6})
        if command.startswith("order"):
            if self.fail_orders:
                return "Order rejected: MARKET_CLOSED"
            self.open, self.polls_after_order = True, 0
            return block({"positionId": 1, "status": "opened"})
        if command == "positions":
            self.polls_after_order += 1
            if self.open and self.server_closes_after_polls is not None and self.polls_after_order > self.server_closes_after_polls:
                self.open = False
            if not self.open:
                return block({"positions": []})
            return block({"positions": [{"id": 1, "entryPrice": 1.14858, "stopLoss": 1.14706, "stopLossPips": 15.2,
                                         "takeProfit": 1.15156, "takeProfitPips": 29.8, "commission": -0.03}]})
        if command.startswith("position close"):
            self.open = False
            return block({"status": "closed"})
        if command == "deals":
            return block({"deals": [{"positionId": 1, "executionPrice": self.exit_price, "grossProfit": -0.08,
                                     "commission": -0.03, "netProfit": -0.14}]})
        raise AssertionError(command)


def fake_clock():
    now = [1_715_000_000.0]
    return (lambda: now[0]), (lambda seconds: now.__setitem__(0, now[0] + seconds))


def test_one_cycle_measures_spread_slippage_costs_and_that_the_stop_is_on_the_server() -> None:
    clock, sleep = fake_clock()

    record = soak.one_cycle(FakeCli(), "buy", now=clock, sleep=sleep)

    assert record["spread_pips"] == 0.6 and record["entry_slippage_pips"] == pytest.approx(0.2)
    assert record["stop_on_server"] is True and record["net_usd"] == -0.14 and record["outcome"] == "timeout"
    assert record["commission_usd"] == pytest.approx(-0.03)  # the deal total, not summed with the entry side


def test_the_protective_levels_come_from_the_settings_and_a_sell_puts_the_stop_above() -> None:
    clock, sleep = fake_clock()
    orders: list[str] = []
    cli = FakeCli()

    def spy(*commands):
        if commands[0].startswith("order"):
            orders.append(commands[0])
        return cli(*commands)

    soak.one_cycle(spy, "sell", now=clock, sleep=sleep)
    soak.one_cycle(spy, "buy", soak.SCALP, now=clock, sleep=sleep)

    assert " sell 1000 1.15000 1.14550 yes" in orders[0]  # plumbing: stop 15 pips above the bid, target 30 below
    assert " buy 1000 1.14826 1.14871 yes" in orders[1]  # scalp: stop 3 pips below the ask, target 1.5 above


@pytest.mark.parametrize(("exit_price", "outcome"), [(1.14873, "target"), (1.14826, "stop")])
def test_a_scalp_closed_by_the_server_is_a_target_or_a_stop_and_is_never_closed_by_the_bot(exit_price, outcome) -> None:
    clock, sleep = fake_clock()
    cli = FakeCli(server_closes_after_polls=2, exit_price=exit_price)

    record = soak.one_cycle(cli, "buy", soak.SCALP, now=clock, sleep=sleep)

    assert record["outcome"] == outcome
    assert not any(call.startswith("position close") for call in cli.calls)
    assert cli.calls.count("positions") == 3  # the confirmation and two polls


def test_a_scalp_that_hits_neither_level_is_closed_at_the_time_limit() -> None:
    clock, sleep = fake_clock()
    cli = FakeCli()

    record = soak.one_cycle(cli, "sell", soak.SCALP, now=clock, sleep=sleep)

    assert record["outcome"] == "timeout" and "position close all yes" in cli.calls
    assert cli.calls.count("positions") == 1 + int(soak.SCALP.hold_seconds / soak.SCALP.poll_seconds)


def test_the_trading_window_is_the_fx_week_without_the_rollover_hour() -> None:
    window = soak.Window()
    utc = lambda text: datetime.fromisoformat(text).replace(tzinfo=UTC)  # noqa: E731

    assert window.is_open(utc("2024-05-14 14:00"))  # Tuesday 10:00 in New York
    assert not window.is_open(utc("2024-05-14 20:45"))  # Tuesday 16:45: the rollover hour
    assert window.is_open(utc("2024-05-14 21:45"))  # Tuesday 17:45: reopened
    assert window.is_open(utc("2024-05-12 22:00"))  # Sunday 18:00 in New York: the week is open
    assert not window.is_open(utc("2024-05-12 19:00"))  # Sunday 15:00: still closed
    assert not window.is_open(utc("2024-05-17 20:45"))  # Friday 16:45: closed for the week
    assert not window.is_open(utc("2024-05-18 14:00"))  # Saturday


def test_it_refuses_to_trade_with_a_stop_file_a_full_day_or_a_daily_loss(tmp_path) -> None:
    log = tmp_path / "soak.jsonl"
    now = datetime(2024, 5, 14, 14, 0, tzinfo=UTC)
    window = soak.Window()
    assert soak.should_run(now, log, window, stop_file=tmp_path / "STOP") is None

    (tmp_path / "STOP").write_text("")
    assert soak.should_run(now, log, window, stop_file=tmp_path / "STOP") == "stop file present"

    log.write_text("".join(json.dumps({"at": "2024-05-14T10:00:00+00:00", "net_usd": -1.5}) + "\n" for _ in range(7)))
    assert soak.should_run(now, log, window, stop_file=tmp_path / "none") == "daily loss limit reached"
    log.write_text("".join(json.dumps({"at": "2024-05-14T10:00:00+00:00", "net_usd": 0.0}) + "\n" for _ in range(40)))
    assert soak.should_run(now, log, window, stop_file=tmp_path / "none") == "daily trade limit reached"
    assert soak.should_run(now, log, window, soak.SCALP, stop_file=tmp_path / "none") is None  # the scalp allows more


def test_the_loop_logs_each_cycle_alternates_sides_and_starts_flat(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(soak, "should_run", lambda *a, **k: None)
    clock, sleep = fake_clock()
    cli = FakeCli()
    cli.open = True  # a stray position from an earlier run
    log = tmp_path / "soak.jsonl"

    code = soak.soak(cli, log, cycles=2, clock=clock, sleep=sleep)

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert code == 0 and [line["side"] for line in lines] == ["buy", "sell"]
    assert cli.calls[:2] == ["positions", "position close all yes"]


def test_repeated_errors_close_what_is_open_and_stop_the_soak(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(soak, "should_run", lambda *a, **k: None)
    clock, sleep = fake_clock()
    cli = FakeCli(fail_orders=True)
    log = tmp_path / "soak.jsonl"

    code = soak.soak(cli, log, cycles=None, clock=clock, sleep=sleep)

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert code == 1 and len(lines) == soak.MAX_CONSECUTIVE_ERRORS and all("error" in line for line in lines)
    assert cli.open is False

"""The demo soak: one measured cycle, the guards that stop it, and its error handling."""

import json
from datetime import UTC, datetime

import pytest

from scripts.research import fx_soak as soak


def block(payload: dict) -> str:
    return "noise\n" + json.dumps(payload, indent=2) + "\n"


class FakeCli:
    def __init__(self, fail_orders: bool = False) -> None:
        self.calls: list[str] = []
        self.open = False
        self.fail_orders = fail_orders

    def __call__(self, *commands: str) -> str:
        command = commands[0]
        self.calls.append(command)
        if command.startswith("price"):
            return block({"bid": 1.14850, "ask": 1.14856, "spread": 0.6})
        if command.startswith("order"):
            if self.fail_orders:
                return "Order rejected: MARKET_CLOSED"
            self.open = True
            return block({"positionId": 1, "status": "opened"})
        if command == "positions":
            if not self.open:
                return block({"positions": []})
            return block({"positions": [{"entryPrice": 1.14858, "stopLoss": 1.14706, "stopLossPips": 15.2,
                                         "takeProfit": 1.15156, "takeProfitPips": 29.8, "commission": -0.03}]})
        if command.startswith("position close"):
            self.open = False
            return block({"status": "closed"})
        if command == "deals":
            return block({"deals": [{"executionPrice": 1.1485, "grossProfit": -0.08, "commission": -0.03, "netProfit": -0.14}]})
        raise AssertionError(command)


def fake_clock():
    now = [1_715_000_000.0]
    return (lambda: now[0]), (lambda seconds: now.__setitem__(0, now[0] + seconds))


def test_one_cycle_measures_spread_slippage_costs_and_that_the_stop_is_on_the_server() -> None:
    clock, sleep = fake_clock()

    record = soak.one_cycle(FakeCli(), "buy", clock, sleep)

    assert record["spread_pips"] == 0.6 and record["entry_slippage_pips"] == pytest.approx(0.2)
    assert record["stop_on_server"] is True and record["net_usd"] == -0.14
    assert record["commission_usd"] == pytest.approx(-0.06)
    assert soak.PIP * soak.STOP_PIPS == pytest.approx(0.0015)


def test_a_sell_places_the_stop_above_and_measures_slippage_against_the_bid() -> None:
    clock, sleep = fake_clock()
    cli = FakeCli()
    original = cli.__call__

    def spy(*commands):
        if commands[0].startswith("order"):
            cli.orders = commands[0]
        return original(*commands)

    soak.one_cycle(spy, "sell", clock, sleep)

    assert " sell 1000 1.15000 1.14550 yes" in cli.orders  # stop 15 pips above the bid, target 30 below


def test_the_trading_window_is_new_york_business_hours_without_the_rollover() -> None:
    window = soak.Window()
    utc = lambda text: datetime.fromisoformat(text).replace(tzinfo=UTC)  # noqa: E731

    assert window.is_open(utc("2024-05-14 14:00"))  # 10:00 in New York, a Tuesday
    assert not window.is_open(utc("2024-05-14 20:45"))  # 16:45 in New York: too close to the rollover
    assert not window.is_open(utc("2024-05-18 14:00"))  # Saturday
    assert not window.is_open(utc("2024-05-14 03:00"))  # 23:00 the evening before in New York


def test_it_refuses_to_trade_with_a_stop_file_a_full_day_or_a_daily_loss(tmp_path) -> None:
    log = tmp_path / "soak.jsonl"
    now = datetime(2024, 5, 14, 14, 0, tzinfo=UTC)
    window = soak.Window()
    assert soak.should_run(now, log, window, tmp_path / "STOP") is None

    (tmp_path / "STOP").write_text("")
    assert soak.should_run(now, log, window, tmp_path / "STOP") == "stop file present"

    log.write_text("".join(json.dumps({"at": "2024-05-14T10:00:00+00:00", "net_usd": -1.5}) + "\n" for _ in range(7)))
    assert soak.should_run(now, log, window, tmp_path / "none") == "daily loss limit reached"
    log.write_text("".join(json.dumps({"at": "2024-05-14T10:00:00+00:00", "net_usd": 0.0}) + "\n" for _ in range(40)))
    assert soak.should_run(now, log, window, tmp_path / "none") == "daily trade limit reached"


def test_the_loop_logs_each_cycle_alternates_sides_and_starts_flat(tmp_path, monkeypatch) -> None:
    all_day = soak.Window(0, 24 * 60)
    monkeypatch.setattr(soak, "Window", lambda: all_day)
    monkeypatch.setattr(soak, "STOP_FILE", tmp_path / "STOP")
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

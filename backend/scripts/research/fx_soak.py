"""Plumbing soak on the Fusion DEMO account: open a minimum lot with a server-side stop, hold, close, measure.

It has no edge and is not a strategy: it alternates buy and sell of 1,000 EURUSD units on a fixed
schedule so that, over days, the real cost of a round trip (spread, slippage, commission) and the
behaviour of stops and orders can be compared with the simulator. Every cycle is one JSON line in
`data/soak/demo_soak.jsonl`. It goes through `scripts/ctrader_demo.sh`, which is pinned to the demo
account, so it cannot touch the live account.

Safety: it stops when the file `~/.config/trademaster/STOP` exists, when the day's net result on the
demo falls below the loss limit, after a maximum number of trades per day, and after repeated errors
(closing anything left open first). It never trades during the New York rollover or on weekends.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "ctrader_demo.sh"
LOG = Path("data/soak/demo_soak.jsonl")
STOP_FILE = Path.home() / ".config" / "trademaster" / "STOP"
SYMBOL, UNITS, PIP = "EURUSD", 1000, 0.0001
STOP_PIPS, TARGET_PIPS, HOLD_SECONDS = 15, 30, 300
CYCLE_SECONDS = 900
MAX_TRADES_PER_DAY = 40
DAILY_LOSS_LIMIT_USD = 10.0
MAX_CONSECUTIVE_ERRORS = 5

Cli = Callable[..., str]


class SoakError(Exception):
    pass


@dataclass(frozen=True)
class Window:
    """Trading window in New York time: Monday to Friday, 01:00 to 16:30 (the rollover is at 17:00)."""

    start_minute: int = 60
    end_minute: int = 16 * 60 + 30

    def is_open(self, when: datetime) -> bool:
        local = when.astimezone(NEW_YORK)
        return local.weekday() < 5 and self.start_minute <= local.hour * 60 + local.minute < self.end_minute


def run_cli(*commands: str) -> str:
    result = subprocess.run(["bash", str(WRAPPER), "run", *commands], capture_output=True, text=True, timeout=600)  # noqa: S603, S607
    return result.stdout + result.stderr


def last_json(text: str) -> dict:
    blocks = re.findall(r"\{.*?\n\}", text, flags=re.S)
    if not blocks:
        raise SoakError(f"no JSON in the CLI output: {text[-300:]!r}")
    return json.loads(blocks[-1])


def one_cycle(cli: Cli, side: str, now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> dict:
    """One measured round trip; returns the record to log."""
    quote = last_json(cli(f"price {SYMBOL}"))
    ask, bid = quote["ask"], quote["bid"]
    sign = 1 if side == "buy" else -1
    reference = ask if side == "buy" else bid
    stop = round(reference - sign * STOP_PIPS * PIP, 5)
    target = round(reference + sign * TARGET_PIPS * PIP, 5)
    sent_at = now()
    last_json(cli(f"order place-market {SYMBOL} {side} {UNITS} {stop:.5f} {target:.5f} yes"))
    position = last_json(cli("positions"))["positions"][0]
    confirmed_at = now()
    sleep(HOLD_SECONDS)
    still_open = last_json(cli("positions"))["positions"]
    protected = bool(still_open) and still_open[0].get("stopLoss") is not None
    cli("position close all yes")
    deal = last_json(cli("deals"))["deals"][-1]
    entry = position["entryPrice"]
    return {
        "at": datetime.fromtimestamp(sent_at, UTC).isoformat(), "side": side, "units": UNITS,
        "quote_bid": bid, "quote_ask": ask, "spread_pips": round((ask - bid) / PIP, 2),
        "entry_price": entry, "entry_slippage_pips": round(sign * (entry - reference) / PIP, 2),
        "stop_on_server": protected, "stop_pips": position.get("stopLossPips"),
        "target_pips": position.get("takeProfitPips"), "latency_seconds": round(confirmed_at - sent_at, 1),
        "exit_price": deal["executionPrice"], "gross_usd": deal["grossProfit"],
        "commission_usd": deal["commission"], "net_usd": deal["netProfit"],  # the deal carries the whole round trip
    }


def flatten(cli: Cli) -> None:
    if last_json(cli("positions"))["positions"]:
        cli("position close all yes")


def today_net(log: Path, day: str) -> tuple[int, float]:
    trades, net = 0, 0.0
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("at", "").startswith(day) and "net_usd" in record:
                trades, net = trades + 1, net + record["net_usd"]
    return trades, net


def should_run(now: datetime, log: Path, window: Window, stop_file: Path = STOP_FILE) -> str | None:
    """Why the soak must not trade right now, or None when it may."""
    if stop_file.exists():
        return "stop file present"
    if not window.is_open(now):
        return "outside the trading window"
    trades, net = today_net(log, now.astimezone(UTC).strftime("%Y-%m-%d"))
    if trades >= MAX_TRADES_PER_DAY:
        return "daily trade limit reached"
    if net <= -DAILY_LOSS_LIMIT_USD:
        return "daily loss limit reached"
    return None


def soak(cli: Cli = run_cli, log: Path = LOG, cycles: int | None = None, clock: Callable[[], float] = time.time,
         sleep: Callable[[float], None] = time.sleep) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    flatten(cli)
    errors, done, side = 0, 0, "buy"
    while cycles is None or done < cycles:
        reason = should_run(datetime.fromtimestamp(clock(), UTC), log, Window())
        if reason in ("stop file present",):
            sys.stdout.write(f"stopping: {reason}\n")
            return 0
        if reason is not None:
            sleep(60)
            continue
        started = clock()
        try:
            record = one_cycle(cli, side, clock, sleep)
            errors, done, side = 0, done + 1, "sell" if side == "buy" else "buy"
        except (SoakError, subprocess.SubprocessError, OSError, KeyError, IndexError, json.JSONDecodeError) as error:
            errors += 1
            record = {"at": datetime.fromtimestamp(started, UTC).isoformat(), "error": f"{type(error).__name__}: {error}"[:300]}
            flatten_safely(cli)
            if errors >= MAX_CONSECUTIVE_ERRORS:
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                sys.stdout.write("stopping: too many consecutive errors\n")
                return 1
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        sleep(max(0.0, CYCLE_SECONDS - (clock() - started)))
    return 0


def flatten_safely(cli: Cli) -> None:
    try:
        flatten(cli)
    except (SoakError, subprocess.SubprocessError, OSError, KeyError, json.JSONDecodeError):
        pass  # the server-side stop still protects whatever is open


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cycles", type=int, default=None, help="stop after this many trades (default: run until stopped)")
    return soak(cycles=parser.parse_args().cycles)


if __name__ == "__main__":
    raise SystemExit(main())

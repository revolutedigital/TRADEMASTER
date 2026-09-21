"""Live runner on the Fusion DEMO account: one pre-registered strategy, decisions on live bars, real orders.

It uses the same compiled strategy step, bar rules, executor, risk guard and journal as the lab and the
runner core, over a persistent cTrader CLI session (about 0.5 s per order). The quotes are polled a few
times per second, so a bar's high and low reflect the sampled quotes and are slightly narrower than the
tick-built bars of the lab; that is a known difference and part of what the demo measures.

    python -m scripts.research.fx_demo_runner --strategy C1      # run until stopped
    python -m scripts.research.fx_demo_runner --report           # performance so far

Stop it with `touch ~/.config/trademaster/STOP`: it closes what is open and exits. The kill switch also
closes everything if the day's loss reaches 1% of the equity.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.fx import analytics
from app.fx import strategy as fx
from app.fx.instruments import ConversionRates
from app.fx.runner.bot import Bot
from app.fx.runner.ctrader_cli import CliVenue, PtySession, Transport, json_of
from app.fx.runner.executor import Executor
from app.fx.runner.journal import Journal, closed_trades
from app.fx.runner.reconcile import reconcile, record_exits
from app.fx.runner.risk import RiskGuard, RiskLimits
from app.fx.runner.venue import OrderRejected, VenueUnavailable
from scripts.research import fx_fast_lab as lab

SYMBOL, ACCOUNT, CTID = "EURUSD", 10139135, "igorrevolute"
IMAGE = "ghcr.io/spotware/ctrader-console@sha256:285484fad431e0ffa4ca96662e82ea66cead93c97cf0e3c46006e80cab4734ba"
PASSWORD_FILE = Path.home() / ".config" / "trademaster" / "ctid.pwd"
STOP_FILE = Path.home() / ".config" / "trademaster" / "STOP"
DATA = Path("data/soak")
LIMITS = RiskLimits(risk_fraction=0.00035, max_daily_loss_fraction=0.01, max_open_positions=1, max_spread_pips=1.5)
POLL_SECONDS = 0.25
MAX_FAILED_LOGINS = 3
SYNC_SECONDS = 15.0  # how often the broker's own exits (stop, target) are read into the journal
PERIODS = {60: "m1", 300: "m5", 900: "m15", 3600: "h1"}


def container_name(strategy: str) -> str:
    return f"ctrader-demo-{strategy}"


def cli_argv(strategy: str) -> list[str]:
    """The CLI in a container, pinned to the demo account. No password here: `PtySession` types it at the prompt."""
    return ["docker", "run", "-it", "--rm", "--name", container_name(strategy), "--log-opt", "max-size=8m",
            "--log-opt", "max-file=2", IMAGE, f"--ctid={CTID}", f"--account={ACCOUNT}"]


def password() -> str:
    return PASSWORD_FILE.read_text().rstrip("\r\n")


def remove_container(strategy: str) -> None:
    """A docker client killed in tty mode leaves its container running: remove it by name (no error if none)."""
    subprocess.run(  # noqa: S603
        ["docker", "rm", "-f", container_name(strategy)], capture_output=True, timeout=30, check=False  # noqa: S607
    )


def warm_up(bot: Bot, transport: Transport, now: float) -> int:
    """Feed the strategy the completed bars just before now, so it can trade from the first live bar."""
    seconds = bot.builder.seconds
    quote = json_of(transport.send(f"price {SYMBOL}"))
    half_spread = 0.5 * (quote["ask"] - quote["bid"])
    candles = json_of(transport.send(f"candles {SYMBOL} {PERIODS[seconds]} 90", 60.0))["bars"]
    fed = 0
    for candle in candles:
        opened = datetime.fromisoformat(candle["timestamp"].replace("Z", "+00:00")).timestamp()
        if opened + seconds > now:
            continue  # still forming
        row = np.zeros(fx.BAR_WIDTH)
        row[fx.BAR_TIME] = opened
        for column, key in ((fx.BID_OPEN, "open"), (fx.BID_HIGH, "high"), (fx.BID_LOW, "low"), (fx.BID_CLOSE, "close")):
            row[column] = candle[key] - half_spread
            row[column + 4] = candle[key] + half_spread
        bot.strategy.on_bar(row, fx.FLAT)
        fed += 1
    return fed


def build(strategy: str, transport: Transport, data: Path = DATA):
    config = lab.configurations()[strategy]
    venue = CliVenue(transport, ACCOUNT)
    guard = RiskGuard(LIMITS, data / "risk.json")
    journal = Journal(data / f"journal_{strategy}.jsonl")
    executor = Executor(venue, guard, journal, ConversionRates({}))
    bot = Bot(key=strategy, symbol=SYMBOL, seconds=config.seconds, step=config.step, init=config.init,
              state_size=config.state_size, params=config.params(SYMBOL), executor=executor, venue=venue)
    return bot, venue, guard, journal, executor


async def loop(bot: Bot, venue: CliVenue, guard: RiskGuard, executor: Executor, stop_file: Path,
               clock: Callable[[], float] = time.time, iterations: int | None = None,
               sleep: Callable[[float], object] = asyncio.sleep) -> str:
    """Poll quotes and feed the bot until stopped; returns why it stopped.

    A stop or target fills on the broker's server without the bot sending anything, so every few seconds the
    loop reads such exits (with their result) into the journal: that is what the performance report counts.
    """
    count, last_equity, last_sync = 0, 0.0, 0.0
    while iterations is None or count < iterations:
        count += 1
        now = clock()
        if stop_file.exists():
            await executor.flatten_all("stop file")
            return "stop file"
        if guard.killed:
            await executor.flatten_all(f"kill switch: {guard.kill_reason}")
            return f"kill switch: {guard.kill_reason}"
        await bot.on_quote(await venue.quote(SYMBOL))
        await bot.on_clock(now)
        guard.heartbeat(now)
        if now - last_equity > 30:
            guard.observe(now, (await venue.account()).equity)
            last_equity = now
        if now - last_sync > SYNC_SECONDS:
            for position_id in await record_exits(venue, executor.journal, venue.exit_of):
                sys.stdout.write(f"{datetime.now(UTC):%H:%M:%S}Z position {position_id} was closed by the broker\n")
            last_sync = now
        await sleep(POLL_SECONDS)
    return "iterations done"


async def run(strategy: str) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    failed_logins = 0
    while True:
        remove_container(strategy)
        session = PtySession(cli_argv(strategy), password=password())
        try:
            try:
                await asyncio.to_thread(session.start)
            except VenueUnavailable:
                failed_logins += 1
                if failed_logins >= MAX_FAILED_LOGINS:  # a wrong password would otherwise be retried for ever
                    sys.stdout.write(f"{failed_logins} logins failed in a row, giving up\n")
                    return 1
                raise
            failed_logins = 0
            bot, venue, guard, journal, executor = build(strategy, session)
            await asyncio.to_thread(warm_up, bot, session, time.time())
            report = await reconcile(venue, journal, venue.exit_of)
            sys.stdout.write(f"{datetime.now(UTC):%H:%M:%S}Z running {strategy}; reconcile {report}\n")
            reason = await loop(bot, venue, guard, executor, STOP_FILE)
            sys.stdout.write(f"stopping: {reason}\n")
            return 0
        except (VenueUnavailable, OrderRejected) as error:
            sys.stdout.write(f"{datetime.now(UTC):%H:%M:%S}Z session problem, restarting in 10s: {error}\n")
            await asyncio.sleep(10)
        finally:
            session.close()
            remove_container(strategy)


def report(strategy: str) -> None:
    trades = closed_trades(Journal(DATA / f"journal_{strategy}.jsonl").events())
    if not trades:
        sys.stdout.write("no closed trades yet\n")
        return
    result = analytics.performance(trades, start_equity=1000.0)
    sys.stdout.write(
        f"{strategy}: {result.trades} trades, win rate {result.win_rate:.0%}, net US$ {result.total_pnl:.2f}, "
        f"profit factor {result.profit_factor}, avg R {result.avg_r}, max drawdown {result.max_drawdown_pct:.2f}%, "
        f"avg hold {result.avg_hold_seconds:.0f}s\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strategy", choices=["C1", "C2"], default="C1")
    parser.add_argument("--report", action="store_true")
    arguments = parser.parse_args()
    if arguments.report:
        report(arguments.strategy)
        return 0
    if not PASSWORD_FILE.exists():
        sys.stdout.write("password file missing\n")
        return 2
    sys.stdout.reconfigure(line_buffering=True)  # the log is how a long run is followed
    return asyncio.run(run(arguments.strategy))


if __name__ == "__main__":
    raise SystemExit(main())

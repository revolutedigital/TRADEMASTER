"""Gotobi on the demo account (round C14, configuration B, approved on 2003-2011 after discovery on 2012-2026):
on each gotobi day, buy USDJPY at 03:00 JST (a Monday: an hour after the weekly reopening) with a stop one daily ATR
away, sell just before the 09:55 JST fix; sell USDJPY at 09:56 JST and buy it back just before 12:00 JST.

    python -m scripts.research.fx_gotobi_demo        (touch ~/.config/trademaster/STOP to close and stop;
                                                      touch ~/.config/trademaster/RESTART_GOTOBI to stop and keep)

Only the demo account of fx_demo_runner is used; the password is typed at the CLI prompt from its file. The sizing is
the banca's: one daily ATR worth 0.5% of the equity, or the minimum volume. A leg that was missed is never opened
late; an open leg past its exit time is closed at once.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.fx.runner.ctrader_cli import CliVenue, PtySession, json_of
from scripts.research import fx_banca_demo as banca
from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c14_gotobi as c14
from scripts.research import fx_demo_runner as demo

DATA = Path("data/gotobi_demo")
STATE, JOURNAL = DATA / "state.json", DATA / "journal.jsonl"
RESTART = demo.STOP_FILE.with_name("RESTART_GOTOBI")
NAME = "GOTOBI"  # container suffix
SYMBOL = "USDJPY"
TOKYO, NEW_YORK = ZoneInfo("Asia/Tokyo"), ZoneInfo("America/New_York")
LATE_SECONDS = 10 * 60  # an opening more than 10 minutes late is skipped
TICK_SECONDS = 15
ATR_DAYS = 14


@dataclass(frozen=True)
class Leg:
    name: str
    side: int
    open_at: float
    close_at: float


def legs_of(day: pd.Timestamp) -> list[Leg]:
    """The two legs of configuration B for a gotobi day (a Tokyo date), in UTC seconds."""
    tokyo = datetime(day.year, day.month, day.day, tzinfo=TOKYO)
    buy_at = tokyo + timedelta(hours=3)
    if buy_at.astimezone(NEW_YORK).weekday() == 6:  # Sunday in New York: shut until 17:00, open an hour later
        sunday = buy_at.astimezone(NEW_YORK)
        buy_at = datetime(sunday.year, sunday.month, sunday.day, 18, 0, tzinfo=NEW_YORK)
    return [Leg("antes do fix", 1, buy_at.timestamp(), (tokyo + timedelta(hours=9, minutes=54, seconds=45)).timestamp()),
            Leg("reversão", -1, (tokyo + timedelta(hours=9, minutes=56)).timestamp(),
                (tokyo + timedelta(hours=11, minutes=59, seconds=45)).timestamp())]


def upcoming_days(now: float, holidays: set[pd.Timestamp]) -> list[pd.Timestamp]:
    """Gotobi days whose last leg has not ended yet, next first."""
    today = pd.Timestamp(datetime.fromtimestamp(now, TOKYO).date())
    days = c14.gotobi_days(str((today - pd.Timedelta(days=3)).date()), str((today + pd.Timedelta(days=45)).date()),
                           holidays)
    return [d for d in days if legs_of(d)[-1].close_at > now]


class Runner:
    def __init__(self, session: PtySession) -> None:
        self.session = session
        self.venue = CliVenue(session, demo.ACCOUNT)
        self.holidays = c14.japanese_holidays()
        self.state = json.loads(STATE.read_text()) if STATE.exists() else {"legs": {}}

    def log(self, event: str, **fields) -> None:
        with JOURNAL.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": time.time(), "event": event, **fields}, default=float) + "\n")

    def save(self) -> None:
        STATE.write_text(json.dumps(self.state, indent=1, default=float))

    def atr(self, now: float) -> float:
        bars = banca.closed_bars(json_of(self.session.send(f"candles {SYMBOL} d1 60", timeout=60))["bars"], 86400, now)
        return float(c3.atr(bars[:, 1], bars[:, 2], bars[:, 3], bars[:, 4], ATR_DAYS)[-1])

    async def volume(self, atr: float) -> float:
        spec = json_of(self.session.send(f"symbol {SYMBOL}", timeout=60))
        quote = await self.venue.quote(SYMBOL)
        equity = (await self.venue.account()).equity
        per_unit = atr / (0.5 * (quote.bid + quote.ask))  # one daily ATR per unit, in dollars
        volume = banca.position_volume(banca.RISK_PER_ATR * equity, per_unit, float(spec["minVolume"]),
                                       float(spec["volumeStep"]))
        return volume if volume is not None else int(float(spec["minVolume"]))

    async def open_leg(self, key: str, leg: Leg, now: float) -> None:
        atr = self.atr(now)
        quote = await self.venue.quote(SYMBOL)
        price = quote.ask if leg.side == 1 else quote.bid
        stop = round(price - leg.side * atr, 3)
        position = await self.venue.market_order(symbol=SYMBOL, side=leg.side, units=await self.volume(atr),
                                                 stop_price=stop, target_price=None, client_order_id=key)
        if position.stop_price is None:
            await self.venue.close(position.id)
            self.log("refused", key=key, reason="no server-side stop")
            return
        self.state["legs"][key] = {"status": "open", "id": position.id, "side": leg.side, "entry": position.entry_price,
                                   "stop": stop, "atr": atr, "opened_at": now, "close_at": leg.close_at,
                                   "units": position.units}
        self.log("open", key=key, leg=leg.name, side=leg.side, entry=position.entry_price, stop=stop, atr=atr,
                 units=position.units)
        self.save()

    async def close_leg(self, key: str, reason: str) -> None:
        record = self.state["legs"][key]
        open_ids = {p.id for p in await self.venue.positions()}
        if record["id"] in open_ids:
            profit = await self.venue.close(record["id"])
        else:  # the server closed it (the stop)
            closed = await self.venue.exit_of(record["id"], SYMBOL)
            profit, reason = (closed.result if closed else None), "stop no servidor"
        record.update(status="closed", closed_at=time.time(), profit=profit, reason=reason)
        self.log("close", key=key, reason=reason, profit=profit)
        self.save()

    async def tick(self, now: float) -> None:
        for key, record in list(self.state["legs"].items()):
            if record["status"] == "open" and now >= record["close_at"]:
                await self.close_leg(key, "horário")
        for day in upcoming_days(now, self.holidays)[:1]:
            for leg in legs_of(day):
                key = f"{day.date()}|{leg.name}"
                if key in self.state["legs"] or now < leg.open_at:
                    continue
                if now - leg.open_at > LATE_SECONDS or now >= leg.close_at:
                    self.state["legs"][key] = {"status": "skipped", "reason": "horário perdido"}
                    self.log("skipped", key=key)
                    self.save()
                    continue
                await self.open_leg(key, leg, now)

    async def flatten(self, reason: str) -> None:
        for key, record in list(self.state["legs"].items()):
            if record["status"] == "open":
                await self.close_leg(key, reason)


async def loop() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    demo.remove_container(NAME)
    session = demo.cli_session(NAME)
    try:
        await asyncio.to_thread(session.start)
        runner = Runner(session)
        runner.log("start", next_days=[str(d.date()) for d in upcoming_days(time.time(), runner.holidays)[:3]])
        while True:
            if RESTART.exists():
                RESTART.unlink()
                runner.log("restart")
                return 0
            if demo.STOP_FILE.exists():
                await runner.flatten("arquivo STOP")
                runner.log("stop")
                return 0
            try:
                await runner.tick(time.time())
            except Exception as error:  # noqa: BLE001 - log and retry on the next tick; positions keep their stops
                runner.log("error", message=f"{type(error).__name__}: {str(error)[:300]}")
            await asyncio.sleep(TICK_SECONDS)
    finally:
        session.close()
        demo.remove_container(NAME)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(loop()))

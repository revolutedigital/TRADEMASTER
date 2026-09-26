"""C13 front A ("banca de day trade") on the demo account, forward only — approved on the untouched 2003-2011 data
(docs/forex/c13-holdout.md, +0.050 R/trade, t 2.97, 9 of 9 years): the 34 family sets of C10 raise their hand on the
37 banca markets, one fixed management (stop 1 daily ATR, target 3R, until the end of the FX day), and the frozen
model of the discovery (data/c13/banca_model.json) decides which candidate to take.

Live minute data: the CLI only serves ~1,000 M1 bars per plain `candles <symbol> <period> <count>` call, but its
`candles <symbol> <period> <from> <to>` form does accept an explicit date range (checked 2026-09-26; a window over
~960 minutes still gets capped, so this runner asks in chunks under that). Each market keeps its own store
(data/c13/live_m1/<symbol>.npy), seeded from the HistData tail (data/c3/m1) and filled forward to now every cycle.
The exact research functions (fx_c13_banca.trade_rows/add_recent) run on that store every cycle: this is what keeps
every family's recent-result features (`recent_set_market`/`recent_set_all`) identical to what the frozen model was
trained on — computed from every candidate a family raised, taken or not, exactly as in the discovery.

A brand-new candidate (its signal bar just closed, its own exit has not happened yet) cannot appear in trade_rows'
output, which only returns trades whose management has already resolved. It is found separately, from the same
context and family signal used inside trade_rows, as the earliest entry of its (symbol, set) not yet covered by a
resolved trade in the shared pool — the live equivalent of the backtest's one-trade-at-a-time rule.

Declared deviation from the backtest: the store's live tail comes from the CLI's own quotes (composite/mid), while
the HistData history is bid-only; the same kind of live/research quote gap the other two demo runners already
accept.

    python -m scripts.research.fx_c13_demo        (touch ~/.config/trademaster/STOP to flatten and stop;
                                                    touch ~/.config/trademaster/RESTART_C13 to stop and keep)

Only the demo account (10139135) is used, pinned by fx_demo_runner.cli_argv. Gold, silver, WTI and Brent are
excluded from real orders (Igor, 2026-09-25), though they still feed the shared recent-result pool, exactly as in
the discovery. This banca gets its own daily guard: -3% of the day's starting equity closes every C13-A position and
pauses new entries until the next FX day (Igor, 2026-09-26: no objection to the default offered; separate from the
old banca's -US$10/+US$20 guard, which C13-A never shared).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx.runner.ctrader_cli import CliVenue, PtySession, json_of
from app.fx.runner.venue import OrderRejected, VenueUnavailable
from app.fx.sessions import NEW_YORK, ROLLOVER_SECONDS, local_time_of_day
from scripts.research import fx_banca_demo as old_banca
from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c6_lab as c6
from scripts.research import fx_c10_core as core
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_c13_banca as banca
from scripts.research import fx_demo_runner as demo
from scripts.research import fx_fast_lab as lab
from scripts.research.fx_c10_families import FAMILIES

DATA = Path("data/c13")
LIVE_ARRAYS = DATA / "live_m1"
STATE_FILE, JOURNAL = DATA / "demo_state.json", DATA / "demo_journal.jsonl"
STOP_FILE = demo.STOP_FILE
RESTART_FILE = STOP_FILE.with_name("RESTART_C13")
NAME = "C13"

HISTORY_DAYS = 300  # calendar days of live M1 kept: comfortably above the ATR100/SMA50 lookback and the 200-trade pool
CHUNK_SECONDS = 900 * 60  # under the CLI's ~1,000-bar cap per `candles` call (checked: 961 minutes came back whole)
CYCLE_SECONDS = 300  # re-run the research pipeline on the growing store every 5 minutes, for parity with discovery
MAX_LIVE_DELAY_SECONDS = 240  # a pending candidate older than this is logged as missed, not opened at a stale price
DAILY_STOP_FRACTION = 0.03  # Igor, 2026-09-26: this banca's own daily stop; closes and pauses until the next FX day
GUARD_SECONDS = 30


def symbol_universe() -> list[str]:
    manifest = pd.read_csv(c3.MANIFEST)
    included = set(manifest.loc[manifest["symbol_included"], "symbol"])
    return [s for s in c10.SYMBOLS if s in included]


def day_end_after(ts: float) -> float:
    """The next 17:00 New York timestamp strictly after `ts`: the end of `ts`'s FX day (app.fx.sessions.fx_day)."""
    tod = float(local_time_of_day(np.float64(ts), NEW_YORK))
    end = ts - tod + ROLLOVER_SECONDS
    return end if end > ts else end + 86400


# --- the live M1 store, one file per market ---------------------------------------------------------------------


def parse_bars(payload: dict, now: float) -> np.ndarray:
    """(t, o, h, l, c) of the bars in a `candles` reply that had fully closed by `now`."""
    rows = [(pd.Timestamp(bar["timestamp"]).timestamp(), bar["open"], bar["high"], bar["low"], bar["close"])
            for bar in payload.get("bars", [])]
    rows = [row for row in rows if row[0] + 60 <= now]
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def fetch_range(session, fusion: str, start: float, end: float, now: float) -> np.ndarray:
    frm = pd.Timestamp(start, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    to = pd.Timestamp(end, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = json_of(session.send(f"candles {fusion} m1 {frm} {to}", timeout=60))
    return parse_bars(payload, now)


def seed_symbol(session, symbol: str, fusion: str, now: float) -> np.ndarray:
    """The symbol's live M1 store: the HistData tail on first use, always filled forward to `now`, kept and saved
    for the next cycle so a restart does not redo the backfill."""
    path = LIVE_ARRAYS / f"{symbol}.npy"
    cutoff = now - HISTORY_DAYS * 86400
    if path.exists():
        m1 = np.load(path)
    else:
        history = np.load(c3.ARRAYS / f"{symbol}.npy", mmap_mode="r")
        m1 = np.ascontiguousarray(history[history[:, 0] >= cutoff])
    last = float(m1[-1, 0]) if len(m1) else cutoff
    chunks = [m1] if len(m1) else []
    while now - last > 60:
        chunk = fetch_range(session, fusion, last + 60, min(last + CHUNK_SECONDS, now), now)
        if len(chunk):
            chunks.append(chunk)
            last = float(chunk[-1, 0])
        else:
            last = min(last + CHUNK_SECONDS, now)
    merged = np.concatenate(chunks) if chunks else np.empty((0, 5))
    frame = pd.DataFrame(merged, columns=["t", "o", "h", "l", "c"]).drop_duplicates("t", keep="last").sort_values("t")
    merged = frame[frame["t"] >= cutoff].to_numpy(dtype=np.float64)
    LIVE_ARRAYS.mkdir(parents=True, exist_ok=True)
    np.save(path, merged)
    return merged


# --- pending candidates: the live edge trade_rows cannot see yet -------------------------------------------------


def pending_entries(symbol: str, m1: np.ndarray) -> pd.DataFrame:
    """Every raw entry of every family set — informative, not yet gated by the management's one-trade-at-a-time
    rule — with its features at the signal bar. The same computation as fx_c13_banca.trade_rows, stopping short of
    the management simulation (a pending entry's own exit has not happened yet)."""
    cost = c3.cost_of(symbol)
    market_class = banca.CLASS_CODES.index(c6.instrument_class(symbol))
    contexts: dict[str, tuple] = {}
    rows = []
    for set_id, (name, params) in enumerate(c10.family_sets()):
        timeframe = params["timeframe"]
        if timeframe not in contexts:
            ctx = core.build_context(symbol, m1, timeframe)
            contexts[timeframe] = ctx, banca.bar_features(ctx, cost)
        ctx, features = contexts[timeframe]
        side, stop_ref = FAMILIES[name].signals(ctx, params)
        entries, sides, stops, atrs = core.entry_arrays(ctx, side, stop_ref, len(m1))
        if not len(entries):
            continue
        bar = np.searchsorted(ctx.last_minute, entries - 1)
        if not np.array_equal(ctx.last_minute[bar], entries - 1):
            continue  # a data hole at the signal bar: no candidate here (the same guard as trade_rows)
        step = core.TIMEFRAME_MINUTES[timeframe] * 60
        decision = np.ceil((m1[entries - 1, 0] + 60) / step) * step
        trade_side = sides.astype(float)
        frame = pd.DataFrame({
            "symbol": symbol, "set_id": set_id, "side": trade_side, "decision_time": decision,
            "entry_time": m1[entries, 0], "stop_ref": stops, "atr_d1": atrs, "market_class": market_class,
            "timeframe_minutes": core.TIMEFRAME_MINUTES[timeframe],
        } | {column: values[bar] for column, values in features.items()})
        frame[list(banca.DIRECTIONAL)] = frame[list(banca.DIRECTIONAL)].mul(trade_side, axis=0)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def resolved_until(pool: pd.DataFrame) -> dict[tuple[str, int], float]:
    """The latest exit_time already known for each (symbol, set): entries at or before it are not fresh candidates,
    the live equivalent of the backtest's busy_until."""
    if not len(pool):
        return {}
    return {(symbol, int(set_id)): float(exit_time) for (symbol, set_id), exit_time in
            pool.groupby(["symbol", "set_id"])["exit_time"].max().items()}


def recent_feature(pool: pd.DataFrame, keys: dict, window: int, decision_time: float) -> float:
    if not len(pool):
        return float("nan")
    mask = np.ones(len(pool), dtype=bool)
    for key, value in keys.items():
        mask &= pool[key].to_numpy() == value
    subset = pool[mask]
    if not len(subset):
        return float("nan")
    return float(c6.recent_mean(np.array([decision_time]), subset["exit_time"].to_numpy(float),
                                subset["net"].to_numpy(float), window)[0])


def recent_features(pool: pd.DataFrame, symbol: str, set_id: int, decision_time: float) -> dict[str, float]:
    context = {"set_id": set_id, "symbol": symbol}
    return {name: recent_feature(pool, {k: context[k] for k in keys}, window, decision_time)
            for name, (keys, window) in banca.RECENT.items()}


# --- the runner ----------------------------------------------------------------------------------------------------


class LiveBanca:
    def __init__(self, session: PtySession) -> None:
        from xgboost import XGBRegressor

        self.session, self.venue = session, CliVenue(session, demo.ACCOUNT)
        event = banca._events("c13_a_discovery")[-1]
        if not event["passes"]:
            raise RuntimeError("front A did not pass discovery: nothing to run on the demo")
        chosen = event["chosen"]
        self.threshold = float(chosen["threshold"])
        if banca.sha256(banca.MODEL_FILE) != event["model_sha256"]:
            raise RuntimeError("the frozen model on disk is not the one recorded in the registry")
        self.model = XGBRegressor()
        self.model.load_model(str(banca.MODEL_FILE))
        self.model_sha = banca.sha256(banca.MODEL_FILE)
        self.grid, self.g = banca.management_row()
        self.universe = symbol_universe()
        self.state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else \
            {"positions": {}, "decided": {}, "guard": {}}
        self.specs: dict[str, dict] = {}
        self.pool = pd.DataFrame()
        self.equity, self.usd_per = 0.0, {}

    def log(self, event: str, **fields) -> None:
        with JOURNAL.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": time.time(), "event": event, **fields}, default=float, sort_keys=True) + "\n")

    def save(self) -> None:
        STATE_FILE.write_text(json.dumps(self.state, indent=1, default=float))

    def paused(self) -> bool:
        return self.state.get("guard", {}).get("status", "operando") != "operando"

    async def spec(self, fusion: str) -> dict:
        if fusion not in self.specs:
            self.specs[fusion] = json_of(self.session.send(f"symbol {fusion}", timeout=60))
        return self.specs[fusion]

    async def refresh_market(self) -> None:
        self.equity = (await self.venue.account()).equity
        mids = {pair: (lambda q: 0.5 * (q.bid + q.ask))(await self.venue.quote(pair)) for pair in old_banca.USD_PAIRS}
        self.usd_per = old_banca.usd_per_currency(mids)

    async def refresh_store(self, now: float) -> dict[str, np.ndarray]:
        stores = {}
        for symbol in self.universe:
            fusion = old_banca.FUSION.get(symbol, symbol)
            stores[symbol] = await asyncio.to_thread(seed_symbol, self.session, symbol, fusion, now)
        return stores

    def refresh_pool(self, stores: dict[str, np.ndarray]) -> None:
        frames = [f for f in (banca.trade_rows(symbol, m1) for symbol, m1 in stores.items() if len(m1)) if len(f)]
        self.pool = banca.add_recent(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()

    async def decide(self, symbol: str, m1: np.ndarray, now: float) -> None:
        pending = pending_entries(symbol, m1)
        if not len(pending):
            return
        until = resolved_until(self.pool)
        changed = False
        for set_id, rows in pending.groupby("set_id"):
            fresh = rows[rows["entry_time"] > until.get((symbol, int(set_id)), -np.inf)].sort_values("entry_time")
            if not len(fresh):
                continue
            row = fresh.iloc[0]
            key = f"{int(set_id)}|{symbol}"
            if self.state["decided"].get(key) == row["entry_time"]:
                continue
            self.state["decided"][key] = float(row["entry_time"])
            changed = True
            await self._decide_one(key, symbol, int(set_id), row, now)
        if changed:
            self.save()

    async def _decide_one(self, key: str, symbol: str, set_id: int, row: pd.Series, now: float) -> None:
        stale = now - row["entry_time"] > MAX_LIVE_DELAY_SECONDS
        # the same guard as manage(): no entry with less than MIN_MINUTES_LEFT of its FX day still to run
        too_late = day_end_after(row["entry_time"]) - row["entry_time"] < (core.MIN_MINUTES_LEFT - 1) * 60
        features = row[[f for f in banca.FEATURES if f not in banca.RECENT]].to_dict()
        features |= recent_features(self.pool, symbol, set_id, row["decision_time"])
        x = np.array([[features[f] for f in banca.FEATURES]], dtype=np.float32)
        prediction = float(self.model.predict(x)[0])
        take = prediction > self.threshold and not stale and not too_late and symbol not in old_banca.EXCLUDED
        self.log("signal", key=key, symbol=symbol, set_id=set_id, side=int(row["side"]), prediction=prediction,
                 threshold=self.threshold, take=take, stale=stale, too_late=too_late, model_sha=self.model_sha)
        if take:
            await self._open(key, symbol, row)

    async def _open(self, key: str, symbol: str, row: pd.Series) -> None:
        fusion = old_banca.FUSION.get(symbol, symbol)
        side, atr = int(row["side"]), float(row["atr_d1"])
        quote = await self.venue.quote(fusion)
        price = quote.ask if side == 1 else quote.bid
        if self.grid[self.g, core.STOP_ATR] > 0:
            distance = self.grid[self.g, core.STOP_ATR] * atr
        elif not np.isnan(row["stop_ref"]):
            room = side * (price - row["stop_ref"])
            if room <= 0:
                self.log("refused", key=key, reason="a entrada já passou do stop estrutural")
                return
            distance = max(room, core.MIN_STOP_ATR * atr)
        else:
            self.log("refused", key=key, reason="sem stop")
            return
        stop = price - side * distance
        # this management has no first target (TP1_R = 0): with tp1 undefined, `manage()` treats the position as
        # past its first target from the first minute, so the single target it ever checks is the runner one
        target = price + side * self.grid[self.g, core.RUNNER_R] * distance \
            if self.grid[self.g, core.RUNNER_R] > 0 else None
        per_unit = distance * self.usd_per[old_banca.QUOTE.get(fusion, fusion[3:])]
        spec = await self.spec(fusion)
        volume = old_banca.position_volume(banca.RISK_PER_TRADE * self.equity, per_unit, float(spec["minVolume"]),
                                           float(spec["volumeStep"]))
        if volume is None:
            self.log("refused", key=key, reason="risco do volume mínimo maior que 0,5% do patrimônio")
            return
        digits = int(spec["digits"])
        stop, target = round(stop, digits), round(target, digits) if target is not None else None
        if (side == 1 and stop >= price) or (side == -1 and stop <= price):
            self.log("refused", key=key, reason="stop do lado errado do preço")
            return
        position = await self.venue.market_order(symbol=fusion, side=side, units=volume, stop_price=stop,
                                                 target_price=target, client_order_id=f"{key}|{int(time.time())}")
        if position.stop_price is None:
            await self.venue.close(position.id)
            self.log("refused", key=key, reason="sem stop no servidor")
            return
        day_end = day_end_after(row["entry_time"])
        self.state["positions"][key] = {"id": position.id, "symbol": symbol, "fusion": fusion,
                                        "side": side, "entry": position.entry_price, "unit": distance,
                                        "stop": stop, "target": target, "day_end": day_end, "volume": volume}
        self.log("entry", key=key, position_id=position.id, entry=position.entry_price, stop=stop, target=target,
                 volume=volume, day_end=day_end)
        self.save()

    async def enforce_day_end(self, now: float) -> None:
        for key, position in list(self.state["positions"].items()):
            if now >= position["day_end"]:
                await self._close(key, position, "fim do dia")

    async def sync(self) -> None:
        """Positions the server closed on its own (stop, target) are journaled."""
        open_ids = {p.id for p in await self.venue.positions()}
        for key, position in list(self.state["positions"].items()):
            if position["id"] not in open_ids:
                await self._record_exit(key, position, "servidor")

    async def flatten(self, reason: str) -> None:
        for key, position in list(self.state["positions"].items()):
            await self._close(key, position, reason)

    async def _close(self, key: str, position: dict, reason: str) -> None:
        await self.venue.close(position["id"])
        await self._record_exit(key, position, reason)

    async def _record_exit(self, key: str, position: dict, reason: str) -> None:
        closed = await self.venue.exit_of(position["id"], position["fusion"])
        exit_price = closed.price if closed else float("nan")
        net = (position["side"] * (exit_price - position["entry"]) - c3.cost_of(position["symbol"])) / position["unit"]
        self.log("exit", key=key, position_id=position["id"], reason=reason, net=net, exit=exit_price,
                 profit=closed.result if closed else None)
        self.state["positions"].pop(key, None)
        self.save()

    async def guard(self, now: float) -> None:
        """The day's own stop: -3% of the day's starting equity closes every C13-A position until the next FX day."""
        day = int(lab.fx_day_array(np.array([now]))[0])
        guard = self.state.setdefault("guard", {})
        if guard.get("day") != day:
            guard.update(day=day, start_equity=self.equity, status="operando")
            self.log("new_day", day=day, start_equity=self.equity)
            self.save()
            return
        if guard["status"] == "operando" and self.equity - guard["start_equity"] <= -DAILY_STOP_FRACTION * guard["start_equity"]:
            guard["status"] = "limite do dia"
            await self.flatten("limite do dia")
            self.log("day_closed", result=self.equity - guard["start_equity"])
            self.save()


async def loop() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    demo.remove_container(NAME)
    session = PtySession(demo.cli_argv(NAME), password=demo.password())
    try:
        await asyncio.to_thread(session.start)
        runner = LiveBanca(session)
        await runner.refresh_market()
        await runner.sync()
        runner.log("start", model_sha=runner.model_sha, threshold=runner.threshold, universe=len(runner.universe))
        while True:
            now = time.time()
            if RESTART_FILE.exists():
                RESTART_FILE.unlink()
                runner.log("restart")
                return 0
            if STOP_FILE.exists():
                await runner.flatten("arquivo STOP")
                runner.log("stop")
                return 0
            cycle_start = now
            try:
                await runner.refresh_market()
                stores = await runner.refresh_store(now)
                runner.refresh_pool(stores)
                if not runner.paused():
                    for symbol, m1 in stores.items():
                        if len(m1):
                            await runner.decide(symbol, m1, time.time())
                await runner.enforce_day_end(time.time())
                await runner.sync()
                await runner.guard(time.time())
            except (VenueUnavailable, OrderRejected) as error:
                runner.log("error", message=f"{type(error).__name__}: {str(error)[:300]}")
            elapsed = time.time() - cycle_start
            await asyncio.sleep(max(5.0, CYCLE_SECONDS - elapsed))
    finally:
        session.close()
        demo.remove_container(NAME)


def report() -> int:
    events = [json.loads(line) for line in JOURNAL.read_text().splitlines()] if JOURNAL.exists() else []
    signals = [e for e in events if e["event"] == "signal"]
    exits = [e for e in events if e["event"] == "exit"]
    taken = sum(e["take"] for e in signals)
    net = np.array([e["net"] for e in exits if np.isfinite(e["net"])])
    money = sum(e["profit"] for e in exits if e.get("profit") is not None)
    sys.stdout.write(f"sinais: {len(signals)}, aceitos pela banca: {taken}, operações fechadas: {len(exits)}, "
                     f"média {net.mean() if net.size else float('nan'):+.3f} R, resultado na demo US$ {money:+.2f}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:] or ["run"])[0]
    if command == "report":
        return report()
    return asyncio.run(loop())


if __name__ == "__main__":
    raise SystemExit(main())

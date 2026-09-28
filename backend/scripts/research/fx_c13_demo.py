"""C13 front A ("banca de day trade") on the demo account, forward only — approved on the untouched 2003-2011 data
(docs/forex/c13-holdout.md, +0.050 R/trade, t 2.97, 9 of 9 years): the 34 family sets of C10 raise their hand on the
37 banca markets, one fixed management (stop 1 daily ATR, target 3R, until the end of the FX day), and the frozen
model of the discovery (data/c13/banca_model.json) decides which candidate to take.

Live minute data: the CLI only serves ~1,000 M1 bars per plain `candles <symbol> <period> <count>` call, but its
`candles <symbol> <period> <from> <to>` form does accept an explicit date range (checked 2026-09-26; a window over
~960 minutes still gets capped, so this runner asks in chunks under that). Each market keeps a base store
(data/c13/live_m1/<symbol>.npy, ~3 years, seeded from the HistData arrays of data/c3/m1 and rewritten only at start)
and a small tail of the minutes added since (<symbol>.tail.npy), so a cycle never rewrites years of minutes.

Parity with the research, measured on August 2026 against the discovery's own rows: every feature at the signal bar
is computed by the research code itself on the store (a 300-day store left `atr_ratio`, whose ATR(100) needs years to
forget its start, 1-4% off; three years close that). The recent-result features (`recent_set_market`,
`recent_set_all`) read a pool made of the discovery's own candidates up to 2026-08 and, from 2026-09 on, of the
candidates fx_c13_banca.trade_rows resolves on the store every cycle — every candidate a family raised, taken or not,
as in the discovery; a store-only pool missed the oldest trades of the sets that fire rarely. Measured this way (pool
switching on 2026-08-01): 18,110 of the 18,111 August entries found, `atr_ratio` within 2e-4, both recent-result
features within 0.004, and `boll_z` beyond 0.01 on 1.6% of the entries (median 5e-8: pandas' rolling standard deviation
carries rounding that depends on where the series starts).

A brand-new candidate (its signal bar just closed, its own exit has not happened yet) cannot appear in trade_rows'
output, which only returns trades whose management has resolved. It is found from the same context and family signal
trade_rows uses, as the earliest entry of its (market, set) after the last resolved trade of the pool that the
management would itself accept (a next minute within 5 minutes of the signal bar on the same FX day, at least
MIN_MINUTES_LEFT before the day ends) — the live equivalent of the backtest's one-trade-at-a-time rule.

Money rules (docs/forex/c13-a-sweep.md, c13-a-sizing.md and -2.md, exploratory, chosen on 2016-2022 and read on
2023-2026, as corrected on 2026-09-27): each stop is worth 1% of equity and the stops of open positions add up to at
most 5% of it. At US$ 1,000 the 0.5% of the research skipped ~77% of the model's picks, because one minimum FX volume
(1,000 units) is worth more than US$ 5 per daily ATR; without a cap on the open stops, a burst of simultaneous picks
put several times the balance at risk.

Declared deviations from the backtest: the store's live tail comes from the CLI's own quotes (composite/mid) while
the HistData history is bid-only; the entry is a market order when the cycle finds the candidate. The cycle runs
seconds after each 5-minute mark, with the minute now running standing in for the entry minute (every feature reads
the signal bar and earlier; tested against the real next minute), so an order goes out about a minute after the
backtest's next-minute entry (a late entry of 1-5 minutes kept 86-94% of the mean R, measured on 2016-2026). A
candidate found more than MAX_LIVE_DELAY_SECONDS late is logged and never opened.

    python -m scripts.research.fx_c13_demo        (touch ~/.config/trademaster/STOP to flatten and stop;
                                                    touch ~/.config/trademaster/RESTART_C13 to stop and keep)

Only the demo account (10139135) is used, pinned by fx_demo_runner.cli_argv. Gold, silver, WTI and Brent are
excluded from real orders (Igor, 2026-09-25); none of them is in this universe. Its own daily guard, on the FX day's
first equity: at -5% no new entry until the next FX day, open positions running on (the brake the study measured:
closing trades with a positive expectancy early is itself a cost); at -10% everything is closed too (insurance).
It replaced a -3% close-everything default (Igor, 2026-09-26) on 2026-09-27, when Igor asked to improve the banca.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

HISTORY_DAYS = 1100  # about three years of minutes: ATR(100) of the FX days forgets its start, as in the research
TRADE_ROWS_DAYS = 200  # the store's own resolved trades are only used from POOL_SWITCH on
POOL_SWITCH = banca.SAMPLE_END  # before it, the pool of past candidates is the discovery's own
PENDING_DAYS = 2  # a candidate older than this has resolved (day trades end with their FX day) or is not a trade
CHUNK_SECONDS = 900 * 60  # under the CLI's ~1,000-bar cap per `candles` call (checked: 961 minutes came back whole)
TICK_SECONDS = 15  # stop files and the end-of-day closes
CYCLE_PERIOD, CYCLE_OFFSET = 300, 8  # the pipeline runs just after each 5-minute mark: every family's bars end on
# that grid (whole steps of the New York clock, whole hours from UTC); the minute now running stands in for the entry
STAND_IN_REACH = 300  # a market whose last closed minute is older than the bar that just closed gets no stand-in
FRESH_WAIT_SECONDS = 60  # how long a cycle waits for the CLI to serve the minute that closed the bar
DAY_END_MARGIN = 300  # closed 5 minutes before the 17:00 NY rollover: its swap, and the spreads that already
# widen in the last minutes (the bid of the research data sags from about 16:58)
WORKERS = 8
MAX_LIVE_DELAY_SECONDS = 240  # a pending candidate older than this is logged as missed, not opened at a stale price
# Money rules, chosen on 2016-2022 only and read on 2023-2026 (docs/forex/c13-a-sizing.md and -2.md, exploratory, as
# corrected on 2026-09-27): every stop worth 1% of equity, the open stops adding up to at most 5% of it. Risk growing
# with the model's confidence, a 7.5% cap, a cap per currency and giving the room to the most confident first looked
# better in a first version of the study that sized trades on the starting US$ 1,000 but measured the caps on the
# growing balance; measured consistently, none of them beat these.
# Igor, 2026-09-28: entering below the frozen 0.05 threshold did not hold up in the honest 2023-2026 read (worse
# first year, worse under stress, out of the top 12 without reinvesting) — trying it live anyway, one entry threshold
# below the discovery's own. Then raised well past anything sized in the study (which topped out at 2% flat, and
# that alone had shown a -29% first year and a 51% drawdown on 2016-2022, before this threshold change): 2% in the
# weaker band, 5% above 0.05. The open-risk cap has to move with it — at the old 5%, a single top-tier stop would
# have used the whole cap, so at most one position could ever be open, killing the diversification the edge counts
# on; raised to 20% (four full-size positions, or more smaller ones) to keep that shape. The daily brake (-5%, no
# new entries) and stop (-10%, flatten) are unchanged and now bind much sooner: one full-size loss alone reaches the
# brake. Untested combination, on the demo, by Igor's call; watch and revert with RESTART_C13 if it drags.
RISK_BY_PREDICTION = ((0.02, 0.02), (0.05, 0.05))
MAX_OPEN_RISK = 0.20
MAX_ASSET_RISK: float | None = None  # a cap on one currency in one direction (off)
PRIORITY_BY_PREDICTION = False  # give the room under the caps to the most confident candidates first (off)
# Igor, 2026-09-28: raised with the risk per trade (was 0.05/0.10, sized for a 1% trade) so one full-size loss does
# not brake the day on its own; kept the same 2x ratio and the same "how many losing trades in a row" shape as
# before (three at the top tier to brake, six to flatten) rather than scaling blindly. Still demo money, still an
# untested combination — a truly bad day can now go to -30% before the stop flattens everything.
DAILY_BRAKE = 0.15  # at -15% of the FX day's first equity: no new entry until the next day (open ones run on)
DAILY_STOP_FRACTION = 0.30  # at -30%: everything is closed until the next day (insurance; never reached in the study)
# The research charged one fixed round-trip cost per market at every hour, yet 15% of the accepted entries fall in
# 17:00-17:30 New York, right after the rollover, when real spreads widen many times over (and the weekly open is
# worse). An entry whose spread alone is above this many times the whole modelled round trip is refused: there the
# backtest's edge does not hold. Every entry and refusal logs the spread, to learn the real one by market and hour.
SPREAD_LIMIT = 2.0
# The research's minutes are bid prices. Right after the 17:00 New York rollover the spread widens and the bid sags
# with it, back only by about 18:30 (on the crosses, a sag of 5-29% of the daily ATR at 17:15, larger every year since
# 2023): the backtest bought that sagging bid, where a real buyer pays the widened ask. Its longs of 17:15-17:30 made
# +0.27-0.33 R in 2023-2026 against ~0 for its shorts there; without the longs of this window the edge is the same in
# both periods (+0.059 R per trade, t 3.5, in 2016-2022; +0.067 R, t 2.2, in 2023-2026). No long enters in it.
ROLLOVER_LONG_BLOCK = (17 * 60, 18 * 60 + 30)  # New York minutes of the day, start included, end excluded


def rollover_long(side: int, ts: float) -> bool:
    """A long entering at `ts` inside ROLLOVER_LONG_BLOCK (the New York clock)."""
    minute = float(local_time_of_day(np.float64(ts), NEW_YORK)) // 60
    return side == 1 and ROLLOVER_LONG_BLOCK[0] <= minute < ROLLOVER_LONG_BLOCK[1]


def symbol_universe() -> list[str]:
    manifest = pd.read_csv(c3.MANIFEST)
    included = set(manifest.loc[manifest["symbol_included"], "symbol"])
    return [s for s in c10.SYMBOLS if s in included]


def day_end_after(ts: float) -> float:
    """The next 17:00 New York timestamp strictly after `ts`: the end of `ts`'s FX day (app.fx.sessions.fx_day)."""
    tod = float(local_time_of_day(np.float64(ts), NEW_YORK))
    end = ts - tod + ROLLOVER_SECONDS
    return end if end > ts else end + 86400


# --- the live M1 store: a base rewritten at start, a small tail per cycle -------------------------------------------


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


def fetch_since(session, fusion: str, last: float, now: float) -> np.ndarray:
    """Every closed minute after `last`, in chunks the CLI serves whole."""
    chunks = []
    while now - last > 60:
        chunk = fetch_range(session, fusion, last + 60, min(last + CHUNK_SECONDS, now), now)
        if len(chunk):
            chunks.append(chunk)
            last = float(chunk[-1, 0])
        else:
            last = min(last + CHUNK_SECONDS, now)
    return np.concatenate(chunks) if chunks else np.empty((0, 5))


def merged(*parts: np.ndarray, since: float = -np.inf) -> np.ndarray:
    frame = pd.DataFrame(np.concatenate(parts), columns=["t", "o", "h", "l", "c"])
    frame = frame.drop_duplicates("t", keep="last").sort_values("t")
    return np.ascontiguousarray(frame[frame["t"] >= since].to_numpy(dtype=np.float64))


def load_store(symbol: str, mmap: bool = False) -> np.ndarray:
    base = np.load(LIVE_ARRAYS / f"{symbol}.npy", mmap_mode="r" if mmap else None)
    tail_path = LIVE_ARRAYS / f"{symbol}.tail.npy"
    return np.concatenate([base, np.load(tail_path)]) if tail_path.exists() else np.asarray(base)


def seed_symbol(session, symbol: str, fusion: str, now: float) -> float:
    """Rewrite the market's base store at start: the HistData history for the years it lacks, what it already had,
    and every minute since from the CLI. Returns the last minute stored."""
    path = LIVE_ARRAYS / f"{symbol}.npy"
    cutoff = now - HISTORY_DAYS * 86400
    history = np.load(c3.ARRAYS / f"{symbol}.npy", mmap_mode="r")
    parts = [np.asarray(history[history[:, 0] >= cutoff])]
    if path.exists():
        parts.append(load_store(symbol))
    base = merged(*parts, since=cutoff)
    base = merged(base, fetch_since(session, fusion, float(base[-1, 0]) if len(base) else cutoff, now), since=cutoff)
    LIVE_ARRAYS.mkdir(parents=True, exist_ok=True)
    np.save(path, base)
    (LIVE_ARRAYS / f"{symbol}.tail.npy").unlink(missing_ok=True)
    return float(base[-1, 0])


# --- pending candidates: the live edge trade_rows cannot see yet ----------------------------------------------------


def pending_entries(symbol: str, m1: np.ndarray, since: float = -np.inf) -> pd.DataFrame:
    """Every raw entry of every family set from `since` on — not yet gated by the one-trade-at-a-time rule — with its
    features at the signal bar and whether the management would accept that entry minute (`valid`). The same
    computation as fx_c13_banca.trade_rows, stopping short of the management simulation."""
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
        recent = m1[entries, 0] >= since
        entries, sides, stops, atrs = entries[recent], sides[recent], stops[recent], atrs[recent]
        if not len(entries):
            continue
        bar = np.searchsorted(ctx.last_minute, entries - 1)
        if not np.array_equal(ctx.last_minute[bar], entries - 1):
            continue  # a data hole at the signal bar: no candidate here (the same guard as trade_rows)
        step = core.TIMEFRAME_MINUTES[timeframe] * 60
        entry_time, signal_end = m1[entries, 0], m1[entries - 1, 0]
        valid = (entry_time - signal_end <= core.MAX_ENTRY_DELAY_SECONDS) & \
            (core.minute_keys(entry_time, 1440) == core.minute_keys(signal_end, 1440))
        trade_side = sides.astype(float)
        frame = pd.DataFrame({
            "symbol": symbol, "set_id": set_id, "side": trade_side,
            "decision_time": np.ceil((signal_end + 60) / step) * step, "entry_time": entry_time, "valid": valid,
            "stop_ref": stops, "atr_d1": atrs, "market_class": market_class,
            "timeframe_minutes": core.TIMEFRAME_MINUTES[timeframe],
        } | {column: values[bar] for column, values in features.items()})
        frame[list(banca.DIRECTIONAL)] = frame[list(banca.DIRECTIONAL)].mul(trade_side, axis=0)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def next_cycle(now: float) -> float:
    """The next moment CYCLE_OFFSET seconds past a CYCLE_PERIOD mark, strictly after `now`."""
    at = now - now % CYCLE_PERIOD + CYCLE_OFFSET
    return at if at > now else at + CYCLE_PERIOD


def with_stand_in(m1: np.ndarray, at: float) -> np.ndarray:
    """The store plus a stand-in for the minute starting `at` (the one now running), when the market's last closed
    minute belongs to the bar that just closed: that bar's candidates then show their entry a minute sooner. Every
    feature reads the signal bar and earlier only, so the stand-in's prices are never read; its time is the entry's
    time, as the next minute's would be."""
    last = float(m1[-1, 0])
    if not at - STAND_IN_REACH <= last < at:
        return m1
    close = m1[-1, 4]
    return np.vstack([m1, [at, close, close, close, close]])


def leaves_room(entry_time: float) -> bool:
    """manage()'s own rule: no entry with less than MIN_MINUTES_LEFT of its FX day still to run."""
    return day_end_after(entry_time) - entry_time >= (core.MIN_MINUTES_LEFT - 1) * 60


def resolved_until(pool: pd.DataFrame) -> dict[tuple[str, int], float]:
    """The latest exit_time already known for each (symbol, set): entries at or before it are not fresh candidates,
    the live equivalent of the backtest's busy_until."""
    if not len(pool):
        return {}
    return {(symbol, int(set_id)): float(exit_time) for (symbol, set_id), exit_time in
            pool.groupby(["symbol", "set_id"])["exit_time"].max().items()}


def recent_feature(pool: pd.DataFrame, keys: dict, window: int, decision_time: float) -> float:
    subset = pool
    for key, value in keys.items():  # the set first: it leaves a thirty-fourth of the pool for the market's strings
        if not len(subset):
            break
        subset = subset[subset[key].to_numpy() == value]
    if not len(subset):
        return float("nan")
    return float(c6.recent_mean(np.array([decision_time]), subset["exit_time"].to_numpy(float),
                                subset["net"].to_numpy(float), window)[0])


def recent_features(pool: pd.DataFrame, symbol: str, set_id: int, decision_time: float) -> dict[str, float]:
    context = {"set_id": set_id, "symbol": symbol}
    return {name: recent_feature(pool, {k: context[k] for k in keys}, window, decision_time)
            for name, (keys, window) in banca.RECENT.items()}


POOL_COLUMNS = ["symbol", "set_id", "entry_time", "exit_time", "net"]


def history_pool() -> pd.DataFrame:
    """The discovery's own candidates before POOL_SWITCH: the past the recent-result features read, as in research."""
    frame = pd.read_parquet(banca.DATASET, columns=POOL_COLUMNS)
    return frame[frame["entry_time"] < POOL_SWITCH].reset_index(drop=True)


def market_cycle(job: tuple[str, np.ndarray, float, float | None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One market's share of a cycle (runs in a worker): the trades resolved on the store since POOL_SWITCH, and the
    raw candidates of the last PENDING_DAYS."""
    symbol, tail, now, stand_in_at = job
    base = np.load(LIVE_ARRAYS / f"{symbol}.npy", mmap_mode="r")
    m1 = np.concatenate([base, tail]) if len(tail) else np.asarray(base)
    recent = np.ascontiguousarray(m1[m1[:, 0] >= now - TRADE_ROWS_DAYS * 86400])
    resolved = banca.trade_rows(symbol, recent)
    resolved = resolved.loc[resolved["entry_time"] >= POOL_SWITCH, POOL_COLUMNS] if len(resolved) else \
        pd.DataFrame(columns=POOL_COLUMNS)
    live_edge = with_stand_in(m1, stand_in_at) if stand_in_at is not None else m1
    return resolved, pending_entries(symbol, live_edge, since=now - PENDING_DAYS * 86400)


def spread_ok(spread: float, modelled_cost: float) -> bool:
    """Whether the live spread leaves the backtest's cost assumption standing (see SPREAD_LIMIT)."""
    return spread <= SPREAD_LIMIT * modelled_cost


def open_risk(positions: dict) -> float:
    return float(sum(p.get("risk_usd", 0.0) for p in positions.values()))


def risk_for(prediction: float) -> float:
    """The share of equity a stop is worth at this prediction (0 below the threshold)."""
    share = 0.0
    for floor, risk in RISK_BY_PREDICTION:
        if prediction > floor:
            share = risk
    return share


def assets_of(symbol: str) -> tuple[str, str | None]:
    """What a long position in `symbol` is long and short of: a pair's two currencies, the dollar for the dollar
    index, an index itself."""
    fusion = old_banca.FUSION.get(symbol, symbol)
    if fusion == symbol:
        return symbol[:3], symbol[3:]
    return ("USD", None) if fusion == "USDX" else (fusion, None)


def asset_exposure(positions: dict) -> dict[str, float]:
    """The stop money open on each asset, signed by direction."""
    exposure: dict[str, float] = {}
    for position in positions.values():
        long, short = assets_of(position["symbol"])
        signed = position["side"] * position.get("risk_usd", 0.0)
        exposure[long] = exposure.get(long, 0.0) + signed
        if short:
            exposure[short] = exposure.get(short, 0.0) - signed
    return exposure


def asset_room(positions: dict, symbol: str, side: int, risk_usd: float, limit_usd: float) -> bool:
    exposure = asset_exposure(positions)
    long, short = assets_of(symbol)
    fits = abs(exposure.get(long, 0.0) + side * risk_usd) <= limit_usd
    return fits and (short is None or abs(exposure.get(short, 0.0) - side * risk_usd) <= limit_usd)


# --- the runner ----------------------------------------------------------------------------------------------------


class LiveBanca:
    def __init__(self, session: PtySession, pool: ProcessPoolExecutor) -> None:
        from xgboost import XGBRegressor

        self.session, self.venue, self.workers = session, CliVenue(session, demo.ACCOUNT), pool
        event = banca._events("c13_a_discovery")[-1]
        if not event["passes"]:
            raise RuntimeError("front A did not pass discovery: nothing to run on the demo")
        self.threshold = RISK_BY_PREDICTION[0][0]  # the live entry floor: may run looser than the frozen discovery
        # one above (docs/forex/c13-a-sizing.md), a deliberate, tracked choice rather than the pre-registered result
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
        self.history = history_pool()
        self.pool = self.history
        self.tails: dict[str, np.ndarray] = {}
        self.last: dict[str, float] = {}
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

    def seed(self, now: float) -> None:
        for symbol in self.universe:
            self.last[symbol] = seed_symbol(self.session, symbol, old_banca.FUSION.get(symbol, symbol), now)
            self.tails[symbol] = np.empty((0, 5))

    def extend_tails(self, now: float) -> None:
        for symbol in self.universe:
            new = fetch_since(self.session, old_banca.FUSION.get(symbol, symbol), self.last[symbol], now)
            if len(new):
                self.tails[symbol] = merged(self.tails[symbol], new)
                self.last[symbol] = float(self.tails[symbol][-1, 0])
                np.save(LIVE_ARRAYS / f"{symbol}.tail.npy", self.tails[symbol])

    async def refresh_market(self) -> bool:
        """The equity always; the dollar value of each quote currency when the market quotes. False when it does not
        (a weekend: the CLI answers "No live quote"), and nothing can be opened then."""
        self.equity = (await self.venue.account()).equity
        try:
            mids = {pair: (lambda q: 0.5 * (q.bid + q.ask))(await self.venue.quote(pair))
                    for pair in old_banca.USD_PAIRS}
        except OrderRejected:
            return False
        self.usd_per = old_banca.usd_per_currency(mids)
        return True

    async def cycle_markets(self, now: float) -> pd.DataFrame:
        """Every market in parallel: the pool of resolved candidates, and the raw candidates of the last days."""
        loop = asyncio.get_running_loop()
        stand_in_at = now - now % 60  # the minute now running
        jobs = [loop.run_in_executor(self.workers, market_cycle, (s, self.tails[s], now, stand_in_at))
                for s in self.universe]
        results = await asyncio.gather(*jobs)
        resolved = [r for r, _ in results if len(r)]
        self.pool = pd.concat([self.history, *resolved], ignore_index=True) if resolved else self.history
        pending = [p for _, p in results if len(p)]
        return pd.concat(pending, ignore_index=True) if pending else pd.DataFrame()

    async def decide(self, pending: pd.DataFrame, now: float) -> None:
        """Every fresh candidate is scored first; the ones to take then get the room under the caps as they come,
        or by the model's prediction, highest first, with PRIORITY_BY_PREDICTION."""
        if not len(pending):
            return
        until = resolved_until(self.pool)
        pending = pending[pending["valid"]]
        pending = pending[[leaves_room(t) for t in pending["entry_time"]]]
        chosen = []
        for (symbol, set_id), rows in pending.groupby(["symbol", "set_id"]):
            fresh = rows[rows["entry_time"] > until.get((symbol, int(set_id)), -np.inf)].sort_values("entry_time")
            if not len(fresh):
                continue
            row = fresh.iloc[0]
            key = f"{int(set_id)}|{symbol}"
            if self.state["decided"].get(key) == row["decision_time"]:
                continue
            self.state["decided"][key] = float(row["decision_time"])
            prediction = self._score(key, symbol, int(set_id), row, now)
            if prediction is not None:
                chosen.append((prediction, key, symbol, row))
        if PRIORITY_BY_PREDICTION:
            chosen.sort(key=lambda item: -item[0])
        for prediction, key, symbol, row in chosen:
            await self._open(key, symbol, row, risk_for(prediction))
        if len(pending):
            self.save()

    def _score(self, key: str, symbol: str, set_id: int, row: pd.Series, now: float) -> float | None:
        """The model's prediction for a fresh candidate, logged; None unless it is one to take."""
        stale = now - row["entry_time"] > MAX_LIVE_DELAY_SECONDS
        features = row[[f for f in banca.FEATURES if f not in banca.RECENT]].to_dict()
        features |= recent_features(self.pool, symbol, set_id, row["decision_time"])
        x = np.array([[features[f] for f in banca.FEATURES]], dtype=np.float32)
        prediction = float(self.model.predict(x)[0])
        rollover = rollover_long(int(row["side"]), float(row["entry_time"]))
        take = prediction > self.threshold and not stale and not rollover and symbol not in old_banca.EXCLUDED
        self.log("signal", key=key, symbol=symbol, set_id=set_id, side=int(row["side"]), prediction=prediction,
                 threshold=self.threshold, take=take, stale=stale, rollover_long=rollover,
                 delay=now - row["entry_time"], model_sha=self.model_sha)
        return prediction if take else None

    async def _open(self, key: str, symbol: str, row: pd.Series, risk: float) -> None:
        if key in self.state["positions"]:
            self.log("refused", key=key, reason="já tem posição aberta nesse conjunto e mercado")
            return
        fusion = old_banca.FUSION.get(symbol, symbol)
        side, atr = int(row["side"]), float(row["atr_d1"])
        quote = await self.venue.quote(fusion)
        price = quote.ask if side == 1 else quote.bid
        spread, modelled = quote.ask - quote.bid, c3.cost_of(symbol)
        if not spread_ok(spread, modelled):
            self.log("refused", key=key, reason="spread alto", spread=spread, modelled_cost=modelled)
            return
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
        volume = old_banca.position_volume(risk * self.equity, per_unit, float(spec["minVolume"]),
                                           float(spec["volumeStep"]))
        if volume is None:
            self.log("refused", key=key, reason=f"o volume mínimo arrisca mais que {risk:.0%} do patrimônio")
            return
        risk_usd = float(volume) * per_unit
        if open_risk(self.state["positions"]) + risk_usd > MAX_OPEN_RISK * self.equity:
            self.log("refused", key=key, reason=f"os stops abertos passariam de {MAX_OPEN_RISK:.1%} do patrimônio")
            return
        if MAX_ASSET_RISK is not None and \
                not asset_room(self.state["positions"], symbol, side, risk_usd, MAX_ASSET_RISK * self.equity):
            self.log("refused", key=key, reason=f"passaria de {MAX_ASSET_RISK:.0%} numa moeda na mesma direção")
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
                                        "stop": stop, "target": target, "day_end": day_end, "volume": volume,
                                        "risk_usd": risk_usd}
        self.log("entry", key=key, position_id=position.id, entry=position.entry_price, stop=stop, target=target,
                 volume=volume, risk_usd=risk_usd, day_end=day_end, spread=spread, modelled_cost=modelled)
        self.save()

    async def enforce_day_end(self, now: float) -> None:
        for key, position in list(self.state["positions"].items()):
            if now >= position["day_end"] - DAY_END_MARGIN:
                await self._close(key, position, "fim do dia")

    async def run_cycle(self, start: float, mark: float) -> bool:
        """One pass of the pipeline; False, without deciding anything, while the CLI has not yet served the minute
        that closed the bars of `mark` (tried again on the next tick, for up to FRESH_WAIT_SECONDS)."""
        await asyncio.to_thread(self.extend_tails, start)
        if max(self.last.values()) < mark - 60 and start < mark + FRESH_WAIT_SECONDS:
            return False
        quoting = await self.refresh_market()
        pending = await self.cycle_markets(start)
        if quoting and not self.paused():
            await self.decide(pending, time.time())
        await self.sync()
        await self.guard(time.time())
        self.state["cycle"] = {"at": time.time(), "seconds": time.time() - start, "candidates": len(pending),
                               "last_minute": max(self.last.values())}
        self.save()
        return True

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
        """The day's brake and stop, on the FX day's first equity: at -DAILY_BRAKE no new entry until the next FX day
        (open positions run on); at -DAILY_STOP_FRACTION every C13-A position is closed too."""
        day = int(lab.fx_day_array(np.array([now]))[0])
        guard = self.state.setdefault("guard", {})
        if guard.get("day") != day:
            guard.update(day=day, start_equity=self.equity, status="operando")
            self.log("new_day", day=day, start_equity=self.equity)
            self.save()
            return
        result = self.equity - guard["start_equity"]
        if guard["status"] != "limite do dia" and result <= -DAILY_STOP_FRACTION * guard["start_equity"]:
            guard["status"] = "limite do dia"
            await self.flatten("limite do dia")
            self.log("day_closed", result=result)
            self.save()
        elif guard["status"] == "operando" and result <= -DAILY_BRAKE * guard["start_equity"]:
            guard["status"] = "freio do dia"
            self.log("day_brake", result=result)
            self.save()


async def loop() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    demo.remove_container(NAME)
    session = demo.cli_session(NAME)
    # forkserver: this process runs threads (the CLI session, asyncio.to_thread), and forking it could copy a held lock
    workers = ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("forkserver"))
    try:
        await asyncio.to_thread(session.start)
        runner = LiveBanca(session, workers)
        await asyncio.to_thread(runner.seed, time.time())
        await runner.refresh_market()
        await runner.sync()
        runner.log("start", model_sha=runner.model_sha, threshold=runner.threshold, universe=len(runner.universe),
                   risk_by_prediction=RISK_BY_PREDICTION, max_open_risk=MAX_OPEN_RISK, max_asset_risk=MAX_ASSET_RISK,
                   daily_brake=DAILY_BRAKE, daily_stop=DAILY_STOP_FRACTION, rollover_long_block=ROLLOVER_LONG_BLOCK,
                   day_end_margin=DAY_END_MARGIN)
        cycle_at = time.time()  # the first cycle right away, then on the 5-minute grid
        while True:
            if RESTART_FILE.exists():
                RESTART_FILE.unlink()
                runner.log("restart")
                return 0
            if STOP_FILE.exists():
                await runner.flatten("arquivo STOP")
                runner.log("stop")
                return 0
            try:
                await runner.enforce_day_end(time.time())
                if time.time() >= cycle_at:
                    try:
                        if await runner.run_cycle(time.time(), cycle_at - CYCLE_OFFSET):
                            cycle_at = next_cycle(time.time())
                    except (VenueUnavailable, OrderRejected):
                        raise  # the session logs in again by itself: the cycle is retried on the next tick
                    except Exception:
                        cycle_at = next_cycle(time.time())  # a fault of its own: retried on the grid, not every tick
                        raise
            except Exception as error:  # noqa: BLE001 - logged; open positions keep their server-side stops
                runner.log("error", message=f"{type(error).__name__}: {str(error)[:300]}")
            await asyncio.sleep(TICK_SECONDS)
    finally:
        workers.shutdown(cancel_futures=True)
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

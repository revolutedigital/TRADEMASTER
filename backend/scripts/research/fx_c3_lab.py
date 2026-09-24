"""Lab of round C3: published strategies with their authors' parameters, in 41 markets.

See docs/forex/c3-preregistration.md. Data: HistData M1 ASCII (bid only); its clock is New York time with daylight saving (US dates before
2018-12, European dates after), checked against the project's validated M1. Signals are taken on
H1/H4/D1 bars at bar close and executed at the open of the next minute; stops, targets and gaps are followed
minute by minute; the result of a trade is measured in ATR(20) of the strategy's timeframe, net of a fixed cost.

    python -m scripts.research.fx_c3_lab prepare           # zips -> per-instrument M1 arrays
    python -m scripts.research.fx_c3_lab manifest          # data quality only, before any strategy
    python -m scripts.research.fx_c3_lab calibrate         # placebo on group A
    python -m scripts.research.fx_c3_lab discover          # group A
    python -m scripts.research.fx_c3_lab replicate         # group B, only what discovery approved

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from scripts.research import fx_fast2_registry as registry
from scripts.research import fx_fast_lab as lab
from scripts.research import fx_fast_stats as stats
from scripts.research import fx_histdata_ticks as ticks

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs" / "forex"
PREREGISTRATION = DOCS / "c3-preregistration.md"
CONFIGS_FILE = DOCS / "c3-configs.json"
REGISTRY = DOCS / "c3-registry.jsonl"
MANIFEST = DOCS / "c3-data-manifest.csv"
RAW = Path("data/raw/histdata_m1_extra")
ARRAYS = Path("data/c3/m1")
LAB = Path("data/c3")
NULL_T_FILE = LAB / "null_t_discovery.npy"

GROUP_A = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP")
GROUP_B = ("EURCHF", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY", "EURAUD", "EURCAD", "EURNZD", "GBPAUD", "GBPCAD",
           "GBPCHF", "GBPNZD", "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "CADCHF", "XAUUSD", "XAGUSD",
           "SPXUSD", "NSXUSD", "UDXUSD", "WTIUSD", "BCOUSD", "GRXEUR", "UKXGBP", "JPXJPY", "FRXEUR", "AUXAUD", "HKXHKD")
COST_PIPS = {"EURUSD": 0.9, "USDJPY": 0.9, "GBPUSD": 1.1, "AUDUSD": 1.1, "USDCAD": 1.1, "USDCHF": 1.1, "NZDUSD": 1.3,
             "EURGBP": 1.3, "EURJPY": 1.4, "AUDJPY": 1.4, "EURCHF": 1.5, "CADJPY": 1.6, "AUDCAD": 1.7, "AUDCHF": 1.8,
             "AUDNZD": 1.8, "EURAUD": 1.8, "EURCAD": 1.8, "GBPJPY": 1.8, "NZDJPY": 1.8, "CHFJPY": 2.0, "CADCHF": 2.0,
             "NZDCAD": 2.1, "GBPCHF": 2.2, "GBPAUD": 2.3, "NZDCHF": 2.3, "GBPCAD": 2.4, "EURNZD": 2.6, "GBPNZD": 3.3}
COST_PRICE = {"XAUUSD": 0.35, "XAGUSD": 0.035, "SPXUSD": 0.8, "NSXUSD": 2.5, "GRXEUR": 2.0, "UKXGBP": 1.5,
              "FRXEUR": 1.5, "AUXAUD": 1.5, "JPXJPY": 15.0, "HKXHKD": 12.0, "WTIUSD": 0.05, "BCOUSD": 0.05,
              "UDXUSD": 0.06}
STRESS_MULTIPLIER = 2.0
LEGACY_SWITCH = pd.Timestamp("2018-12-01")  # the source's clock follows US dates before, European after
SEGMENT_GAP_SECONDS = 7 * 86400
MIN_MONTH_SHARE, MIN_MONTHS = 0.60, 36
BOOTSTRAP_SEED, PLACEBO_SEED, PLACEBO_REPLICATES = 20260924, 20260925, 200
MIN_TRADES, MIN_PAIR_SHARE_A, MIN_PAIR_SHARE_B = 200, 0.60, 0.55
TIMEFRAMES = ("H1", "H4", "D1")
SAMPLE = ("2012-01", "2026-08")


def cost_of(symbol: str) -> float:
    """Round-trip cost in price units."""
    if symbol in COST_PRICE:
        return COST_PRICE[symbol]
    return COST_PIPS[symbol] * (0.01 if symbol.endswith("JPY") else 0.0001)


# --- data -------------------------------------------------------------------------------------------------------


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        name = next(entry for entry in archive.namelist() if entry.endswith(".csv"))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, sep=";", header=None, names=["ts", "o", "h", "l", "c", "v"], dtype={"ts": str})
    clock = pd.to_datetime(frame["ts"], format="%Y%m%d %H%M%S")
    legacy = clock < LEGACY_SWITCH
    utc = pd.Series(pd.NaT, index=clock.index, dtype="datetime64[ns, UTC]")
    for rule, mask in (("us", legacy), ("europe", ~legacy)):
        if mask.any():
            utc[mask] = ticks.clock_to_utc(clock[mask].reset_index(drop=True), rule).as_unit("ns").to_numpy()
    keep = utc.notna().to_numpy()
    frame, utc = frame[keep], utc[keep]
    seconds = (utc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return pd.DataFrame({"t": seconds.to_numpy(), "o": frame["o"], "h": frame["h"], "l": frame["l"], "c": frame["c"]})


def prepare(symbols=GROUP_A + GROUP_B, raw: Path = RAW, out: Path = ARRAYS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        parts = [read_zip(path) for path in sorted(raw.glob(f"DAT_ASCII_{symbol}_M1_*.zip"))]
        if not parts:
            sys.stdout.write(f"{symbol}: no data\n")
            continue
        frame = pd.concat(parts).drop_duplicates("t", keep="last").sort_values("t")
        np.save(out / f"{symbol}.npy", frame[["t", "o", "h", "l", "c"]].to_numpy(dtype=np.float64))
        sys.stdout.write(f"{symbol}: {len(frame):,} minutes\n")


def build_manifest(symbols=GROUP_A + GROUP_B, arrays: Path = ARRAYS) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        path = arrays / f"{symbol}.npy"
        if not path.exists():
            continue
        m1 = np.load(path)
        months = pd.to_datetime(m1[:, 0], unit="s").strftime("%Y-%m")
        counts = pd.Series(1, index=months).groupby(level=0).size()
        bad = pd.Series(m1[:, 1:5].min(axis=1) <= 0, index=months).groupby(level=0).any()
        median = counts.median()
        for month, count in counts.items():
            inside = SAMPLE[0] <= month <= SAMPLE[1]
            rows.append({"symbol": symbol, "month": month, "minutes": int(count),
                         "included": bool(inside and count >= MIN_MONTH_SHARE * median and not bad[month])})
    table = pd.DataFrame(rows)
    good = table[table["included"]].groupby("symbol").size()
    table["symbol_included"] = table["symbol"].map(lambda s: bool(good.get(s, 0) >= MIN_MONTHS))
    return table


def load_included(symbol: str, manifest: pd.DataFrame, arrays: Path = ARRAYS) -> np.ndarray:
    rows = manifest[(manifest["symbol"] == symbol) & manifest["included"] & manifest["symbol_included"]]
    m1 = np.load(arrays / f"{symbol}.npy", mmap_mode="r")
    months = pd.to_datetime(m1[:, 0], unit="s").strftime("%Y-%m")
    return np.ascontiguousarray(m1[np.isin(months, rows["month"].to_numpy())])


# --- bars -------------------------------------------------------------------------------------------------------


def bucket_keys(seconds: np.ndarray, timeframe: str) -> np.ndarray:
    """Integer id of the bar every minute belongs to (New York clock; the FX day starts at 17:00)."""
    local = pd.to_datetime(seconds, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    shifted = (local + pd.Timedelta(hours=7)).to_numpy().astype("datetime64[h]").astype(np.int64)
    if timeframe == "H1":
        return shifted
    if timeframe == "H4":
        return shifted // 4
    return shifted // 24


@njit(cache=True)
def aggregate(m1, keys):
    """Bars from minutes: (last minute index, open, high, low, close) per bar."""
    n = m1.shape[0]
    count = 0
    for i in range(n):
        if i == 0 or keys[i] != keys[i - 1]:
            count += 1
    last = np.empty(count, dtype=np.int64)
    bars = np.empty((count, 4))
    j = -1
    for i in range(n):
        if i == 0 or keys[i] != keys[i - 1]:
            j += 1
            bars[j, 0] = m1[i, 1]
            bars[j, 1] = m1[i, 2]
            bars[j, 2] = m1[i, 3]
        else:
            bars[j, 1] = max(bars[j, 1], m1[i, 2])
            bars[j, 2] = min(bars[j, 2], m1[i, 3])
        bars[j, 3] = m1[i, 4]
        last[j] = i
    return last, bars


# --- indicators (causal: bar i uses bars up to i) ---------------------------------------------------------------


def wilder(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    if values.size < n:
        return out
    out[n - 1] = values[:n].mean()
    for i in range(n, values.size):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def atr(o, h, l, c, n: int) -> np.ndarray:
    previous = np.concatenate(([c[0]], c[:-1]))
    true_range = np.maximum(h - l, np.maximum(np.abs(h - previous), np.abs(l - previous)))
    return wilder(true_range, n)


def rsi(c: np.ndarray, n: int) -> np.ndarray:
    delta = np.diff(c, prepend=np.nan)[1:]
    gain = wilder(np.maximum(delta, 0.0), n)
    loss = wilder(np.maximum(-delta, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(loss == 0, 100.0, 100.0 - 100.0 / (1.0 + gain / loss))
    return np.concatenate(([np.nan], value))


def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def ema(x, n):
    out = pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy().copy()
    out[: n - 1] = np.nan
    return out


def prior_max(x, n):
    return pd.Series(x).rolling(n).max().shift(1).to_numpy()


def prior_min(x, n):
    return pd.Series(x).rolling(n).min().shift(1).to_numpy()


# --- strategies -------------------------------------------------------------------------------------------------


@dataclass
class Plan:
    """What a strategy wants at the close of every bar; executed at the next minute's open by `simulate`."""

    enter: np.ndarray  # +1 long, -1 short, 0 nothing
    exit_long: np.ndarray
    exit_short: np.ndarray
    stop_level: np.ndarray  # absolute protective stop for an entry (nan: none)
    stop_distance: np.ndarray  # stop at this distance from the fill (nan: none)
    target_r: float  # target at this multiple of the fill-to-stop distance (nan: none)
    trail_k: float  # chandelier: best price since entry minus k * trail_atr (nan: none)
    trail_atr: np.ndarray
    max_bars: int  # exit at the close of this many bars after the entry (0: none)
    unit: np.ndarray  # ATR(20) at the signal bar: the unit of the result


def _plan(n, **fields) -> Plan:
    nan = np.full(n, np.nan)
    base = dict(enter=np.zeros(n, dtype=np.int8), exit_long=np.zeros(n, dtype=bool), exit_short=np.zeros(n, dtype=bool),
                stop_level=nan, stop_distance=nan, target_r=np.nan, trail_k=np.nan, trail_atr=nan, max_bars=0, unit=nan)
    base.update(fields)
    return Plan(**base)


def turtle(o, h, l, c, entry_n, exit_n) -> Plan:
    n20 = atr(o, h, l, c, 20)
    enter = np.where(c > prior_max(h, entry_n), 1, np.where(c < prior_min(l, entry_n), -1, 0)).astype(np.int8)
    return _plan(c.size, enter=enter, exit_long=c < prior_min(l, exit_n), exit_short=c > prior_max(h, exit_n),
                 stop_distance=2.0 * n20, unit=n20)


def tsmom(o, h, l, c) -> Plan:
    momentum = c / pd.Series(c).shift(252).to_numpy() - 1.0
    enter = np.where(momentum > 0, 1, np.where(momentum < 0, -1, 0)).astype(np.int8)
    return _plan(c.size, enter=enter, unit=atr(o, h, l, c, 20))


def rsi2(o, h, l, c) -> Plan:
    value, trend, fast = rsi(c, 2), sma(c, 200), sma(c, 5)
    enter = np.where((c > trend) & (value < 10), 1, np.where((c < trend) & (value > 90), -1, 0)).astype(np.int8)
    return _plan(c.size, enter=enter, exit_long=c > fast, exit_short=c < fast, unit=atr(o, h, l, c, 20))


def squeeze(o, h, l, c) -> Plan:
    middle = sma(c, 20)
    deviation = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    upper, lower = middle + 2 * deviation, middle - 2 * deviation
    width = (upper - lower) / middle
    at_low = width <= pd.Series(width).rolling(125).min().to_numpy()
    setup = np.concatenate(([False], at_low[:-1]))  # the previous bar closed at the 125-bar low of the width
    enter = np.where(setup & (c > upper), 1, np.where(setup & (c < lower), -1, 0)).astype(np.int8)
    stop = np.where(enter == 1, lower, np.where(enter == -1, upper, np.nan))
    return _plan(c.size, enter=enter, exit_long=c < middle, exit_short=c > middle, stop_level=stop,
                 unit=atr(o, h, l, c, 20))


def nr7(o, h, l, c) -> Plan:
    rng = h - l
    narrow = rng <= pd.Series(rng).rolling(7).min().to_numpy()
    previous_narrow = np.concatenate(([False], narrow[:-1]))
    previous_high, previous_low = np.concatenate(([np.nan], h[:-1])), np.concatenate(([np.nan], l[:-1]))
    enter = np.where(previous_narrow & (c > previous_high), 1,
                     np.where(previous_narrow & (c < previous_low), -1, 0)).astype(np.int8)
    stop = np.where(enter == 1, previous_low, np.where(enter == -1, previous_high, np.nan))
    return _plan(c.size, enter=enter, stop_level=stop, max_bars=3, unit=atr(o, h, l, c, 20))


def macd200(o, h, l, c) -> Plan:
    line = ema(c, 12) - ema(c, 26)
    signal = pd.Series(line).ewm(span=9, adjust=False).mean().to_numpy()
    before = np.concatenate(([np.nan], line[:-1] - signal[:-1]))
    now = line - signal
    trend = ema(c, 200)
    long = (before <= 0) & (now > 0) & (line < 0) & (c > trend)
    short = (before >= 0) & (now < 0) & (line > 0) & (c < trend)
    enter = np.where(long, 1, np.where(short, -1, 0)).astype(np.int8)
    stop = np.where(long, prior_min(l, 10), np.where(short, prior_max(h, 10), np.nan))
    return _plan(c.size, enter=enter, stop_level=stop, target_r=1.5, unit=atr(o, h, l, c, 20))


def chandelier(o, h, l, c) -> Plan:
    fast, slow = ema(c, 50), ema(c, 200)
    above = fast > slow
    was_above = np.concatenate(([False], above[:-1]))
    valid = ~np.isnan(slow) & ~np.isnan(np.concatenate(([np.nan], slow[:-1])))
    enter = np.where(valid & above & ~was_above, 1, np.where(valid & ~above & was_above, -1, 0)).astype(np.int8)
    a22 = atr(o, h, l, c, 22)
    return _plan(c.size, enter=enter, stop_distance=3.0 * a22, trail_k=3.0, trail_atr=a22, unit=atr(o, h, l, c, 20))


STRATEGIES: dict[str, tuple[str, Callable[..., Plan]]] = {
    "TURTLE20": ("D1", lambda o, h, l, c: turtle(o, h, l, c, 20, 10)),
    "TURTLE55": ("D1", lambda o, h, l, c: turtle(o, h, l, c, 55, 20)),
    "TSMOM": ("D1", tsmom),
    "RSI2_D1": ("D1", rsi2), "RSI2_H4": ("H4", rsi2), "RSI2_H1": ("H1", rsi2),
    "SQUEEZE_D1": ("D1", squeeze), "SQUEEZE_H4": ("H4", squeeze),
    "NR7_D1": ("D1", nr7), "NR7_H4": ("H4", nr7),
    "MACD200_H1": ("H1", macd200), "MACD200_H4": ("H4", macd200),
    "CHANDELIER_D1": ("D1", chandelier), "CHANDELIER_H4": ("H4", chandelier),
}


# --- simulation -------------------------------------------------------------------------------------------------

REASON_STOP, REASON_TARGET, REASON_RULE, REASON_END = 1, 2, 3, 4


@njit(cache=True)
def _record(out, k, side, entry_index, exit_index, entry, exit_, unit, reason):
    out[k, 0], out[k, 1], out[k, 2], out[k, 3] = side, entry_index, exit_index, entry
    out[k, 4], out[k, 5], out[k, 6] = exit_, unit, reason


@njit(cache=True)
def simulate(m1, last, bar_high, bar_low, enter, exit_long, exit_short, stop_level, stop_distance, target_r,
             trail_k, trail_atr, max_bars, unit, gap_seconds):
    """Walk the minutes; decisions at bar closes, fills at the next minute's open. Returns one row per trade:
    (side, entry index, exit index, entry price, exit price, unit, reason)."""
    n_bars = last.shape[0]
    out = np.empty((n_bars + 1, 7))
    trades = 0
    side = 0
    entry_price = stop = target = best = 0.0
    entry_index = 0
    trade_unit = 0.0
    held = 0
    pending_exit = False
    pending_side = 0
    pending_level = pending_distance = pending_unit = 0.0
    first = 0
    for j in range(n_bars):
        for i in range(first, last[j] + 1):
            ends_segment = i + 1 < m1.shape[0] and m1[i + 1, 0] - m1[i, 0] > gap_seconds
            if i == first and j > 0:
                open_price = m1[i, 1]
                if pending_exit and side != 0:
                    _record(out, trades, side, entry_index, i, entry_price, open_price, trade_unit, REASON_RULE)
                    trades += 1
                    side = 0
                if pending_side != 0 and side == 0:
                    fill = open_price
                    level = np.nan
                    if not np.isnan(pending_level):
                        level = pending_level
                    elif not np.isnan(pending_distance):
                        level = fill - pending_side * pending_distance
                    if np.isnan(level) or (pending_side == 1 and level < fill) or (pending_side == -1 and level > fill):
                        side = pending_side
                        entry_price, entry_index, trade_unit, held, best = fill, i, pending_unit, 0, fill
                        stop = level
                        target = np.nan
                        if not np.isnan(target_r) and not np.isnan(level):
                            target = fill + side * target_r * abs(fill - level)
                pending_exit = False
                pending_side = 0
            if side != 0:
                exit_price = np.nan
                reason = 0
                if side == 1:
                    if not np.isnan(stop) and m1[i, 3] <= stop:
                        exit_price, reason = min(stop, m1[i, 1]), REASON_STOP
                    elif not np.isnan(target) and m1[i, 2] >= target:
                        exit_price, reason = max(target, m1[i, 1]), REASON_TARGET
                else:
                    if not np.isnan(stop) and m1[i, 2] >= stop:
                        exit_price, reason = max(stop, m1[i, 1]), REASON_STOP
                    elif not np.isnan(target) and m1[i, 3] <= target:
                        exit_price, reason = min(target, m1[i, 1]), REASON_TARGET
                if reason == 0 and ends_segment:
                    exit_price, reason = m1[i, 4], REASON_END
                if reason != 0:
                    _record(out, trades, side, entry_index, i, entry_price, exit_price, trade_unit, reason)
                    trades += 1
                    side = 0
            if ends_segment:
                pending_exit = False
                pending_side = 0
        first = last[j] + 1
        # decisions at the close of bar j
        if side != 0:
            held += 1
            if not np.isnan(trail_k) and not np.isnan(trail_atr[j]):
                if side == 1:
                    best = max(best, bar_high[j])
                    candidate = best - trail_k * trail_atr[j]
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                else:
                    best = min(best, bar_low[j])
                    candidate = best + trail_k * trail_atr[j]
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
            if (side == 1 and (exit_long[j] or enter[j] == -1)) or (side == -1 and (exit_short[j] or enter[j] == 1)) \
                    or (max_bars > 0 and held >= max_bars):
                pending_exit = True
        if enter[j] != 0 and (side == 0 or (pending_exit and enter[j] != side)) and not np.isnan(unit[j]):
            pending_side = enter[j]
            pending_level, pending_distance, pending_unit = stop_level[j], stop_distance[j], unit[j]
    if side != 0 and m1.shape[0] > 0:
        k = m1.shape[0] - 1
        _record(out, trades, side, entry_index, k, entry_price, m1[k, 4], trade_unit, REASON_END)
        trades += 1
    return out[:trades]


def run_instrument(m1: np.ndarray, keys: dict[str, np.ndarray], configs: list[str], cost: float) -> dict[str, np.ndarray]:
    """Trades per configuration: columns side, entry index, exit index, entry, exit, unit, reason, net base, net stress."""
    results = {}
    bars_by_tf = {tf: aggregate(m1, keys[tf]) for tf in {STRATEGIES[k][0] for k in configs}}
    for key in configs:
        tf, build = STRATEGIES[key]
        last, bars = bars_by_tf[tf]
        o, h, l, c = bars[:, 0], bars[:, 1], bars[:, 2], bars[:, 3]
        plan = build(o, h, l, c)
        trades = simulate(m1, last, h, l, plan.enter, plan.exit_long, plan.exit_short, plan.stop_level,
                          plan.stop_distance, plan.target_r, plan.trail_k, plan.trail_atr, plan.max_bars, plan.unit,
                          float(SEGMENT_GAP_SECONDS))
        move = trades[:, 0] * (trades[:, 4] - trades[:, 3])
        net = (move - cost) / trades[:, 5]
        stressed = (move - STRESS_MULTIPLIER * cost) / trades[:, 5]
        results[key] = np.column_stack([trades, net, stressed])
    return results


# --- placebo ----------------------------------------------------------------------------------------------------


@njit(cache=True)
def sign_flip(m1, coins):
    """Every minute's move away from the previous close changes sign with the coin; high and low mirror."""
    out = np.empty_like(m1)
    previous_original = m1[0, 1]
    previous_new = previous_original
    for t in range(m1.shape[0]):
        s = float(coins[t])
        o = previous_new + s * (m1[t, 1] - previous_original)
        c = previous_new + s * (m1[t, 4] - previous_original)
        if s > 0:
            h = previous_new + (m1[t, 2] - previous_original)
            lo = previous_new + (m1[t, 3] - previous_original)
        else:
            h = previous_new - (m1[t, 3] - previous_original)
            lo = previous_new - (m1[t, 2] - previous_original)
        out[t, 0] = m1[t, 0]
        out[t, 1], out[t, 4] = o, c
        out[t, 2] = max(h, o, c)
        out[t, 3] = min(lo, o, c)
        previous_original = m1[t, 4]
        previous_new = c
    return out


# --- samples and statistics -------------------------------------------------------------------------------------


@dataclass
class Sample:
    keys: list[str]
    sums: np.ndarray
    counts: np.ndarray
    stress_mean: np.ndarray
    instrument_means: dict[str, dict[str, float]]
    trades: dict[str, pd.DataFrame]


def fx_days(seconds: np.ndarray) -> np.ndarray:
    return lab.fx_day_array(np.asarray(seconds, dtype=np.float64))


def run_group(symbols, manifest, configs=None, seed: int | None = None) -> Sample:
    configs = configs or list(STRATEGIES)
    per_config: dict[str, list[pd.DataFrame]] = {k: [] for k in configs}
    table = None
    for symbol in symbols:
        if not manifest[(manifest["symbol"] == symbol) & manifest["symbol_included"]].shape[0]:
            continue
        m1 = load_included(symbol, manifest)
        if seed is not None:
            first = int(manifest_first_minute(manifest))
            if table is None:
                table = np.random.default_rng(seed).integers(0, 2, 12_000_000).astype(np.int8) * 2 - 1
            m1 = sign_flip(m1, table[(m1[:, 0] // 60).astype(np.int64) - first])
        keys = {tf: bucket_keys(m1[:, 0], tf) for tf in TIMEFRAMES}
        for key, trades in run_instrument(m1, keys, configs, cost_of(symbol)).items():
            per_config[key].append(pd.DataFrame({
                "symbol": symbol, "side": trades[:, 0], "entry_time": m1[trades[:, 1].astype(np.int64), 0],
                "exit_time": m1[trades[:, 2].astype(np.int64), 0], "reason": trades[:, 6],
                "net": trades[:, 7], "stress": trades[:, 8]}))
    frames = {k: pd.concat(v) if v else pd.DataFrame(columns=["symbol", "entry_time", "net", "stress"])
              for k, v in per_config.items()}
    all_days = np.unique(np.concatenate([fx_days(f["entry_time"].to_numpy()) for f in frames.values() if len(f)]
                                        or [np.array([0])]))
    day_parts, values, ids = [], [], []
    for index, key in enumerate(configs):
        frame = frames[key]
        day_parts.append(np.searchsorted(all_days, fx_days(frame["entry_time"].to_numpy())))
        values.append(frame["net"].to_numpy(dtype=float))
        ids.append(np.full(len(frame), index))
    sums, counts = stats.daily_tables(np.concatenate(day_parts), np.concatenate(values), np.concatenate(ids),
                                      len(all_days), len(configs))
    stress_mean = np.array([frames[k]["stress"].mean() if len(frames[k]) else np.nan for k in configs])
    means = {k: frames[k].groupby("symbol")["net"].mean().to_dict() if len(frames[k]) else {} for k in configs}
    return Sample(list(configs), sums, counts, stress_mean, means, frames)


def manifest_first_minute(manifest: pd.DataFrame) -> int:
    return int(pd.Timestamp(f"{manifest['month'].min()}-01").timestamp() // 60) - 24 * 60


def evaluate_discovery(sample: Sample, symbols, null_best: np.ndarray, draws: int = stats.BOOTSTRAP_DRAWS) -> pd.DataFrame:
    mean, error, t = stats.cluster_t(sample.sums, sample.counts)
    _, p_a = stats.bootstrap_adjusted_p_values(sample.sums, sample.counts, draws=draws, seed=BOOTSTRAP_SEED)
    p_b = stats.placebo_p_value(t, null_best)
    rows = []
    for k, key in enumerate(sample.keys):
        frame = sample.trades[key]
        present = [s for s in symbols if s in sample.instrument_means[key]]
        share = sum(sample.instrument_means[key][s] > 0 for s in present) / max(len(present), 1)
        trades = len(frame)
        approved = bool(p_a[k] <= 0.05 and p_b[k] <= 0.05 and mean[k] > 0 and sample.stress_mean[k] > 0
                        and share >= MIN_PAIR_SHARE_A and trades >= MIN_TRADES)
        rows.append(describe(key, frame, mean[k], sample.stress_mean[k], t[k], share) |
                    {"p_bootstrap": p_a[k], "p_placebo": p_b[k], "approved": approved})
    return pd.DataFrame(rows)


def describe(key, frame, mean, stress_mean, t, share) -> dict:
    wins = frame["net"] > 0
    return {"config": key, "trades": len(frame), "mean_atr": mean, "mean_atr_stress": stress_mean, "t": t,
            "win_rate": float(wins.mean()) if len(frame) else np.nan,
            "avg_win": float(frame.loc[wins, "net"].mean()) if wins.any() else np.nan,
            "avg_loss": float(frame.loc[~wins, "net"].mean()) if (~wins).any() else np.nan,
            "instruments_positive": share,
            "trades_per_year": len(frame) / max((frame["entry_time"].max() - frame["entry_time"].min()) / 31_557_600, 1e-9)
            if len(frame) > 1 else 0.0}


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def evaluate_replication(sample: Sample, symbols) -> pd.DataFrame:
    mean, error, t = stats.cluster_t(sample.sums, sample.counts)
    from scipy.stats import norm

    p_raw = norm.sf(np.where(np.isfinite(t), t, -np.inf))
    p = holm(p_raw)
    rows = []
    for k, key in enumerate(sample.keys):
        frame = sample.trades[key]
        present = [s for s in symbols if s in sample.instrument_means[key]]
        share = sum(sample.instrument_means[key][s] > 0 for s in present) / max(len(present), 1)
        approved = bool(p[k] <= 0.05 and mean[k] > 0 and sample.stress_mean[k] > 0 and share >= MIN_PAIR_SHARE_B
                        and len(frame) >= MIN_TRADES)
        rows.append(describe(key, frame, mean[k], sample.stress_mean[k], t[k], share) | {"p_holm": p[k], "approved": approved})
    return pd.DataFrame(rows)


# --- commands ---------------------------------------------------------------------------------------------------


def _append(event: dict) -> None:
    registry.append_event(event, registry=REGISTRY, preregistration=PREREGISTRATION, grid=CONFIGS_FILE)


def _events() -> list[dict]:
    return registry.read_events(REGISTRY)


def _placebo_worker(seed: int) -> Sample:
    manifest = pd.read_csv(MANIFEST)
    sample = run_group(GROUP_A, manifest, seed=seed)
    sample.trades = {k: f[["symbol", "entry_time", "net", "stress"]] for k, f in sample.trades.items()}
    return sample


def _write(name: str, title: str, table: pd.DataFrame, extra: str = "") -> Path:
    path = DOCS / name
    body = table.to_string(index=False, float_format=lambda value: f"{value:.4g}")
    path.write_text(f"# {title}\n\n{extra}\n\n```\n{body}\n```\n", encoding="utf-8")
    return path


def calibrate(workers: int, replicates: int) -> int:
    commit = lab.code_commit()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        samples = list(pool.map(_placebo_worker, [PLACEBO_SEED + i for i in range(replicates)]))
    null_t = np.stack([stats.cluster_t(s.sums, s.counts)[2] for s in samples])
    LAB.mkdir(parents=True, exist_ok=True)
    np.save(NULL_T_FILE, null_t)
    best = np.where(np.isfinite(null_t), null_t, -np.inf).max(axis=1)
    false = sum(bool(evaluate_discovery(s, GROUP_A, np.delete(best, i))["approved"].any()) for i, s in enumerate(samples))
    rate = false / replicates
    summary = pd.DataFrame({"config": list(STRATEGIES), "median_t": np.nanmedian(null_t, axis=0),
                            "p95_t": np.nanpercentile(null_t, 95, axis=0)})
    _write("c3-calibration-report.md", "C3: calibração do critério em dados sem vantagem", summary,
           f"Commit do código: {commit}. Conjuntos placebo do grupo A: {replicates}. Falsa aprovação do critério "
           f"inteiro (cada conjunto contra os outros): {rate:.1%} (limite 8%). Melhor t por conjunto: mediana "
           f"{np.median(best):.2f}, p95 {np.percentile(best, 95):.2f}.")
    sys.stdout.write(f"false approval {rate:.3f}\n")
    if rate > 0.08:
        return 2
    _append({"event": "calibration_report", "false_pass_rate": rate, "replicates": replicates, "code_commit": commit})
    return 0


def discover() -> int:
    commit = lab.code_commit()
    manifest = pd.read_csv(MANIFEST)
    sample = run_group(GROUP_A, manifest)
    null_t = np.load(NULL_T_FILE)
    table = evaluate_discovery(sample, GROUP_A, np.where(np.isfinite(null_t), null_t, -np.inf).max(axis=1))
    for row in table.to_dict("records"):
        _append({"event": "run", "configuration": row["config"], "sample": "discovery", "code_commit": commit,
                 "result": lab._plain(row)})  # noqa: SLF001
    approved = table.loc[table["approved"], "config"].tolist()
    _write("c3-discovery-report.md", "C3: descoberta no grupo A (10 pares, 2012-01 a 2026-08)",
           table.sort_values("t", ascending=False),
           "Resultado em ATR(20) por operação, líquido do custo. `avg_win`/`avg_loss` em ATR.")
    _append({"event": "discovery_report", "approved": approved, "report": "c3-discovery-report.md"})
    sys.stdout.write(f"approved: {approved}\n")
    return 0


def replicate() -> int:
    commit = lab.code_commit()
    approved = [e for e in _events() if e["event"] == "discovery_report"][-1]["approved"]
    if not approved:
        sys.stdout.write("nothing approved in discovery: stop and talk to Igor\n")
        return 0
    manifest = pd.read_csv(MANIFEST)
    sample = run_group(GROUP_B, manifest, configs=approved)
    table = evaluate_replication(sample, GROUP_B)
    for row in table.to_dict("records"):
        _append({"event": "run", "configuration": row["config"], "sample": "replication", "code_commit": commit,
                 "result": lab._plain(row)})  # noqa: SLF001
    passed = table.loc[table["approved"], "config"].tolist()
    _write("c3-replication-report.md", "C3: réplica no grupo B (31 mercados novos, 2012-01 a 2026-08)",
           table.sort_values("t", ascending=False))
    _append({"event": "replication_report", "approved": passed, "report": "c3-replication-report.md"})
    sys.stdout.write(f"replicated: {passed}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["prepare", "manifest", "calibrate", "discover", "replicate"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=PLACEBO_REPLICATES)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare()
        return 0
    if args.command == "manifest":
        table = build_manifest()
        table.to_csv(MANIFEST, index=False)
        summary = table.groupby("symbol").agg(months=("month", "size"), included=("included", "sum"),
                                              symbol_included=("symbol_included", "first"))
        sys.stdout.write(summary.to_string() + "\n")
        return 0
    if args.command == "calibrate":
        return calibrate(args.workers, args.replicates)
    if args.command == "discover":
        return discover()
    return replicate()


if __name__ == "__main__":
    raise SystemExit(main())

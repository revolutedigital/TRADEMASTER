"""Round C11: day trades on money flows and the calendar. Eleven fixed time windows with a published reason to exist
(dealer flows around the currency fixes, month-end equity hedging at the London fix, the overnight drift of equity
index futures, the turn of the month), each held for its window and flat before the FX day ends. Discovery on
2012-2026 with the max-t bootstrap over the eleven, then one run on untouched 2003-2011 data.
See docs/forex/c11-preregistration.md.

    python -m scripts.research.fx_c11_lab discover [--workers N]
    python -m scripts.research.fx_c11_lab holdout [--workers N]

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c5_lab as c5
from scripts.research import fx_c10_core as core
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_fast_stats as stats

PREREGISTRATION = c3.DOCS / "c11-preregistration.md"
REGISTRY = c3.DOCS / "c11-registry.jsonl"
MIN_TRADES, MIN_MARKET_SHARE = 200, 0.55  # market share as in C10: markets with 20+ trades
HOLDOUT_MIN_TRADES, SIGNIFICANCE = 300, 0.05
BOOTSTRAP_SEED, PLACEBO_SEED, PLACEBO_COPIES = 20261011, 20261012, 20
STOP_ATR = 1.0  # a protective stop of one daily ATR; the window's end is the exit
ENTRY_TOLERANCE = 5  # the entry is the first minute within five minutes of the window's start

LONG_USD = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1, "USDJPY": 1, "USDCAD": 1, "USDCHF": 1, "UDXUSD": 1}
SHORT_USD = {symbol: -side for symbol, side in LONG_USD.items()}
EQUITY = ("SPXUSD", "NSXUSD", "GRXEUR", "UKXGBP", "JPXJPY", "FRXEUR", "AUXAUD", "HKXHKD")
US_EQUITY_YAHOO, FOREIGN_EQUITY_YAHOO = "^GSPC", ("^GDAXI", "^FTSE", "^N225", "^FCHI", "^AXJO", "^HSI")


@dataclass(frozen=True)
class Window:
    name: str
    zone: str  # a timezone, or FX: the FX day's clock (New York + 7 hours, midnight = 17:00 New York)
    start: int  # local minute of the entry
    minutes: int  # clock minutes held; 0 = to the end of the FX day
    markets: dict[str, int]  # side of each market before the day rule
    days: str = "all"  # "all", "month_end" (last weekday of the month, side times the month-end sign), "tom"
    tolerance: int = ENTRY_TOLERANCE  # the entry is the first minute within this many minutes of the start


NY, FX = "America/New_York", "FX"
WINDOWS = (
    Window("fix_tokyo_before", "Asia/Tokyo", 8 * 60 + 55, 60, LONG_USD),
    Window("fix_tokyo_after", "Asia/Tokyo", 9 * 60 + 55, 60, SHORT_USD),
    Window("fix_ecb_before", "Europe/Berlin", 13 * 60 + 15, 60, LONG_USD),
    Window("fix_ecb_after", "Europe/Berlin", 14 * 60 + 15, 60, SHORT_USD),
    Window("fix_london_before", "Europe/London", 15 * 60, 60, LONG_USD),
    Window("fix_london_after", "Europe/London", 16 * 60, 60, SHORT_USD),
    Window("month_end_london_before", "Europe/London", 15 * 60, 60, LONG_USD, days="month_end"),
    Window("month_end_london_after", "Europe/London", 16 * 60, 60, SHORT_USD, days="month_end"),
    Window("overnight_us", NY, 18 * 60 + 5, 920, {"SPXUSD": 1, "NSXUSD": 1}),  # to 09:25 New York
    Window("europe_open_us", NY, 2 * 60, 120, {"SPXUSD": 1, "NSXUSD": 1}),  # 02:00 to 04:00 New York
    # each market's first minute from 18:05 New York on, held to the end of the FX day
    Window("turn_of_month", FX, 65, 0, dict.fromkeys(EQUITY, 1), days="tom", tolerance=600),
)
SYMBOLS = tuple(sorted({s for w in WINDOWS for s in w.markets}))


def month_end_sign(closes: dict[str, pd.Series], day: pd.Timestamp) -> int:
    """+1 (long USD into the London fix) when US equities lagged foreign equities over the month up to the day before,
    -1 when they led (Melvin and Prins 2015: foreign holders then sell dollars to rebalance their hedges), 0 unknown."""
    month_start = day.replace(day=1)
    returns = {}
    for name, series in closes.items():
        before_month, before_day = series[series.index < month_start], series[series.index < day]
        if len(before_month) and len(before_day) and before_day.index[-1] >= month_start:
            returns[name] = before_day.iloc[-1] / before_month.iloc[-1] - 1
    foreign = [returns[n] for n in FOREIGN_EQUITY_YAHOO if n in returns]
    if US_EQUITY_YAHOO not in returns or not foreign:
        return 0
    relative = returns[US_EQUITY_YAHOO] - float(np.mean(foreign))
    return 0 if relative == 0 else (-1 if relative > 0 else 1)


def yahoo_closes() -> dict[str, pd.Series]:
    out = {}
    for symbol in (US_EQUITY_YAHOO, *FOREIGN_EQUITY_YAHOO):
        frame = pd.read_csv(c5.DATA / f"{symbol.lstrip('^')}.csv").dropna(subset=["c"])
        frame = frame[frame["c"] > 0]
        out[symbol] = pd.Series(frame["c"].to_numpy(), index=pd.to_datetime(frame["t"], unit="s").dt.normalize()) \
            .groupby(level=0).last()
    return out


def _weekdays(first: str, last: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    days = pd.Series(pd.bdate_range(first, last))
    month = [days.dt.year, days.dt.month]
    return days, days.groupby(month).cumcount(), days == days.groupby(month).transform("max")


def last_weekdays(first: str, last: str) -> set[pd.Timestamp]:
    days, _, is_last = _weekdays(first, last)
    return set(days[is_last])


def turn_of_month_days(first: str, last: str) -> set[pd.Timestamp]:
    """The last weekday of each month and the first three of the next."""
    days, rank, is_last = _weekdays(first, last)
    return set(days[is_last | (rank < 3)])


def window_entries(m1: np.ndarray, window: Window, day_side: dict[pd.Timestamp, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """Entry minute and side multiplier of every day of the window: the first minute of the local day within
    `tolerance` minutes after the start. `day_side` maps a date to a multiplier; days missing from it are not
    traded."""
    if window.zone == FX:
        clock = core.minute_keys(m1[:, 0], 1)
        day = clock // 1440
        minute = clock - day * 1440
    else:
        minute, day = core.local_minutes(m1[:, 0], window.zone)
    inside = np.flatnonzero((minute >= window.start) & (minute < window.start + window.tolerance))
    if not len(inside):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    first = inside[np.concatenate(([True], day[inside][1:] != day[inside][:-1]))]
    if day_side is None:
        return first.astype(np.int64), np.ones(len(first), dtype=np.int64)
    dates = pd.to_datetime(day[first] * 86400, unit="s")  # the local date (the FX day's date on the FX clock)
    multiplier = np.array([day_side.get(d, 0) for d in dates], dtype=np.int64)
    keep = multiplier != 0
    return first[keep].astype(np.int64), multiplier[keep]


def day_rules(first: str, last: str) -> dict[str, dict[pd.Timestamp, int]]:
    closes = yahoo_closes()
    return {"month_end": {d: month_end_sign(closes, d) for d in last_weekdays(first, last)},
            "tom": dict.fromkeys(turn_of_month_days(first, last), 1)}


def market_worker(job: tuple) -> dict:
    """Trades of every window that includes this market: (window, FX day, R net, R stress)."""
    symbol, manifest_path, arrays, rules, coin_seed = job
    m1 = c3.load_included(symbol, pd.read_csv(manifest_path), arrays=Path(arrays))
    if coin_seed is not None:
        coins = np.random.default_rng([coin_seed, SYMBOLS.index(symbol)]).integers(0, 2, len(m1)) * 2 - 1
        m1 = c3.sign_flip(m1, coins.astype(np.int8))
    day_end = core.day_end_minutes(m1)
    daily = core.build_context(symbol, m1, "H1")  # only for the daily ATR known before each minute
    atr_by_minute = np.full(len(m1), np.nan)
    atr_by_minute[daily.last_minute] = daily.atr_d1
    atr_by_minute = pd.Series(atr_by_minute).bfill().to_numpy()  # each minute takes its own hour's (causal) value
    fx_days = core.minute_keys(m1[:, 0], 1440)
    cost = c3.cost_of(symbol)
    out = []
    for w, window in enumerate(WINDOWS):
        if symbol not in window.markets:
            continue
        entries, multiplier = window_entries(m1, window, None if window.days == "all" else rules[window.days])
        sides = window.markets[symbol] * multiplier
        atrs = atr_by_minute[entries]
        ok = np.isfinite(atrs)
        grid = np.array([[STOP_ATR, 0.0, 0.0, 0.0, 0.0, float(window.minutes)]])
        trades = core.manage(m1, day_end, entries[ok], sides[ok], np.full(ok.sum(), np.nan), atrs[ok], cost, grid, 0,
                             False)
        for trade in trades:
            out.append((w, int(fx_days[int(trade[1])]), trade[3], trade[4]))
    return {"symbol": symbol, "trades": np.array(out, dtype=float).reshape(-1, 4)}


def run(manifest_path: Path, arrays: Path, rules, workers: int, coin_seed=None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    symbols = [s for s in SYMBOLS if s in set(manifest.loc[manifest["symbol_included"], "symbol"])]
    jobs = [(s, str(manifest_path), str(arrays), rules, coin_seed) for s in symbols]
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(market_worker, jobs):
            frame = pd.DataFrame(out["trades"], columns=["config", "day", "net", "stress"])
            frames.append(frame.assign(symbol=out["symbol"], config=frame["config"].astype(int)))
    return pd.concat(frames, ignore_index=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def _registered() -> tuple[str, str]:
    if _git("status", "--porcelain", "--", "scripts", "app", str(PREREGISTRATION), str(REGISTRY)):
        raise RuntimeError("commit the code, the pre-registration and the registry before running")
    return _git("rev-parse", "HEAD"), hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()


def _events(kind: str) -> list[dict]:
    lines = REGISTRY.read_text(encoding="utf-8").splitlines() if REGISTRY.exists() else []
    return [event for event in map(json.loads, lines) if event.get("event") == kind]


def _append(event: dict) -> None:
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=float) + "\n")


def _table(trades: pd.DataFrame, configs: list[int]) -> pd.DataFrame:
    result = c10.evaluate(trades, configs)
    result.insert(1, "window", [WINDOWS[k].name for k in result["config"]])
    return result


def discover(workers: int) -> int:
    commit, prereg = _registered()
    if _events("c11_discovery"):
        raise RuntimeError("discovery already ran: its result is in the registry")
    rules = day_rules("2011-12-01", "2026-09-30")
    trades = run(c3.MANIFEST, c3.ARRAYS, rules, workers)
    configs = list(range(len(WINDOWS)))
    all_days = np.unique(trades["day"])
    day = np.searchsorted(all_days, trades["day"].to_numpy())
    sums, counts = stats.daily_tables(day, trades["net"].to_numpy(), trades["config"].to_numpy(), len(all_days),
                                      len(configs))
    _, p_maxt = stats.bootstrap_adjusted_p_values(sums, counts, seed=BOOTSTRAP_SEED)
    table = _table(trades, configs)  # market_share: markets with 20+ trades
    table["p_maxt"] = p_maxt
    table["approved"] = ((table["p_maxt"] <= SIGNIFICANCE) & (table["mean_r"] > 0) & (table["mean_r_stress"] > 0)
                         & (table["market_share"] >= MIN_MARKET_SHARE) & (table["trades"] >= MIN_TRADES))
    trades["year"] = pd.to_datetime(trades["day"] * 86400, unit="s").dt.year
    by_year = trades.pivot_table(index="config", columns="year", values="net", aggfunc="mean").round(3)
    by_year.index = [WINDOWS[k].name for k in by_year.index]
    by_market = trades.pivot_table(index="config", columns="symbol", values="net", aggfunc="mean").round(3)
    by_market.index = [WINDOWS[k].name for k in by_market.index]
    candidates = [int(k) for k in table.loc[table["approved"], "config"]]
    text = (f"# C11: descoberta (2012-01 a 2026-08)\n\nCommit do código: {commit}. **Aprovadas: {len(candidates)} de "
            f"{len(WINDOWS)}.** Unidade: R = 1 ATR diário (o stop de proteção); saída no fim da janela.\n\n"
            f"```\n{c10._fmt(table)}\n```\n\nPor ano:\n\n```\n{by_year.to_string()}\n```\n\n"
            f"Por mercado:\n\n```\n{by_market.to_string()}\n```\n")
    (c3.DOCS / "c11-discovery.md").write_text(text, encoding="utf-8")
    _append({"date": date.today().isoformat(), "event": "c11_discovery", "code_commit": commit,
             "preregistration_sha256": prereg, "candidates": candidates, "results": table.to_dict(orient="records")})
    sys.stdout.write(text)
    return 0


def holdout(workers: int) -> int:
    commit, prereg = _registered()
    if _events("c11_holdout"):
        raise RuntimeError("the untouched period is used once: the holdout already ran")
    discoveries = _events("c11_discovery")
    if not discoveries:
        raise RuntimeError("run discovery first")
    configs = [int(k) for k in discoveries[-1]["candidates"]]
    if not configs:
        sys.stdout.write("no candidates from discovery: nothing to confirm\n")
        return 0
    discovery_mean = {r["config"]: r["mean_r"] for r in discoveries[-1]["results"]}
    trades = run(c10.HOLDOUT_MANIFEST, c10.HOLDOUT_ARRAYS, day_rules("2002-12-01", "2012-01-31"), workers)
    result = _table(trades[trades["config"].isin(configs)], configs)
    result["p_holm"] = c3.holm(result["p_one_sided"].fillna(1.0).to_numpy())
    result["approved"] = ((result["p_holm"] <= SIGNIFICANCE) & (result["mean_r"] > 0) & (result["mean_r_stress"] > 0)
                          & (result["trades"] >= HOLDOUT_MIN_TRADES) & (result["market_share"] >= MIN_MARKET_SHARE))
    approved = [int(k) for k in result.loc[result["approved"], "config"]]
    if approved:
        rules = day_rules("2011-12-01", "2026-09-30")
        copies = []
        for i in range(PLACEBO_COPIES):
            flipped = run(c3.MANIFEST, c3.ARRAYS, rules, workers, coin_seed=PLACEBO_SEED + i)
            copies.append(_table(flipped[flipped["config"].isin(approved)], approved))
        placebo = pd.concat(copies)
        for k in approved:
            reached = int((placebo.loc[placebo["config"] == k, "mean_r"] >= discovery_mean[k]).sum())
            result.loc[result["config"] == k, "placebo_reached"] = reached
            if reached > 1:
                result.loc[result["config"] == k, "approved"] = False
    text = (f"# C11: o dado intocado (2003-01 a 2011-12)\n\nCommit do código: {commit}. **Aprovadas: "
            f"{int(result['approved'].sum())} de {len(configs)}.**\n\n```\n{c10._fmt(result)}\n```\n")
    (c3.DOCS / "c11-holdout.md").write_text(text, encoding="utf-8")
    _append({"date": date.today().isoformat(), "event": "c11_holdout", "code_commit": commit,
             "preregistration_sha256": prereg, "approved": [int(k) for k in result.loc[result["approved"], "config"]],
             "results": result.to_dict(orient="records")})
    sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c11_lab")
    parser.add_argument("command", choices=["discover", "holdout"])
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args(argv)
    return discover(args.workers) if args.command == "discover" else holdout(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())

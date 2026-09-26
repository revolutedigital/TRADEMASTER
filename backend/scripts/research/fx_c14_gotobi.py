"""Round C14: Gotobi. Japanese importers pay on the 5th, 10th, ... of the month at the rate banks fix at 09:55 JST, and
banks buy dollars before the fix. Three fixed configurations on USDJPY (and EURJPY); discovery on 2012-2026 with the
max-t bootstrap over the three and a label placebo, then one run on the untouched 2003-2011 minutes.
See docs/forex/c14-preregistration.md.

    python -m scripts.research.fx_c14_gotobi discover
    python -m scripts.research.fx_c14_gotobi holdout

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_c13_carry as carry
from scripts.research import fx_fast_stats as stats

PREREGISTRATION = c3.DOCS / "c14-preregistration.md"
REGISTRY = c3.DOCS / "c14-registry.jsonl"
HOLIDAYS = c3.DOCS / "c14-japan-holidays.csv"  # Cabinet Office syukujitsu.csv (1955-2027), saved as UTF-8
DISCOVERY, HOLDOUT = ("2012-01-01", "2026-08-31"), ("2003-01-01", "2011-12-31")
POST_PUBLICATION = "2017-01-01"
GOTO = (5, 10, 15, 20, 25, 30)
HOUR = 3600
ENTRY_JST, EXIT_BAR_JST = 3 * HOUR, 9 * HOUR + 54 * 60  # 03:00 and the bar that starts at 09:54
REVERSAL_JST, REVERSAL_EXIT_BAR_JST = 9 * HOUR + 56 * 60, 11 * HOUR + 59 * 60
JST = 9 * HOUR
TOLERANCE, WEEKLY_OPEN_WAIT, CLOSED_GAP = 300, HOUR, 12 * HOUR
CONFIGS = ("A", "B", "C")
LEGS = {  # config -> (pair, side, window start, window exit bar, weight)
    "A": [("USDJPY", 1, ENTRY_JST, EXIT_BAR_JST, 1.0)],
    "B": [("USDJPY", 1, ENTRY_JST, EXIT_BAR_JST, 1.0), ("USDJPY", -1, REVERSAL_JST, REVERSAL_EXIT_BAR_JST, 1.0)],
    "C": [("USDJPY", 1, ENTRY_JST, EXIT_BAR_JST, 0.5), ("EURJPY", 1, ENTRY_JST, EXIT_BAR_JST, 0.5)],
}
CURRENCIES = {"USDJPY": ("USD", "JPY"), "EURJPY": ("EUR", "JPY")}
MARKUP = {"base": 0.01, "stress": 0.02}
BOOTSTRAP_SEED, PLACEBO_SEED, PLACEBO_COPIES = 20261031, 20261032, 20
SIGNIFICANCE, MIN_POSITIVE_YEARS_DISCOVERY, MIN_POSITIVE_YEARS_HOLDOUT = 0.05, 0.6, 5


# --- calendar -------------------------------------------------------------------------------------------------------


def japanese_holidays(path: Path = HOLIDAYS) -> set[pd.Timestamp]:
    frame = pd.read_csv(path, encoding="utf-8")
    return set(pd.to_datetime(frame.iloc[:, 0], format="%Y/%m/%d"))


def tokyo_business_days(start: str, end: str, holidays: set[pd.Timestamp]) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    year_end = (days.month == 12) & (days.day == 31) | (days.month == 1) & (days.day <= 3)
    return days[~year_end & ~days.isin(list(holidays))]


def gotobi_days(start: str, end: str, holidays: set[pd.Timestamp]) -> pd.DatetimeIndex:
    """Gotobi dates moved to the previous Tokyo business day, once each."""
    business = tokyo_business_days(pd.Timestamp(start) - pd.Timedelta(days=15), end, holidays)
    out = set()
    for month in pd.period_range(start, end, freq="M"):
        for day in GOTO:  # in February the last day stands for the 30th; the 31st never counts
            nominal = pd.Timestamp(month.year, month.month, min(day, month.days_in_month))
            position = business.searchsorted(nominal, side="right") - 1
            if position >= 0:
                out.add(business[position])
    return pd.DatetimeIndex(sorted(d for d in out if pd.Timestamp(start) <= d <= pd.Timestamp(end)))


# --- one leg on one day ---------------------------------------------------------------------------------------------


def entry_minute(times: np.ndarray, nominal: float) -> int | None:
    """The first minute within 5 minutes of `nominal`; if the market was shut then (the weekly open comes later),
    the first minute an hour after it reopens."""
    e = int(np.searchsorted(times, nominal))
    if e >= len(times):
        return None
    if times[e] - nominal <= TOLERANCE:
        return e
    if e > 0 and times[e] - times[e - 1] >= CLOSED_GAP:  # a weekend: wait an hour after the reopening
        later = int(np.searchsorted(times, times[e] + WEEKLY_OPEN_WAIT))
        return later if later < len(times) else None
    return None


def exit_minute(times: np.ndarray, nominal_bar: float) -> int | None:
    x = int(np.searchsorted(times, nominal_bar, side="right")) - 1
    return x if x >= 0 and nominal_bar - times[x] <= TOLERANCE else None


def rollover_days(fx_days: np.ndarray, e: int, x: int) -> int:
    """Calendar days of swap between two minutes: each 17:00 New York rollover counts 1, Wednesday's 3."""
    total = 0
    for day in range(int(fx_days[e]), int(fx_days[x])):
        weekday = pd.Timestamp(day * 86400, unit="s").dayofweek  # the FX day's date is its New York closing date
        total += 3 if weekday == 2 else 1
    return total


def leg_result(m1: np.ndarray, fx_days: np.ndarray, atr: np.ndarray, rates: pd.DataFrame, pair: str, side: int,
               start: float, exit_bar: float, cost: float) -> dict | None:
    times = m1[:, 0]
    e, x = entry_minute(times, start), exit_minute(times, exit_bar)
    if e is None or x is None or x <= e or not np.isfinite(atr[e]):
        return None
    entry = m1[e, 1]
    stop = entry - side * atr[e]
    exit_price = m1[x, 4]
    for i in range(e, x + 1):  # the stop first inside each minute; a gap fills at the open
        o, h, l = m1[i, 1], m1[i, 2], m1[i, 3]
        if (side == 1 and l <= stop) or (side == -1 and h >= stop):
            exit_price = min(stop, o) if side == 1 else max(stop, o)
            x = i
            break
    days = rollover_days(fx_days, e, x)
    base, quote = CURRENCIES[pair]
    day = pd.Timestamp(times[e], unit="s").normalize()
    differential = rates.at[day, base] - rates.at[day, quote]
    swap = {scenario: side * differential - markup for scenario, markup in MARKUP.items()}
    swap = {scenario: value * days / 365 * entry for scenario, value in swap.items()}
    swap["stress"] = min(swap["stress"], 0.0)  # the stress takes no credit
    gross = side * (exit_price - entry)
    return {"net": (gross - cost + swap["base"]) / atr[e], "stress": (gross - 2 * cost + swap["stress"]) / atr[e],
            "pips": (gross - cost + swap["base"]) / (0.01 if pair.endswith("JPY") else 0.0001)}


# --- a sample -------------------------------------------------------------------------------------------------------


def load_market(pair: str, manifest_path: Path, arrays: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m1 = c3.load_included(pair, pd.read_csv(manifest_path), arrays=Path(arrays))
    ctx = core.build_context(pair, m1, "H1")  # the daily ATR known before each minute (C11's convention)
    atr = np.full(len(m1), np.nan)
    atr[ctx.last_minute] = ctx.atr_d1
    return m1, core.minute_keys(m1[:, 0], 1440), pd.Series(atr).bfill().to_numpy()


def day_results(days: pd.DatetimeIndex, markets: dict, rates: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, Tokyo day): the day's net R, stress R and pips (legs summed, C's two markets weighted)."""
    rows = []
    for config in CONFIGS:
        for day in days:
            midnight_utc = (day - pd.Timedelta(seconds=JST)).timestamp()  # 00:00 JST of the day, in UTC
            legs = []
            for pair, side, start, exit_bar, weight in LEGS[config]:
                m1, fx_days, atr = markets[pair]
                result = leg_result(m1, fx_days, atr, rates, pair, side, midnight_utc + start, midnight_utc + exit_bar,
                                    c3.cost_of(pair))
                if result is None:
                    break
                legs.append({k: weight * v for k, v in result.items()})
            if len(legs) == len(LEGS[config]):
                rows.append({"config": config, "day": day, **{k: sum(leg[k] for leg in legs) for k in legs[0]}})
    return pd.DataFrame(rows)


def sample(manifest_path: Path, arrays: Path, start: str, end: str, rates_by_month: dict) -> tuple[dict, pd.DataFrame]:
    markets = {pair: load_market(pair, manifest_path, arrays) for pair in CURRENCIES}
    first = min(pd.Timestamp(m1[0, 0], unit="s") for m1, _, _ in markets.values()).normalize()
    last = max(pd.Timestamp(m1[-1, 0], unit="s") for m1, _, _ in markets.values()).normalize()
    all_days = pd.date_range(max(first, pd.Timestamp(start)) - pd.Timedelta(days=1), min(last, pd.Timestamp(end)))
    rates = carry.rate_panel(rates_by_month, all_days)
    return markets, rates


def summary(results: pd.DataFrame) -> dict:
    if not len(results):
        return {"days": 0, "mean_net": np.nan, "mean_stress": np.nan, "t": np.nan, "p_one_sided": 1.0}
    values = results["net"].to_numpy(dtype=float)
    t = values.mean() / (values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 and values.std() > 0 else np.nan
    years = results.groupby(results["day"].dt.year)["net"].mean()
    return {"days": len(values), "mean_net": float(values.mean()), "mean_stress": float(results["stress"].mean()),
            "mean_pips": float(results["pips"].mean()), "win_rate": float((values > 0).mean()), "t": float(t),
            "p_one_sided": float(norm.sf(t)) if np.isfinite(t) else 1.0, "positive_years": int((years > 0).sum()),
            "years": len(years)}


def placebo_days(real: pd.DatetimeIndex, business: pd.DatetimeIndex, seed: int) -> pd.DatetimeIndex:
    """As many random Tokyo business days in each month as the real gotobi days of that month."""
    rng = np.random.default_rng(seed)
    picked = []
    counts = pd.Series(1, index=real).groupby(real.to_period("M")).size()
    by_month = pd.Series(business, index=business).groupby(business.to_period("M"))
    for month, count in counts.items():
        pool = by_month.get_group(month).to_numpy()
        picked.extend(rng.choice(pool, size=min(count, len(pool)), replace=False))
    return pd.DatetimeIndex(sorted(picked))


# --- commands -------------------------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def _registered(*extra: Path) -> tuple[str, str]:
    if _git("status", "--porcelain", "--", "scripts", "app", str(PREREGISTRATION), str(REGISTRY), *map(str, extra)):
        raise RuntimeError("commit the code, the pre-registration and the registry before running")
    return _git("rev-parse", "HEAD"), hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()


def _events(kind: str) -> list[dict]:
    lines = REGISTRY.read_text(encoding="utf-8").splitlines() if REGISTRY.exists() else []
    return [event for event in map(json.loads, lines) if event.get("event") == kind]


def _append(event: dict) -> None:
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=float) + "\n")


def discover() -> int:
    commit, prereg = _registered(HOLIDAYS)
    if _events("c14_discovery"):
        raise RuntimeError("discovery already ran: its result is in the registry")
    holidays = japanese_holidays()
    markets, rates = sample(c3.MANIFEST, c3.ARRAYS, *DISCOVERY, carry.download_rates(cached_only=True))
    gotobi = gotobi_days(*DISCOVERY, holidays)
    business = tokyo_business_days(*DISCOVERY, holidays)
    real = day_results(gotobi, markets, rates)
    other = day_results(business.difference(gotobi), markets, rates)
    configs = list(CONFIGS)
    all_days = np.sort(real["day"].unique())
    day_index = np.searchsorted(all_days, real["day"].to_numpy())
    sums, counts = stats.daily_tables(day_index, real["net"].to_numpy(), real["config"].map(configs.index).to_numpy(),
                                      len(all_days), len(configs))
    _, p_maxt = stats.bootstrap_adjusted_p_values(sums, counts, seed=BOOTSTRAP_SEED)
    copies = []
    for copy in range(PLACEBO_COPIES):
        fake = day_results(placebo_days(gotobi, business, PLACEBO_SEED + copy), markets, rates)
        copies.append(fake.groupby("config")["net"].mean())
    placebo = pd.DataFrame(copies)
    rows = []
    for k, config in enumerate(configs):
        mine = real[real["config"] == config]
        s = summary(mine)
        late = summary(mine[mine["day"] >= POST_PUBLICATION])
        baseline = summary(other[other["config"] == config])
        reached = int((placebo[config] >= s["mean_net"]).sum())
        passes = bool(p_maxt[k] <= SIGNIFICANCE and s["mean_net"] > 0 and s["mean_stress"] > 0
                      and s["positive_years"] >= MIN_POSITIVE_YEARS_DISCOVERY * s["years"]
                      and s["mean_net"] > baseline["mean_net"] and late["mean_net"] > 0 and reached <= 1)
        rows.append({"config": config} | s | {"p_maxt": float(p_maxt[k]), "non_gotobi_mean": baseline["mean_net"],
                                              "mean_2017_on": late["mean_net"], "placebo_reached": reached,
                                              "passes": passes})
    table = pd.DataFrame(rows)
    by_year = real.pivot_table(index=real["day"].dt.year, columns="config", values="net", aggfunc="mean").round(3)
    text = (f"# C14: Gotobi, descoberta (2012-01 a 2026-08)\n\nCommit do código: {commit}. Dias gotobi: {len(gotobi):,}; "
            f"dias úteis de Tóquio sem gotobi: {len(business) - len(gotobi):,}. **Passam: "
            f"{int(table['passes'].sum())} de {len(table)}.** Unidade: R = 1 ATR diário, líquido de custo e swap.\n\n"
            f"```\n{c10._fmt(table)}\n```\n\nPor ano (R médio por dia):\n\n```\n{by_year.to_string()}\n```\n")
    (c3.DOCS / "c14-discovery.md").write_text(text, encoding="utf-8")
    _append({"date": date.today().isoformat(), "event": "c14_discovery", "code_commit": commit,
             "preregistration_sha256": prereg, "candidates": list(table.loc[table["passes"], "config"]),
             "results": table.to_dict(orient="records")})
    sys.stdout.write(text)
    return 0


def holdout() -> int:
    commit, prereg = _registered(HOLIDAYS, c10.HOLDOUT_MANIFEST)
    if _events("c14_holdout"):
        raise RuntimeError("the untouched period is used once: the C14 holdout already ran")
    discoveries = _events("c14_discovery")
    if not discoveries:
        raise RuntimeError("run discovery first")
    configs = list(discoveries[-1]["candidates"])
    if not configs:
        sys.stdout.write("no configuration passed discovery: the untouched data stays untouched\n")
        return 0
    markets, rates = sample(c10.HOLDOUT_MANIFEST, c10.HOLDOUT_ARRAYS, *HOLDOUT, carry.download_rates(cached_only=True))
    results = day_results(gotobi_days(*HOLDOUT, japanese_holidays()), markets, rates)
    table = pd.DataFrame([{"config": c} | summary(results[results["config"] == c]) for c in configs])
    table["p_holm"] = c3.holm(table["p_one_sided"].fillna(1.0).to_numpy())
    table["approved"] = ((table["p_holm"] <= SIGNIFICANCE) & (table["mean_net"] > 0) & (table["mean_stress"] > 0)
                         & (table["positive_years"] >= MIN_POSITIVE_YEARS_HOLDOUT))
    text = (f"# C14: Gotobi no período intocado (2003-01 a 2011-12)\n\nCommit do código: {commit}. **Aprovadas: "
            f"{int(table['approved'].sum())} de {len(table)}.**\n\n```\n{c10._fmt(table)}\n```\n")
    (c3.DOCS / "c14-holdout.md").write_text(text, encoding="utf-8")
    _append({"date": date.today().isoformat(), "event": "c14_holdout", "code_commit": commit,
             "preregistration_sha256": prereg, "approved": list(table.loc[table["approved"], "config"]),
             "results": table.to_dict(orient="records")})
    sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c14_gotobi")
    parser.add_argument("command", choices=["discover", "holdout"])
    return discover() if parser.parse_args(argv).command == "discover" else holdout()


if __name__ == "__main__":
    raise SystemExit(main())

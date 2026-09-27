"""Round C15: the FOMC-day premium in currencies (H1) and buying the S&P 500 at VIX spikes (H2), as pre-registered
in docs/forex/c15-preregistration.md. One run; its result goes to docs/forex/c15-registry.jsonl.

    python -m scripts.research.fx_c15_lab run

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_c13_carry as carry
from scripts.research import fx_c14_gotobi as c14

PREREGISTRATION = c3.DOCS / "c15-preregistration.md"
REGISTRY = c3.DOCS / "c15-registry.jsonl"
REPORT = c3.DOCS / "c15-report.md"
FOMC_DATES = c3.DOCS / "c15-fomc-dates.csv"
VIX_FILE = Path("data/c15/VIX_History.csv")
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
SPX_FILE = Path("data/c5/GSPC.csv")
NEW_YORK = ZoneInfo("America/New_York")
MAIN, INFORMATIVE = ("2014-01-01", "2026-08-31"), ("2003-01-01", "2013-12-31")
PLACEBO_COPIES, PLACEBO_SEED, PLACEBO_MAX = 20, 20261027, 1
SIGNIFICANCE = 0.05
VIX_RATIO, VIX_MEAN_DAYS, HOLD_DAYS = 1.25, 20, 10
SPX_COST_BP = {"base": 1.0, "stress": 2.0}
SPX_FINANCING = {"base": 0.06, "stress": 0.09}  # yearly, on the calendar days a long index position is carried
MIN_POSITIVE_YEARS = 0.6


# --- H1: the FOMC-day premium ---------------------------------------------------------------------------------------


def ny_time(day: pd.Timestamp, hour: int, minute: int = 0) -> float:
    return pd.Timestamp(day.year, day.month, day.day, hour, minute, tz=NEW_YORK).timestamp()


def previous_business_day(day: pd.Timestamp) -> pd.Timestamp:
    return day - pd.offsets.BDay(1)


def leg(m1: np.ndarray, fx_days: np.ndarray, atr: np.ndarray, rates: pd.DataFrame, pair: str, side: int,
        start: float, exit_bar: float) -> dict | None:
    """One leg, as C14's legs (entry within 5 minutes, exit at the close of `exit_bar`, a 1-ATR stop checked first
    in each minute, one swap per 17:00 New York rollover, Wednesday's three): its net and stress results in basis
    points of the notional and in R."""
    times = m1[:, 0]
    e, x = c14.entry_minute(times, start), c14.exit_minute(times, exit_bar)
    if e is None or x is None or x <= e or not np.isfinite(atr[e]):
        return None
    entry = m1[e, 1]
    stop = entry - side * atr[e]
    exit_price = m1[x, 4]
    touched = np.flatnonzero(m1[e:x + 1, 3] <= stop if side == 1 else m1[e:x + 1, 2] >= stop)
    if len(touched):  # the stop first inside each minute; a gap fills at the open
        x = e + int(touched[0])
        exit_price = min(stop, m1[x, 1]) if side == 1 else max(stop, m1[x, 1])
    days = c14.rollover_days(fx_days, e, x)
    base, quote = pair[:3], pair[3:]
    day = pd.Timestamp(times[e], unit="s").normalize()
    differential = rates.at[day, base] - rates.at[day, quote]
    swap = {scenario: (side * differential - markup) * days / 365 * entry for scenario, markup in c14.MARKUP.items()}
    swap["stress"] = min(swap["stress"], 0.0)
    gross, cost = side * (exit_price - entry), c3.cost_of(pair)
    move = {"net": gross - cost + swap["base"], "stress": gross - c3.STRESS_MULTIPLIER * cost + swap["stress"]}
    return {**{f"{k}_bp": v / entry * 1e4 for k, v in move.items()}, **{f"{k}_r": v / atr[e] for k, v in move.items()}}


def window_results(days: pd.DatetimeIndex, markets: dict, rates: pd.DataFrame) -> pd.DataFrame:
    """For every day D: long each currency whose 3-month rate is above the US one against the dollar, from 16:00
    New York on the business day before D to the close of D's 15:59 bar; the mean of the legs."""
    rows = []
    for day in days:
        eve = previous_business_day(day)
        start, exit_bar = ny_time(eve, 16), ny_time(day, 15, 59)
        chosen = [c for c in carry.CURRENCIES if rates.at[day.normalize(), c] > rates.at[day.normalize(), "USD"]]
        legs = []
        for currency in chosen:
            pair, sign = carry.PAIRS[currency]
            if pair not in markets:
                continue
            result = leg(*markets[pair], rates, pair, sign, start, exit_bar)
            if result is not None:
                legs.append(result)
        if legs:
            rows.append({"day": day, "legs": len(legs), **{k: float(np.mean([g[k] for g in legs])) for k in legs[0]}})
    return pd.DataFrame(rows)


def load_markets(manifest_path: Path, arrays: Path) -> dict:
    return {pair: c14.load_market(pair, manifest_path, arrays) for pair, _ in carry.PAIRS.values()}


def business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def t_test(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float(values.mean()) if len(values) else np.nan, np.nan, 1.0
    t = values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))
    return float(values.mean()), float(t), float(norm.sf(t))


def placebo(fomc: pd.DataFrame, others: pd.DataFrame, copies: int, seed: int) -> list[float]:
    """Means of `copies` draws: each FOMC day swapped for a non-FOMC day of the same year and weekday."""
    rng = np.random.default_rng(seed)
    pools = {key: group["net_bp"].to_numpy() for key, group in
             others.groupby([others["day"].dt.year, others["day"].dt.dayofweek])}
    wanted = fomc.groupby([fomc["day"].dt.year, fomc["day"].dt.dayofweek]).size()
    means = []
    for _ in range(copies):
        draws = [rng.choice(pools[key], size=min(count, len(pools[key])), replace=False)
                 for key, count in wanted.items() if key in pools]
        means.append(float(np.concatenate(draws).mean()))
    return means


def h1(rates_by_month: dict) -> dict:
    fomc_days = pd.DatetimeIndex(pd.read_csv(FOMC_DATES)["date"])
    rates = carry.rate_panel(rates_by_month, pd.date_range("2002-12-01", "2026-09-30"))
    samples = {"main": (MAIN, c3.MANIFEST, c3.ARRAYS),
               "informativo 2003-2011": (("2003-01-01", "2011-12-31"), c10.HOLDOUT_MANIFEST, c10.HOLDOUT_ARRAYS),
               "informativo 2012-2013": (("2012-01-01", "2013-12-31"), c3.MANIFEST, c3.ARRAYS)}
    out, loaded = {}, {}
    for name, ((start, end), manifest, arrays) in samples.items():
        if manifest not in loaded:
            loaded[manifest] = load_markets(manifest, arrays)
        markets = loaded[manifest]
        days = business_days(start, end)
        windows = window_results(days, markets, rates)
        is_fomc = windows["day"].isin(fomc_days)
        fomc, others = windows[is_fomc], windows[~is_fomc]
        same_weekday = others[others["day"].dt.dayofweek.isin(fomc["day"].dt.dayofweek.unique())]
        mean, t, p = t_test(fomc["net_bp"])
        years = fomc.groupby(fomc["day"].dt.year)["net_bp"].mean()
        out[name] = {"events": int(len(fomc)), "mean_net_bp": mean, "mean_stress_bp": float(fomc["stress_bp"].mean()),
                     "mean_net_r": float(fomc["net_r"].mean()), "t": t, "p_one_sided": p,
                     "other_days_bp": float(same_weekday["net_bp"].mean()), "other_days": int(len(same_weekday)),
                     "positive_years": int((years > 0).sum()), "years": int(len(years)),
                     "legs_per_event": float(fomc["legs"].mean())}
        if name == "main":
            copies = placebo(fomc, others, PLACEBO_COPIES, PLACEBO_SEED)
            out[name]["placebo_means_bp"] = copies
            out[name]["placebo_reaching"] = int(sum(m >= mean for m in copies))
    main = out["main"]
    main["approved_before_holm"] = bool(main["mean_net_bp"] > 0 and main["mean_stress_bp"] > 0
                                        and main["p_one_sided"] <= SIGNIFICANCE
                                        and main["mean_net_bp"] > main["other_days_bp"]
                                        and main["placebo_reaching"] <= PLACEBO_MAX)
    return out


# --- H2: the S&P 500 bought at VIX spikes -------------------------------------------------------------------------------


def load_vix() -> pd.Series:
    if not VIX_FILE.exists():
        import urllib.request

        VIX_FILE.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(VIX_URL, headers={"User-Agent": "Mozilla/5.0 (research; trademaster)"})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
            VIX_FILE.write_bytes(response.read())
    frame = pd.read_csv(VIX_FILE)
    return pd.Series(frame["CLOSE"].to_numpy(dtype=float), index=pd.to_datetime(frame["DATE"], format="%m/%d/%Y"))


def load_spx() -> pd.Series:
    frame = pd.read_csv(SPX_FILE)
    days = pd.to_datetime(frame["t"], unit="s", utc=True).dt.tz_convert(NEW_YORK).dt.tz_localize(None).dt.normalize()
    return pd.Series(frame["c"].to_numpy(dtype=float), index=days).groupby(level=0).last()


def spx_trades(spx: pd.Series, entries: np.ndarray) -> pd.DataFrame:
    """A 10-session long from each entry session's close, net of the round trip and of the financing."""
    rows = []
    days = spx.index
    for i in entries:
        j = i + HOLD_DAYS
        if j >= len(spx):
            continue
        gross = (spx.iloc[j] / spx.iloc[i] - 1) * 1e4
        carried = (days[j] - days[i]).days
        rows.append({"day": days[i], **{
            scenario: gross - SPX_COST_BP[scenario] - SPX_FINANCING[scenario] * carried / 365 * 1e4
            for scenario in ("base", "stress")}})
    return pd.DataFrame(rows)


def vix_signals(vix: pd.Series, spx: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Sessions between `start` and `end` (indices into `spx`) whose VIX close is at least VIX_RATIO times the mean of
    the 20 before, skipping those that come while a position is open (its exit session included)."""
    both = pd.concat({"vix": vix, "spx": spx}, axis=1, join="inner").dropna()
    prior = both["vix"].shift(1).rolling(VIX_MEAN_DAYS).mean()
    inside = (both.index >= start) & (both.index <= end)
    fired = np.flatnonzero((both["vix"] >= VIX_RATIO * prior).to_numpy() & inside)
    taken, busy_until = [], -1
    positions = spx.index.get_indexer(both.index[fired])
    for position in positions:
        if position > busy_until:
            taken.append(position)
            busy_until = position + HOLD_DAYS
    return np.array(taken, dtype=int)


def h2() -> dict:
    vix, spx = load_vix(), load_spx()
    start, end = pd.Timestamp("2003-01-01"), pd.Timestamp("2026-08-31")
    inside = (spx.index >= start) & (spx.index <= end)
    trades = spx_trades(spx, vix_signals(vix, spx, start, end))
    baseline = spx_trades(spx, np.flatnonzero(inside))
    mean, t, p = t_test(trades["base"].to_numpy())
    years = trades.groupby(trades["day"].dt.year)["base"].mean()
    out = {"trades": int(len(trades)), "mean_net_bp": mean, "mean_stress_bp": float(trades["stress"].mean()), "t": t,
           "p_one_sided": p, "baseline_bp": float(baseline["base"].mean()),
           "positive_years": int((years > 0).sum()), "years": int(len(years)),
           "vix_sha256": hashlib.sha256(VIX_FILE.read_bytes()).hexdigest()}
    out["approved_before_holm"] = bool(out["mean_net_bp"] > 0 and out["mean_stress_bp"] > 0
                                       and out["p_one_sided"] <= SIGNIFICANCE
                                       and out["mean_net_bp"] > out["baseline_bp"]
                                       and out["positive_years"] >= MIN_POSITIVE_YEARS * out["years"])
    return out


# --- the run ------------------------------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def _registered() -> tuple[str, str]:
    if _git("status", "--porcelain", "--", "scripts", "app", str(PREREGISTRATION), str(REGISTRY), str(FOMC_DATES)):
        raise RuntimeError("commit the code, the pre-registration, the registry and the dates before running")
    return _git("rev-parse", "HEAD"), hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()


def _events(kind: str) -> list[dict]:
    lines = REGISTRY.read_text(encoding="utf-8").splitlines() if REGISTRY.exists() else []
    return [e for e in map(json.loads, lines) if e.get("event") == kind]


def run() -> int:
    commit, prereg = _registered()
    if _events("c15_run"):
        raise RuntimeError("C15 already ran: its result is in the registry")
    first = h1(carry.download_rates(cached_only=True))
    second = h2()
    p_holm = c3.holm(np.array([first["main"]["p_one_sided"], second["p_one_sided"]]))
    first["main"]["p_holm"], second["p_holm"] = float(p_holm[0]), float(p_holm[1])
    first["main"]["approved"] = bool(first["main"]["approved_before_holm"] and p_holm[0] <= SIGNIFICANCE)
    second["approved"] = bool(second["approved_before_holm"] and p_holm[1] <= SIGNIFICANCE)
    event = {"date": date.today().isoformat(), "event": "c15_run", "code_commit": commit,
             "preregistration_sha256": prereg, "h1": first, "h2": second}
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=float) + "\n")
    fmt = lambda d: json.dumps(d, ensure_ascii=False, indent=1, default=lambda v: round(float(v), 4))  # noqa: E731
    text = (f"# C15: prêmio do dia de FOMC nas moedas (H1) e S&P 500 no pico do VIX (H2)\n\n"
            f"Commit do código: {commit}. Pré-registro: `docs/forex/c15-preregistration.md` (sha256 {prereg}).\n\n"
            f"**H1 aprovada: {'sim' if first['main']['approved'] else 'não'}. "
            f"H2 aprovada: {'sim' if second['approved'] else 'não'}.** Holm sobre as duas.\n\n"
            f"## H1 — dia de FOMC (principal: 2014-01 a 2026-08; o resto é informativo)\n\n```\n{fmt(first)}\n```\n\n"
            f"## H2 — S&P 500 no pico do VIX (2003-01 a 2026-08)\n\n```\n{fmt(second)}\n```\n")
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:] or [""])[0]
    if command != "run":
        sys.stderr.write("usage: python -m scripts.research.fx_c15_lab run\n")
        return 2
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

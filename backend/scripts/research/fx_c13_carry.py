"""Round C13, front C: interest-rate carry and trend in the seven currencies against the dollar, held for days or
weeks with the historical swap built from the OECD 3-month interbank rates on FRED. Four fixed strategies; discovery
on 2012-2026 with the max-t bootstrap over the four; the untouched 2003-2011 run is shared with front A and reads
this front's daily returns through `holdout_returns`. See docs/forex/c13-preregistration.md.

    python -m scripts.research.fx_c13_carry discover

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import urllib.request
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_fast_stats as stats

PREREGISTRATION = c3.DOCS / "c13-preregistration.md"
REGISTRY = c3.DOCS / "c13-registry.jsonl"
REPORT = c3.DOCS / "c13-c-discovery.md"
RATES = Path("data/c13/rates")

# currency -> (pair, sign of the pair's move when the currency rises against the dollar)
PAIRS = {"EUR": ("EURUSD", 1), "GBP": ("GBPUSD", 1), "AUD": ("AUDUSD", 1), "NZD": ("NZDUSD", 1),
         "JPY": ("USDJPY", -1), "CHF": ("USDCHF", -1), "CAD": ("USDCAD", -1)}
CURRENCIES = list(PAIRS)
SIGNS = pd.Series({k: float(sign) for k, (_, sign) in PAIRS.items()})
FRED_COUNTRY = {"USD": "US", "EUR": "EZ", "GBP": "GB", "JPY": "JP", "CHF": "CH", "AUD": "AU", "NZD": "NZ", "CAD": "CA"}
RATE_LAG_MONTHS = 2  # a day of month M uses month M-2: the value that was really published by then

MARKUP = {"base": 0.01, "stress": 0.02}  # yearly, per side, on top of the rate differential
COST_MULTIPLIER = {"base": 1.0, "stress": c3.STRESS_MULTIPLIER}
STRATEGIES = ("CARRY", "CARRY_TREND", "TSMOM", "DOLLAR_CARRY")
CARRY_LEGS, LEG_WEIGHT, TREND_DAYS, TSMOM_DAYS, VOL_DAYS = 2, 0.25, 63, 252, 60
DISCOVERY, HOLDOUT = ("2012-01-02", "2026-08-31"), ("2003-01-01", "2011-12-31")
BOOTSTRAP_SEED, SIGNIFICANCE, MIN_POSITIVE_YEARS = 20261021, 0.05, 0.6
ACCOUNT, EXPOSURES = 1000, (1, 3)


# --- data -----------------------------------------------------------------------------------------------------------


def parse_fred(text: str) -> pd.Series:
    """A FRED graph CSV -> monthly values (percent) indexed by month; '.' and empty cells are missing."""
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    if frame.shape[1] != 2:
        raise ValueError(f"not a FRED series: columns {list(frame.columns)}")
    series = pd.Series(frame.iloc[:, 1].astype(float).to_numpy(),
                       index=pd.PeriodIndex(pd.to_datetime(frame.iloc[:, 0]), freq="M")).sort_index()
    if not series.notna().any():
        raise ValueError(f"FRED series {frame.columns[1]} has no values")
    return series


def download_rates(out: Path = RATES, cached_only: bool = False) -> dict[str, pd.Series]:
    """Monthly 3-month interbank rate (percent) of every currency, downloaded once and then read from the cache."""
    out.mkdir(parents=True, exist_ok=True)
    rates = {}
    for currency, country in FRED_COUNTRY.items():
        name = f"IR3TIB01{country}M156N"
        path = out / f"{name}.csv"
        if not path.exists():
            if cached_only:
                raise RuntimeError(f"{path} is missing: the holdout must use the rate files of the discovery")
            request = urllib.request.Request(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}",
                                             headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
                text = response.read().decode("utf-8")
            parse_fred(text)  # an error page is never cached
            path.write_text(text, encoding="utf-8")
        rates[currency] = parse_fred(path.read_text(encoding="utf-8"))
    return rates


def rate_on(series: pd.Series, days: Iterable) -> np.ndarray:
    """Rate (fraction) known on each day: month M reads month M-2; a missing month, or one after the series ends,
    reads the last value before it."""
    known = series.dropna()
    wanted = pd.DatetimeIndex(days).to_period("M") - RATE_LAG_MONTHS
    position = np.searchsorted(known.index.asi8, wanted.asi8, side="right") - 1
    if (position < 0).any():
        raise ValueError(f"no rate at or before {wanted[position < 0][0]} (the series starts {known.index[0]})")
    return known.to_numpy()[position] / 100.0


def rate_panel(rates: dict[str, pd.Series], days: pd.DatetimeIndex) -> pd.DataFrame:
    """Rates known on each day: USD and the seven currencies."""
    return pd.DataFrame({k: rate_on(rates[k], days) for k in ["USD", *CURRENCIES]}, index=days)


def daily_closes(manifest_path: Path, arrays: Path) -> pd.DataFrame:
    """Close of the last minute of every FX day (17:00 New York), one column per currency, indexed by the FX day's
    date; weekday FX days only, NaN where the pair has no data."""
    manifest = pd.read_csv(manifest_path)
    columns = {}
    for currency, (pair, _) in PAIRS.items():
        m1 = c3.load_included(pair, manifest, arrays=Path(arrays))
        day = core.minute_keys(m1[:, 0], 1440)
        last = np.flatnonzero(np.diff(day, append=np.inf))
        columns[currency] = pd.Series(m1[last, 4], index=pd.to_datetime(day[last] * 86400, unit="s"))
    closes = pd.DataFrame(columns, columns=CURRENCIES).sort_index()
    closes.index.name = "date"
    return closes[closes.index.dayofweek < 5]


# --- returns --------------------------------------------------------------------------------------------------------


def _per_pair(frame: pd.DataFrame, transform) -> pd.DataFrame:
    """`transform` on each currency's own series of available days (lookbacks count that pair's observations)."""
    return pd.DataFrame({k: transform(frame[k].dropna()) for k in CURRENCIES}, columns=CURRENCIES).reindex(frame.index)


def excess_returns(closes: pd.DataFrame, rates: pd.DataFrame, markup: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long and short excess return of each currency against the dollar, from the pair's previous available close:
    the price move in the currency's direction plus the rate differential net of the markup, over calendar days."""
    long, short = {}, {}
    for k, (_, sign) in PAIRS.items():
        price = closes[k].dropna()
        move = sign * np.log(price / price.shift(1))
        years = price.index.to_series().diff().dt.days / 365.0
        differential = rates.loc[price.index, k] - rates.loc[price.index, "USD"]
        long[k] = move + (differential - markup) * years
        short[k] = -move + (-differential - markup) * years
    return (pd.DataFrame(long, columns=CURRENCIES).reindex(closes.index),
            pd.DataFrame(short, columns=CURRENCIES).reindex(closes.index))


def first_of(days: pd.DatetimeIndex, period: str) -> np.ndarray:
    """First FX day of each week (Monday to Sunday, as ISO weeks) with "W", of each month with "M". A sample that
    starts in the middle of a week or month waits for the next one (its first weekday is outside the sample)."""
    keys = days.to_period(period).asi8
    first = np.diff(keys, prepend=keys[:1] - 1) != 0
    if len(days) and (days[0] - pd.offsets.BDay(1)).to_period(period) == days[0].to_period(period):
        first[0] = False
    return first


def hold(targets: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Weights decided at decision closes, held until the next decision; flat before the first."""
    return targets.reindex(days).ffill().fillna(0.0)


# --- strategies (weights held after each day's close) ---------------------------------------------------------------


def carry_weights(closes: pd.DataFrame, rates: pd.DataFrame, with_trend: bool = False) -> pd.DataFrame:
    """CARRY: at the first FX day of each week, long the two highest rates and short the two lowest, 0.25 each, among
    the currencies with a close that day. CARRY_TREND: a leg only stays if its 63-day price move is in its favour."""
    days = closes.index[first_of(closes.index, "W")]
    trend = _per_pair(closes, lambda p: np.log(p / p.shift(TREND_DAYS))) * SIGNS if with_trend else None
    targets = pd.DataFrame(0.0, index=days, columns=CURRENCIES)
    for day in days:
        available = [k for k in CURRENCIES if pd.notna(closes.at[day, k])]
        ranked = list(rates.loc[day, available].sort_values(ascending=False, kind="stable").index)
        legs = min(CARRY_LEGS, len(ranked) // 2)
        sides = {**dict.fromkeys(ranked[:legs], LEG_WEIGHT), **dict.fromkeys(ranked[len(ranked) - legs:], -LEG_WEIGHT)}
        for k, weight in sides.items():
            if trend is None or weight * trend.at[day, k] > 0:  # a trend not yet measurable vetoes the leg
                targets.at[day, k] = weight
    return hold(targets, closes.index)


def tsmom_weights(closes: pd.DataFrame, long_excess: pd.DataFrame) -> pd.DataFrame:
    """TSMOM: at the first FX day of each month, the sign of each currency's 252-day excess return over its 60-day
    daily volatility, normalized to a gross exposure of 1."""
    days = closes.index[first_of(closes.index, "M")]
    momentum = _per_pair(long_excess, lambda r: r.rolling(TSMOM_DAYS).sum())
    volatility = _per_pair(closes, lambda p: np.log(p).diff().rolling(VOL_DAYS).std())
    raw = (np.sign(momentum.loc[days]) / volatility.loc[days]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = raw.abs().sum(axis=1)
    return hold(raw.div(total.where(total > 0), axis=0).fillna(0.0), closes.index)


def dollar_carry_weights(closes: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """DOLLAR_CARRY: at the first FX day of each month, long every currency with a close that day (equal weights,
    gross exposure 1) when their mean rate is above the dollar's, short every one otherwise."""
    days = closes.index[first_of(closes.index, "M")]
    available = closes.loc[days].notna()
    mean_rate = rates.loc[days, CURRENCIES].where(available).mean(axis=1)
    direction = np.where(mean_rate > rates.loc[days, "USD"], 1.0, -1.0)
    count = available.sum(axis=1)
    targets = available.astype(float).mul(direction, axis=0).div(count.where(count > 0), axis=0).fillna(0.0)
    return hold(targets, closes.index)


def strategy_weights(closes: pd.DataFrame, rates: pd.DataFrame, long_excess: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"CARRY": carry_weights(closes, rates), "CARRY_TREND": carry_weights(closes, rates, with_trend=True),
            "TSMOM": tsmom_weights(closes, long_excess), "DOLLAR_CARRY": dollar_carry_weights(closes, rates)}


def net_returns(weights: pd.DataFrame, long: pd.DataFrame, short: pd.DataFrame, closes: pd.DataFrame,
                cost_multiplier: float) -> pd.Series:
    """Daily return of a portfolio: the weights held after the previous close earn the day's excess returns, and
    every change of weight at a close pays half the pair's round-trip cost per unit changed, at that close's price."""
    # ponytail: a pair without a close on a decision day is closed at its last close and its move from there to its
    # next close is dropped; the only whole-month hole of the 7 pairs is AUDUSD 2003-02 (holdout warm-up)
    held = weights.shift(1).fillna(0.0)
    gross = (held.clip(lower=0) * long.fillna(0.0) - held.clip(upper=0) * short.fillna(0.0)).sum(axis=1)
    half_cost = pd.Series({k: c3.cost_of(pair) / 2 for k, (pair, _) in PAIRS.items()})
    turnover = weights.diff().fillna(weights).abs()
    cost = (turnover * closes.ffill().rdiv(half_cost, axis=1)).sum(axis=1)
    return gross - cost_multiplier * cost


def portfolio(closes: pd.DataFrame, rates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily net returns of the four strategies with normal costs ("base") and in the stress ("stress"), and each
    strategy's gross exposure held on each day ("gross")."""
    excess = {scenario: excess_returns(closes, rates, markup) for scenario, markup in MARKUP.items()}
    weights = strategy_weights(closes, rates, excess["base"][0])  # the stress changes the costs, not the decisions
    out = {scenario: pd.DataFrame({name: net_returns(w, *excess[scenario], closes, COST_MULTIPLIER[scenario])
                                   for name, w in weights.items()})
           for scenario in MARKUP}
    out["gross"] = pd.DataFrame({name: w.shift(1).fillna(0.0).abs().sum(axis=1) for name, w in weights.items()})
    return out


def sample_returns(manifest_path: Path, arrays: Path, start: str, end: str,
                   rates: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """The strategies on one sample: only its own FX days, lookbacks included."""
    closes = daily_closes(manifest_path, arrays).loc[start:end]
    return portfolio(closes, rate_panel(rates, closes.index))


def rate_hashes() -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(RATES.glob("*.csv"))}


def holdout_returns(names: Iterable[str]) -> pd.DataFrame:
    """Daily net returns of the named strategies on the untouched 2003-2011 FX days, columns (scenario, strategy);
    only with the very rate files the discovery registered."""
    names = list(names)
    discoveries = _events("c13_c_discovery")
    if not discoveries:
        raise RuntimeError("run front C's discovery first")
    rates = download_rates(cached_only=True)
    if rate_hashes() != discoveries[-1]["rates_sha256"]:
        raise RuntimeError("the rate files are not the ones the discovery used")
    returns = sample_returns(c10.HOLDOUT_MANIFEST, c10.HOLDOUT_ARRAYS, *HOLDOUT, rates)
    return pd.concat({scenario: frame[names] for scenario, frame in returns.items()}, axis=1)


# --- discovery ------------------------------------------------------------------------------------------------------


def days_per_year(index: pd.DatetimeIndex) -> float:
    """The sample's own number of FX days per year (about 260: every weekday)."""
    return len(index) / max((index[-1] - index[0]).days / 365.25, 1 / 365.25)


def summarize(base: pd.DataFrame, stress: pd.DataFrame, gross: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per strategy: mean daily net return, yearly figures, worst drawdown of the cumulative sum, share of positive
    calendar years, mean gross exposure, and the max-t bootstrap over the four."""
    years = base.groupby(base.index.year).sum()
    per_year = days_per_year(base.index)
    t, p_maxt = stats.bootstrap_adjusted_p_values(base.to_numpy(), np.ones(base.shape), seed=BOOTSTRAP_SEED)
    rows = []
    for j, name in enumerate(base.columns):
        daily = base[name]
        path = daily.cumsum()
        volatility = daily.std() * np.sqrt(per_year)
        rows.append({"strategy": name, "days": len(daily), "mean_daily": daily.mean(),
                     "mean_daily_stress": stress[name].mean(), "annual_return": daily.mean() * per_year,
                     "annual_vol": volatility, "mean_gross": float(gross[name].mean()) if gross is not None else np.nan,
                     "sharpe": daily.mean() * per_year / volatility if volatility > 0 else np.nan,
                     "max_drawdown": float((path.cummax().clip(lower=0) - path).max()),
                     "positive_years": float((years[name] > 0).mean()), "t": t[j], "p_maxt": p_maxt[j]})
    table = pd.DataFrame(rows)
    table["passes"] = ((table["p_maxt"] <= SIGNIFICANCE) & (table["mean_daily"] > 0) & (table["mean_daily_stress"] > 0)
                       & (table["positive_years"] >= MIN_POSITIVE_YEARS))
    return table


def write_report(table: pd.DataFrame, base: pd.DataFrame, rates: dict[str, pd.Series], commit: str) -> str:
    money = pd.DataFrame([{"strategy": row.strategy, "exposure": f"{exposure}x",
                           "annual_usd": row.annual_return * ACCOUNT * exposure,
                           "worst_drawdown_usd": row.max_drawdown * ACCOUNT * exposure}
                          for row in table.itertuples() for exposure in EXPOSURES])
    by_year = (base.groupby(base.index.year).sum() * 100).round(2)
    last = ", ".join(f"{k} {series.dropna().index[-1]}" for k, series in rates.items())
    text = (f"# C13 frente C: descoberta (2012-01 a 2026-08, 7 moedas contra o dólar)\n\n"
            "**C não é day trade:** as posições ficam dias ou semanas e pagam ou recebem o swap histórico.\n\n"
            f"Commit do código: {commit}. Dias FX: {len(base):,}. **Aprovadas: {int(table['passes'].sum())} de "
            f"{len(table)}** (p máx-t ≤ 0,05, média > 0 com custo normal e no estresse, positiva em ≥ 60% dos anos). "
            "Retorno diário em fração do capital, com exposição bruta de até 1 (soma de |peso| ≤ 1: pernas vetadas e o "
            "aquecimento ficam em 0; a média está em mean_gross), líquido do markup de 1% ao ano por lado e do custo "
            "de C3; no estresse, markup de 2% e custo dobrado. \"1x\" e \"3x\" multiplicam essa exposição.\n\n"
            f"```\n{c10._fmt(table)}\n```\n\n"
            f"## Em dinheiro, numa conta de US$ 1.000 (custo normal)\n\n```\n{c10._fmt(money)}\n```\n\n"
            f"## Por ano (soma dos retornos diários em %, custo normal)\n\n```\n{by_year.to_string()}\n```\n\n"
            f"Última observação de cada juro no FRED (depois dela, o valor segue o último): {last}.\n")
    REPORT.write_text(text, encoding="utf-8")
    return text


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def _registered() -> tuple[str, str]:
    if _git("status", "--porcelain", "--", "scripts", "app", str(PREREGISTRATION), str(REGISTRY)):
        raise RuntimeError("commit the code, the pre-registration and the registry before running")
    return _git("rev-parse", "HEAD"), hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()


def _events(kind: str) -> list[dict]:
    lines = REGISTRY.read_text(encoding="utf-8").splitlines() if REGISTRY.exists() else []
    return [event for event in map(json.loads, lines) if event.get("event") == kind]


def discover() -> int:
    commit, prereg = _registered()
    if _events("c13_c_discovery"):
        raise RuntimeError("discovery already ran: its result is in the registry")
    rates = download_rates()
    returns = sample_returns(c3.MANIFEST, c3.ARRAYS, *DISCOVERY, rates)
    table = summarize(returns["base"], returns["stress"], returns["gross"])
    text = write_report(table, returns["base"], rates, commit)
    rate_files = rate_hashes()
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"date": date.today().isoformat(), "event": "c13_c_discovery", "code_commit": commit,
                                 "preregistration_sha256": prereg, "rates_sha256": rate_files,
                                 "candidates": list(table.loc[table["passes"], "strategy"]),
                                 "results": table.to_dict(orient="records")}, sort_keys=True, default=float) + "\n")
    sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c13_carry")
    parser.add_argument("command", choices=["discover"])
    parser.parse_args(argv)
    return discover()


if __name__ == "__main__":
    raise SystemExit(main())

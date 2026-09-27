"""C13 front A, exploratory (NOT a pre-registered test, nothing here approves anything): do new features, known at the
signal bar's close, improve the banca's choice of trades? Each group of features is added to the frozen model's own
features and the same model (depth 3, min child weight 2000) is refitted by the discovery's walk-forward: each year
decided by a model trained only on trades that exited before it. The groups are judged on 2016-2022 and read on
2023-2026, which the judgement never saw; the frozen model is the reference.

    python -m scripts.research.fx_c13_features [group ...]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core
from scripts.research import fx_c13_banca as banca

RESULTS = banca.LAB / "features_walk_forward.parquet"
REPORT = c3.DOCS / "c13-a-features.md"
THRESHOLD = 0.05
YEARS = range(2016, 2027)


def usd_side(symbol: pd.Series, side: pd.Series) -> np.ndarray:
    """+1 when the trade is long the US dollar, -1 short, 0 when it has no dollar leg (a cross or a non-dollar
    index); the dollar index counts as the dollar."""
    base, quote = symbol.str[:3], symbol.str[3:]
    out = np.where(base == "USD", 1, np.where(quote == "USD", -1, 0)) * side.to_numpy()
    out[(symbol == "UDXUSD").to_numpy()] = side[(symbol == "UDXUSD").to_numpy()]
    index = ~symbol.str[:3].isin(["EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF"]).to_numpy()
    out[index & (symbol != "UDXUSD").to_numpy()] = 0
    return out.astype(float)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The new features, from the candidates that close their signal bar at the same moment (all known then)."""
    at = frame["decision_time"]
    frame["burst_all"] = frame.groupby(at)["set_id"].transform("size").astype(float)
    frame["burst_market"] = frame.groupby([at, frame["symbol"]])["set_id"].transform("size").astype(float)
    frame["burst_set"] = frame.groupby([at, frame["set_id"]])["symbol"].transform("size").astype(float)
    frame["usd_side"] = usd_side(frame["symbol"], frame["side"])
    dollar = frame["usd_side"].where(frame["usd_side"] != 0)
    frame["usd_breadth"] = dollar.groupby(at).transform("mean")
    frame["usd_count"] = dollar.notna().groupby(at).transform("sum").astype(float)
    frame["usd_aligned"] = (frame["usd_side"] * frame["usd_breadth"]).where(frame["usd_side"] != 0)
    return frame


VOL_DAYS, MOVE_DAYS = 60, 14  # past FX days behind each clock slot's usual volatility and usual move


def _slots(m1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every minute: the FX day (0-based) and the minute of that day counted from 17:00 New York."""
    ny_minute, _ = core.local_minutes(m1[:, 0], "America/New_York")
    day = core.minute_keys(m1[:, 0], 1440)
    return day - day.min(), (ny_minute - 17 * 60) % 1440


def _usual(grid: np.ndarray, days: int) -> np.ndarray:
    """The mean of each clock slot over the `days` FX days before each day (today excluded)."""
    return pd.DataFrame(grid).rolling(days, min_periods=days // 2).mean().shift(1).to_numpy()


def intraday_features(m1: np.ndarray, at: np.ndarray) -> dict[str, np.ndarray]:
    """At the minutes `at` (indices into m1, the signal bars' last minutes), each against the same clock time of the
    past FX days, all known at that minute:
    - sigma_left: the price movement still to come before the 17:00 New York exit, as a standard deviation in price
      (the usual variance of the slots left, scaled by how today's variance so far compares with the usual one,
      clipped to [0.25, 4]) — to be divided by the daily ATR;
    - vol_5m_surprise: the last 5 minutes' range over the usual one for those 5 minutes;
    - vol_day_surprise: today's summed minute ranges so far over the usual sum by this time;
    - move_z: the move since the FX day's first minute over the usual absolute move by this time (unsigned)."""
    day, slot = _slots(m1)
    n_days = int(day.max()) + 1
    grid = lambda values: _put(n_days, day, slot, values)  # noqa: E731
    close = m1[:, 4]
    step = np.diff(close, prepend=close[0]) ** 2
    ranges = m1[:, 2] - m1[:, 3]
    first = np.r_[True, np.diff(day) != 0]
    day_open = m1[np.maximum.accumulate(np.where(first, np.arange(len(m1)), 0)), 1]
    variance = grid(step)
    usual_variance = _usual(variance, VOL_DAYS)
    so_far = np.nancumsum(variance, axis=1)
    usual_so_far = np.nancumsum(usual_variance, axis=1)
    usual_left = np.nansum(usual_variance, axis=1, keepdims=True) - usual_so_far
    scale = np.clip(so_far / usual_so_far, 0.25, 4.0)
    sigma_left = np.sqrt(np.maximum(scale * usual_left, 0.0))
    range_grid = grid(ranges)
    five = pd.DataFrame(range_grid.T).rolling(5, min_periods=1).sum().T.to_numpy()
    summed = np.nancumsum(range_grid, axis=1)
    move = grid(np.abs(close - day_open))
    d, s = day[at], slot[at]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"sigma_left": sigma_left[d, s], "vol_5m_surprise": (five / _usual(five, VOL_DAYS))[d, s],
                "vol_day_surprise": (summed / _usual(summed, VOL_DAYS))[d, s],
                "move_z": (close[at] - day_open[at]) / _usual(move, MOVE_DAYS)[d, s]}


def _put(n_days: int, day: np.ndarray, slot: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.full((n_days, 1440), np.nan)
    grid[day, slot] = values
    return grid


def market_intraday(job: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """The intraday features of one market's candidates, at each signal bar's last minute."""
    symbol, rows = job
    m1 = c3.load_included(symbol, pd.read_csv(c3.MANIFEST))
    at = np.searchsorted(m1[:, 0], rows["entry_time"].to_numpy(), side="left") - 1
    values = intraday_features(m1, at)
    atr = c3.cost_of(symbol) / rows["cost_atr"].to_numpy()  # the daily ATR at the signal bar
    out = pd.DataFrame({"row": rows.index.to_numpy(), "sigma_left": values["sigma_left"] / atr,
                        "vol_5m_surprise": values["vol_5m_surprise"], "vol_day_surprise": values["vol_day_surprise"],
                        "move_z": values["move_z"] * rows["side"].to_numpy()})
    features = out.columns.drop("row")
    out[features] = out[features].where(np.isfinite(out[features]))  # x/0 (a slot that never moves) -> missing
    return out


def add_intraday(frame: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    jobs = [(symbol, group[["entry_time", "cost_atr", "side"]]) for symbol, group in frame.groupby("symbol")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(market_intraday, jobs))
    values = pd.concat(parts).set_index("row").sort_index()
    for column in values.columns:
        frame[column] = values[column].reindex(frame.index).to_numpy()
    return frame


GROUPS = {
    "rajada": ["burst_all", "burst_market", "burst_set"],
    "dolar": ["usd_side", "usd_breadth", "usd_count", "usd_aligned"],
    "rajada+dolar": ["burst_all", "burst_market", "burst_set", "usd_side", "usd_breadth", "usd_count", "usd_aligned"],
    "volatilidade": ["sigma_left", "vol_5m_surprise", "vol_day_surprise", "move_z"],
}
INTRADAY = {"sigma_left", "vol_5m_surprise", "vol_day_surprise", "move_z"}


def walk_forward(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """The discovery's walk-forward for the chosen configuration with `features`: one prediction per row (NaN
    outside the decided years)."""
    x = frame[features].to_numpy(dtype=np.float32)
    y = frame[banca.LABEL].to_numpy(dtype=float)
    out = np.full(len(frame), np.nan)
    for year in YEARS:
        train, test = banca.split(frame, year)
        if test.any():
            out[test] = banca.make_model(3, 2000).fit(x[train], y[train]).predict(x[test])
            sys.stdout.write(f"  {year}: {int(train.sum()):,} to train, {int(test.sum()):,} decided\n")
            sys.stdout.flush()
    return out


def score(frame: pd.DataFrame, prediction: np.ndarray) -> list[dict]:
    rows = []
    years = pd.to_datetime(frame["entry_time"], unit="s").dt.year.to_numpy()
    for period, (first, last) in {"2016-2022": (2016, 2022), "2023-2026": (2023, 2026)}.items():
        inside = (years >= first) & (years <= last) & np.isfinite(prediction)
        days = frame.loc[inside, "fx_day"].nunique()
        accepted = frame[inside & (prediction > THRESHOLD)]
        rows.append({"periodo": period} | banca.score(accepted, days))
    return rows


def main(argv: list[str] | None = None) -> int:
    groups = (argv or sys.argv[1:]) or list(GROUPS)
    frame = pd.read_parquet(banca.DATASET)
    frame = add_features(frame)
    if any(INTRADAY & set(GROUPS[g]) for g in groups):
        started = time.time()
        frame = add_intraday(frame)
        sys.stdout.write(f"intraday features: {time.time() - started:.0f} s\n")
    reference = pd.read_parquet(banca.PREDICTIONS, columns=["row", banca.config_name(3, 2000)])
    base = np.full(len(frame), np.nan)
    base[reference["row"].to_numpy()] = reference[banca.config_name(3, 2000)].to_numpy()
    results = pd.read_parquet(RESULTS) if RESULTS.exists() else pd.DataFrame(index=range(len(frame)))
    results["modelo atual"] = base
    for group in groups:
        started = time.time()
        sys.stdout.write(f"{group}: {GROUPS[group]}\n")
        results[group] = walk_forward(frame, banca.FEATURES + GROUPS[group])
        sys.stdout.write(f"  {time.time() - started:.0f} s\n")
        results.to_parquet(RESULTS)
    table = pd.DataFrame([{"grupo": name} | row for name in results.columns
                          for row in score(frame, results[name].to_numpy())])
    columns = ["grupo", "periodo", "trades", "mean_net", "mean_stress", "win_rate", "t", "market_share",
               "trades_per_day"]
    text = ("# C13, frente A: informações novas para o modelo (exploratório, não é teste pré-registrado)\n\n"
            "Cada grupo de informações é somado às do modelo congelado, e o mesmo modelo (profundidade 3, peso 2000) é "
            "retreinado pelo walk-forward da descoberta: cada ano decidido por um modelo que só viu operações "
            "encerradas antes dele. Operações aceitas com previsão acima de "
            f"{THRESHOLD:g}. Julga-se em 2016-2022 e lê-se em 2023-2026.\n\n"
            f"Grupos: `{json.dumps(GROUPS, ensure_ascii=False)}`\n\n"
            "```\n" + table[columns].to_string(index=False, float_format=lambda v: f"{v:,.4g}") + "\n```\n")
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

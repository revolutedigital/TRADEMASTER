"""Round C13, front A: the day-trade banca. The 34 family sets of C10 raise their hand in the 37 banca markets, all
with one fixed management (stop 1 daily ATR, target 3 R, to the end of the FX day); an XGBoost regressor reads what
is known at the close of the signal bar (clock, volatility, cost against the move, trend, the set's recent results)
and takes the trade when its predicted net R is above a threshold. Model and threshold are chosen by a walk-forward
over 2016-2026 inside the discovery data; the frozen model then meets the untouched 2003-2011 data once.
See docs/forex/c13-preregistration.md.

    python -m scripts.research.fx_c13_banca discover [--workers N]

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c6_lab as c6
from scripts.research import fx_c10_core as core
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_fast_stats as stats
from scripts.research.fx_c10_families import FAMILIES

PREREGISTRATION = c3.DOCS / "c13-preregistration.md"
REGISTRY = c3.DOCS / "c13-registry.jsonl"
REPORT = c3.DOCS / "c13-a-discovery.md"
LAB = Path("data/c13")
DATASET = LAB / "banca_discovery.parquet"
PREDICTIONS = LAB / "banca_walk_forward.parquet"
MODEL_FILE = LAB / "banca_model.json"

MANAGEMENT = "stop 1 ATR · alvo 3R · até o fim do dia"
CLASS_CODES = ("major", "cross", "metal", "index", "energy", "dollar_index")
YEARS = range(2016, 2027)
SAMPLE_END = pd.Timestamp("2026-09-01", tz="UTC").timestamp()  # the discovery data ends with 2026-08
MODEL_PARAMS = dict(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    random_state=20261020, n_jobs=-1)
GRID = ((3, 200), (3, 2000), (5, 200), (5, 2000))  # (max_depth, min_child_weight)
THRESHOLDS = (0.0, 0.05)
MIN_ACCEPTED, MIN_MARKET_TRADES, MIN_MARKET_SHARE = 3000, 20, 0.55
SIGNIFICANCE = 0.05 / (len(GRID) * len(THRESHOLDS))
RISK_PER_TRADE = 0.005

RECENT = {"recent_set_market": (["set_id", "symbol"], 20), "recent_set_all": (["set_id"], 200)}
DIRECTIONAL = ("from_day_open", "from_prev_close", "to_prev_high", "to_prev_low", "trend50", "ret_1", "ret_3",
               "ret_12", "rsi14", "boll_z")
FEATURES = ["set_id", "side", "market_class", "timeframe_minutes", "hour_exchange", "weekday", "minutes_from_open",
            "in_session", "cost_atr", "atr_ratio", "day_range_atr", "prev_range_atr", "from_day_open",
            "from_prev_close", "to_prev_high", "to_prev_low", "trend50", "ret_1", "ret_3", "ret_12", "rsi14", "boll_z",
            "bar_range_ratio", *RECENT]
LABEL = "net"


def management_row() -> tuple[np.ndarray, int]:
    grid, names = core.management_grid()
    if MANAGEMENT not in names:
        raise RuntimeError(f"the registered management row is missing from the C10 grid: {MANAGEMENT}")
    return grid, names.index(MANAGEMENT)


# --- the candidate trades and what is known at their signal bar ------------------------------------------------------


def bar_features(ctx: core.Context, cost: float) -> dict[str, np.ndarray]:
    """Every feature of the pre-registration at the close of each bar, before the side multiplies the directional
    ones. Only bars up to the bar itself and full FX days before its own are read."""
    atr = ctx.atr_d1
    d1 = ctx.d1
    atr100 = c3.atr(d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3], 100)
    before = ctx.bar_day - 1
    atr100_before = np.full(len(atr), np.nan)
    atr100_before[before >= 0] = atr100[before[before >= 0]]
    days = pd.DataFrame({"day": ctx.fx_day, "h": ctx.h, "l": ctx.l}).groupby("day", sort=False)
    day_range = days["h"].cummax().to_numpy() - days["l"].cummin().to_numpy()
    close = pd.Series(ctx.c)
    bar_range = pd.Series(ctx.h - ctx.l)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "hour_exchange": ctx.exchange_minute // 60, "weekday": core.weekday(ctx),
            "minutes_from_open": ctx.exchange_minute - ctx.exchange_open,
            "in_session": core.in_session(ctx, ctx.exchange_open, ctx.exchange_close),
            "cost_atr": cost / atr, "atr_ratio": atr / atr100_before, "day_range_atr": day_range / atr,
            "prev_range_atr": (ctx.prev_high - ctx.prev_low) / atr,
            "from_day_open": (ctx.c - ctx.day_open) / atr, "from_prev_close": (ctx.c - ctx.prev_close) / atr,
            "to_prev_high": (ctx.c - ctx.prev_high) / atr, "to_prev_low": (ctx.c - ctx.prev_low) / atr,
            "trend50": (ctx.prev_close - ctx.sma50_d1) / atr,
            **{f"ret_{k}": (close - close.shift(k)).to_numpy() / atr for k in (1, 3, 12)},
            "rsi14": c3.rsi(ctx.c, 14) - 50,
            "boll_z": ((close - close.rolling(20).mean()) / close.rolling(20).std(ddof=0)).to_numpy(),
            "bar_range_ratio": (bar_range / bar_range.rolling(20).mean().shift(1)).to_numpy(),
        }
    out = {name: np.asarray(values, dtype=float) for name, values in out.items()}
    return {name: np.where(np.isfinite(values), values, np.nan) for name, values in out.items()}  # 0/0, x/0 -> missing


def trade_rows(symbol: str, m1: np.ndarray) -> pd.DataFrame:
    """One row per candidate trade of every family set in one market, with its features at the signal bar
    (the recent-result features come later, from all markets: `add_recent`)."""
    grid, g = management_row()
    day_end = core.day_end_minutes(m1)
    fx_days = core.minute_keys(m1[:, 0], 1440)
    cost = c3.cost_of(symbol)
    market_class = CLASS_CODES.index(c6.instrument_class(symbol))
    contexts: dict[str, tuple[core.Context, dict[str, np.ndarray]]] = {}
    frames = []
    for set_id, (name, params) in enumerate(c10.family_sets()):
        timeframe = params["timeframe"]
        if timeframe not in contexts:
            ctx = core.build_context(symbol, m1, timeframe)
            contexts[timeframe] = ctx, bar_features(ctx, cost)
        ctx, features = contexts[timeframe]
        side, stop_ref = FAMILIES[name].signals(ctx, params)
        entries, sides, stops, atrs = core.entry_arrays(ctx, side, stop_ref, len(m1))
        trades = core.manage(m1, day_end, entries, sides, stops, atrs, cost, grid, g)
        if not len(trades):
            continue
        entry, exit_ = trades[:, 1].astype(np.int64), trades[:, 2].astype(np.int64)
        bar = np.searchsorted(ctx.last_minute, entry - 1)
        if not np.array_equal(ctx.last_minute[bar], entry - 1):
            raise RuntimeError(f"{symbol} {name} {params}: a trade does not follow its signal bar")
        trade_side = side[bar].astype(float)
        step = core.TIMEFRAME_MINUTES[timeframe] * 60
        # the signal bar's nominal close (bars are whole steps of the New York clock, whole hours from UTC)
        decision = np.ceil((m1[entry - 1, 0] + 60) / step) * step
        frame = pd.DataFrame({
            "symbol": symbol, "set_id": set_id, "side": trade_side, "decision_time": decision, "entry_time": m1[entry, 0],
            "exit_time": m1[exit_, 0], "fx_day": fx_days[entry], "net": trades[:, 3], "stress": trades[:, 4],
            "market_class": market_class, "timeframe_minutes": core.TIMEFRAME_MINUTES[timeframe],
        } | {column: values[bar] for column, values in features.items()})
        frame[list(DIRECTIONAL)] = frame[list(DIRECTIONAL)].mul(trade_side, axis=0)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def market_rows(job: tuple[str, str, str]) -> pd.DataFrame:
    symbol, manifest_path, arrays = job
    m1 = c3.load_included(symbol, pd.read_csv(manifest_path), arrays=Path(arrays))
    return trade_rows(symbol, m1)


def add_recent(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean net R of the last 20 trades of the same set and market, and of the last 200 of the same set in every
    market, among those whose exit minute is before the signal bar's close (NaN if none). Adds the columns in place."""
    decision, exit_, net = (frame[c].to_numpy(dtype=float) for c in ("decision_time", "exit_time", "net"))
    for column, (keys, window) in RECENT.items():
        values = np.full(len(frame), np.nan)
        for rows in frame.groupby(keys, sort=False).indices.values():
            values[rows] = c6.recent_mean(decision[rows], exit_[rows], net[rows], window)
        frame[column] = values
    return frame


def build_dataset(manifest_path: Path, arrays: Path, workers: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    included = set(manifest.loc[manifest["symbol_included"], "symbol"])
    jobs = [(s, str(manifest_path), str(arrays)) for s in c10.SYMBOLS if s in included]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = [frame for frame in pool.map(market_rows, jobs) if len(frame)]
    return add_recent(pd.concat(frames, ignore_index=True))


# --- model, walk-forward and scoring ---------------------------------------------------------------------------------


def make_model(depth: int, weight: int):
    from xgboost import XGBRegressor

    return XGBRegressor(**MODEL_PARAMS, max_depth=depth, min_child_weight=weight)


def config_name(depth: int, weight: int) -> str:
    return f"d{depth}_w{weight}"


def _utc(day: str) -> float:
    return pd.Timestamp(day, tz="UTC").timestamp()


def split(frame: pd.DataFrame, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Training rows (exited before the year starts) and decided rows (entered in the year, inside the sample)."""
    start, end = _utc(f"{year}-01-01"), min(_utc(f"{year + 1}-01-01"), SAMPLE_END)
    entry = frame["entry_time"].to_numpy()
    return frame["exit_time"].to_numpy() < start, (entry >= start) & (entry < end)


def walk_forward(frame: pd.DataFrame, grid=GRID, years=YEARS) -> pd.DataFrame:
    """Out-of-sample predictions: one row per decided trade (`row` = its position in `frame`), one column per model
    configuration; the thresholds are applied afterwards without refitting."""
    x = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame[LABEL].to_numpy(dtype=float)
    parts = []
    for year in years:
        train, test = split(frame, year)
        if not test.any():
            continue
        part = pd.DataFrame({"row": np.flatnonzero(test), "year": year})
        for depth, weight in grid:
            part[config_name(depth, weight)] = make_model(depth, weight).fit(x[train], y[train]).predict(x[test])
        parts.append(part)
        sys.stdout.write(f"walk-forward {year}: {int(train.sum()):,} to train, {int(test.sum()):,} decided\n")
        sys.stdout.flush()
    return pd.concat(parts, ignore_index=True)


def score(accepted: pd.DataFrame, days: int | None = None) -> dict:
    """Accepted trades (net, stress, fx_day, symbol): the pre-registered statistics. `days` = FX days of the period
    for the trades-per-day rate (default: the days with an accepted trade)."""
    n = len(accepted)
    if not n:
        return {"trades": 0, "mean_net": np.nan, "mean_stress": np.nan, "win_rate": np.nan, "t": np.nan,
                "p_one_sided": 1.0, "market_share": 0.0, "markets": 0, "trades_per_day": 0.0}
    all_days, day = np.unique(accepted["fx_day"].to_numpy(), return_inverse=True)
    sums, counts = stats.daily_tables(day, accepted["net"].to_numpy(dtype=float), np.zeros(n), len(all_days), 1)
    mean, _, t = stats.cluster_t(sums, counts)
    by_market = accepted.groupby("symbol")["net"].agg(["mean", "size"])
    eligible = by_market[by_market["size"] >= MIN_MARKET_TRADES]
    return {"trades": n, "mean_net": float(mean[0]), "mean_stress": float(accepted["stress"].mean()),
            "win_rate": float((accepted["net"] > 0).mean()), "t": float(t[0]),
            "p_one_sided": float(norm.sf(t[0])) if np.isfinite(t[0]) else 1.0,
            "market_share": float((eligible["mean"] > 0).mean()) if len(eligible) else 0.0, "markets": len(eligible),
            "trades_per_day": n / (days or len(all_days))}


def money_path(accepted: pd.DataFrame) -> dict:
    """Net R per calendar year of exit and the worst drawdown of the cumulative R, trades in exit order, in R and in
    US$ at 0.5% risk per trade on US$ 1,000 (informative)."""
    ordered = accepted.sort_values("exit_time")
    path = ordered["net"].cumsum().to_numpy()
    drawdown = float((np.maximum.accumulate(np.maximum(path, 0)) - path).max()) if len(path) else 0.0
    years = pd.to_datetime(ordered["exit_time"], unit="s").dt.year
    per_year = ordered.groupby(years.to_numpy())["net"].sum()
    usd = RISK_PER_TRADE * 1000
    return {"r_per_year": per_year.round(2).to_dict(), "mean_r_per_year": float(per_year.mean()) if len(per_year) else 0.0,
            "usd_per_year": float(per_year.mean() * usd) if len(per_year) else 0.0, "worst_drawdown_r": drawdown,
            "worst_drawdown_usd": drawdown * usd}


def combos(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    decided = frame.iloc[predictions["row"].to_numpy()]
    days = decided["fx_day"].nunique()
    rows = []
    for depth, weight in GRID:
        predicted = predictions[config_name(depth, weight)].to_numpy()
        for threshold in THRESHOLDS:
            rows.append({"max_depth": depth, "min_child_weight": weight, "threshold": threshold}
                        | score(decided[predicted > threshold], days))
    return pd.DataFrame(rows)


def checks(row: dict) -> dict[str, bool]:
    return {"média > 0": row["mean_net"] > 0, "média no estresse > 0": row["mean_stress"] > 0,
            "p ≤ 0,05 ÷ 8": row["p_one_sided"] <= SIGNIFICANCE,
            "≥ 55% dos mercados positivos": row["market_share"] >= MIN_MARKET_SHARE}


def choose(results: pd.DataFrame) -> tuple[dict | None, bool]:
    """The combination with the highest mean net R among those with 3,000+ accepted trades, and whether it passes."""
    eligible = results[results["trades"] >= MIN_ACCEPTED]
    if not len(eligible):
        return None, False
    chosen = eligible.loc[eligible["mean_net"].idxmax()].to_dict()
    return chosen, bool(all(checks(chosen).values()))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_model(frame: pd.DataFrame, depth: int, weight: int) -> str:
    """The final model: every trade that exited by 2026-08-31, saved; returns its sha256."""
    train = frame["exit_time"].to_numpy() < SAMPLE_END
    model = make_model(depth, weight).fit(frame.loc[train, FEATURES].to_numpy(dtype=np.float32),
                                          frame.loc[train, LABEL].to_numpy(dtype=float))
    LAB.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))
    return sha256(MODEL_FILE)


# --- commands --------------------------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def _registered() -> tuple[str, str]:
    if _git("status", "--porcelain", "--", "scripts", "app", str(PREREGISTRATION), str(REGISTRY)):
        raise RuntimeError("commit the code, the pre-registration and the registry before running")
    return _git("rev-parse", "HEAD"), sha256(PREREGISTRATION)


def _events(kind: str) -> list[dict]:
    lines = REGISTRY.read_text(encoding="utf-8").splitlines() if REGISTRY.exists() else []
    return [event for event in map(json.loads, lines) if event.get("event") == kind]


def _append(event: dict) -> None:
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=float) + "\n")


def discover(workers: int) -> int:
    commit, prereg = _registered()
    if _events("c13_a_discovery"):
        raise RuntimeError("discovery already ran: its result is in the registry")
    frame = build_dataset(c3.MANIFEST, c3.ARRAYS, workers)
    LAB.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATASET)
    predictions = walk_forward(frame)
    predictions.to_parquet(PREDICTIONS)
    results = combos(frame, predictions)
    chosen, passes = choose(results)
    model_hash = None
    by_year = pd.DataFrame()
    money = {}
    if chosen is not None:
        depth, weight = int(chosen["max_depth"]), int(chosen["min_child_weight"])
        model_hash = freeze_model(frame, depth, weight)
        accepted = predictions[predictions[config_name(depth, weight)] > chosen["threshold"]]
        taken = frame.iloc[accepted["row"].to_numpy()]
        by_year = taken.assign(year=accepted["year"].to_numpy()).groupby("year")["net"].agg(trades="size",
                                                                                         mean_net="mean")
        money = money_path(taken)
    baseline = score(frame.iloc[predictions["row"].to_numpy()])
    write_report(frame, results, chosen, passes, by_year, baseline, commit, model_hash, money)
    _append({"date": date.today().isoformat(), "event": "c13_a_discovery", "code_commit": commit,
             "preregistration_sha256": prereg, "chosen": chosen, "passes": passes,
             "results": results.to_dict(orient="records"), "model_sha256": model_hash, "money": money})
    return 0


def write_report(frame, results, chosen, passes, by_year, baseline, commit, model_hash, money=None) -> None:
    money = RISK_PER_TRADE * 1000
    if chosen is None:
        verdict = f"**Nenhuma combinação chegou a {MIN_ACCEPTED:,} operações aceitas: a frente A não passa.**"
    else:
        failed = [name for name, ok in checks(chosen).items() if not ok]
        verdict = (f"**Escolhida: profundidade {int(chosen['max_depth'])}, peso mínimo por folha "
                   f"{int(chosen['min_child_weight'])}, τ = {chosen['threshold']:g}. Passa para o período intocado: "
                   f"{'sim' if passes else 'não'}.**" + (f" Falhou em: {'; '.join(failed)}." if failed else ""))
    text = (
        f"# C13, frente A: banca de day trade, descoberta (walk-forward 2016 a 2026-08)\n\n"
        f"Commit do código: {commit}. Candidatas: {len(frame):,} operações de {frame['set_id'].nunique()} conjuntos "
        f"em {frame['symbol'].nunique()} mercados (2012-01 a 2026-08), todas com a gestão {MANAGEMENT}. Unidade: R "
        f"líquido do custo; 1 R = US$ {money:.0f} com 0,5% de risco numa conta de US$ 1.000.\n\n{verdict}\n\n"
        f"## As 8 combinações (operações aceitas de 2016 a 2026-08, cada ano decidido por um modelo treinado só com "
        f"operações que saíram antes dele)\n\n```\n{c10._fmt(results)}\n```\n\n"
        f"Informativo, todas as candidatas do mesmo período, sem banca: {json.dumps(baseline, default=float)}\n\n"
        f"## A escolhida, por ano\n\n"
        + (f"```\n{by_year.to_string(float_format=lambda v: f'{v:.4g}')}\n```\n\n" if len(by_year) else "Nenhuma.\n\n") +
        + (f"Em dinheiro (0,5% de risco por operação, US$ 1.000): {json.dumps(money, default=float)}\n\n" if money
           else "") +
        f"Modelo final (operações que saíram até 2026-08-31), sha256: {model_hash}.\n")
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def passed() -> bool:
    """Whether front A's registered discovery passed (the shared holdout only carries A when it did)."""
    events = _events("c13_a_discovery")
    if not events:
        raise RuntimeError("run front A's discovery first")
    return bool(events[-1]["passes"])


def holdout_accepted(workers: int) -> tuple[pd.DataFrame, int]:
    """The untouched period through the frozen banca: its candidate trades (features and recent results from the
    period itself) that the chosen threshold accepts, and the number of candidate FX days. Run once, by the shared
    C13 holdout, and only when `passed()`."""
    event = _events("c13_a_discovery")[-1]
    if not event["passes"]:
        raise RuntimeError("front A did not pass discovery: nothing goes to the untouched period")
    if sha256(MODEL_FILE) != event["model_sha256"]:
        raise RuntimeError("the frozen model is not the one in the registry")
    frame = build_dataset(c10.HOLDOUT_MANIFEST, c10.HOLDOUT_ARRAYS, workers)
    model = make_model(int(event["chosen"]["max_depth"]), int(event["chosen"]["min_child_weight"]))
    model.load_model(str(MODEL_FILE))
    frame["prediction"] = model.predict(frame[FEATURES].to_numpy(dtype=np.float32))
    accepted = frame[frame["prediction"] > event["chosen"]["threshold"]].reset_index(drop=True)
    return accepted, int(frame["fx_day"].nunique())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c13_banca")
    parser.add_argument("command", choices=["discover"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    return discover(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())

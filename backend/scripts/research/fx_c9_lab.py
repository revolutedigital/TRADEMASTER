"""Round C9: both bancas with new context features, tuned only by walk-forward inside 2019-2025, frozen for the demo.

See docs/forex/c9-preregistration.md. The entry banca decides which of the 14 strategies' signals to take (as in C6);
the exit banca decides, at each daily close of a winning position, whether to bank it now (as in C8, not for TSMOM).

    python -m scripts.research.fx_c9_lab run

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c6_lab as c6

CONFIGS = list(c3.STRATEGIES)
EXIT_CONFIGS = ("TURTLE20", "TURTLE55", "SQUEEZE_D1", "NR7_D1", "RSI2_D1", "CHANDELIER_D1", "CHANDELIER_H4")
EXCLUDED = frozenset({"XAUUSD", "XAGUSD", "WTIUSD", "BCOUSD"})  # minimum volume too big for the US$ 1,000 demo
SYMBOLS = tuple(s for s in c3.GROUP_A + c3.GROUP_B if s not in EXCLUDED)
CLASSES = ("major", "cross", "metal", "index", "energy", "dollar_index")
GRID = [{"max_depth": d, "n_estimators": n, "min_child_weight": w} for d in (3, 5) for n in (200, 500) for w in (50, 300)]
BASE_PARAMS = {"learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 20260925, "n_jobs": 8}
VALIDATION_YEARS = tuple(range(2019, 2026))
FINAL_END = pd.Timestamp("2026-09-01", tz="UTC").timestamp()
MODEL_DIR = Path("data/banca_demo")
DAILY = ["adx14", "trend_age_side", "bb_pct", "usdx_trend_side", "usdx_ret5_side", "corr_usdx60"]
ENTRY_FEATURES = c6.FEATURES + ["session"] + DAILY + ["streak", "stop_dist_atr", "target_dist_atr"]
EXIT_FEATURES = ["config_id", "side", "class_id", "profit", "best", "giveback", "days", "vol_d1", "trend50_side",
                 "trend200_side", "slope50_side", "rsi_side"] + DAILY + ["streak", "to_stop_atr", "to_target_atr"]


# --- context ------------------------------------------------------------------------------------------------------


def usd_exposure(symbol: str) -> int:
    """+1 if a long position is long the US dollar, -1 if short it, 0 if the dollar is not one of its legs."""
    if symbol == "UDXUSD" or symbol.startswith("USD"):
        return 1
    return -1 if len(symbol) == 6 and symbol.endswith("USD") and symbol not in c6.CLASSES else 0


def session(hour_new_york: np.ndarray) -> np.ndarray:
    """0 Asia (17:00-02:59), 1 London (03:00-07:59), 2 London-New York overlap (08:00-11:59), 3 New York afternoon."""
    h = np.asarray(hour_new_york)
    return np.select([(h >= 3) & (h < 8), (h >= 8) & (h < 12), (h >= 12) & (h < 17)], [1, 2, 3], default=0)


def adx(o, h, l, c, n: int = 14) -> np.ndarray:
    up, down = np.diff(h, prepend=np.nan), -np.diff(l, prepend=np.nan)
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    plus[0] = minus[0] = 0.0
    tr = c3.atr(o, h, l, c, 1)
    atr_n, plus_n, minus_n = c3.wilder(tr, n), c3.wilder(plus, n), c3.wilder(minus, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di, minus_di = 100 * plus_n / atr_n, 100 * minus_n / atr_n
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    out = np.full(dx.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(dx))
    if valid.size >= n:
        out[valid] = c3.wilder(dx[valid], n)
    return out


def run_length(condition: np.ndarray) -> np.ndarray:
    """How many bars in a row, up to each bar, the condition has held."""
    out = np.zeros(condition.shape, dtype=float)
    count = 0
    for i, holds in enumerate(condition):
        count = count + 1 if holds else 0
        out[i] = count
    return out


@dataclass
class Daily:
    """Per daily bar context of one instrument (index = daily bar)."""

    close_time: np.ndarray
    adx14: np.ndarray
    age_up: np.ndarray
    age_down: np.ndarray
    bb_pct: np.ndarray
    usdx_trend: np.ndarray
    usdx_ret5: np.ndarray
    corr_usdx60: np.ndarray


def daily_context(d1: np.ndarray, close_time: np.ndarray, usdx: pd.DataFrame) -> Daily:
    o, h, l, c = d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]
    sma50 = c3.sma(c, 50)
    std20, sma20 = pd.Series(c).rolling(20).std(ddof=0).to_numpy(), c3.sma(c, 20)
    width = 4 * std20 / sma20
    bb_pct = pd.Series(width).rolling(125).apply(lambda w: (w <= w[-1]).mean(), raw=True).to_numpy()
    at = np.searchsorted(usdx["close_time"].to_numpy(), close_time, side="right") - 1  # last USDX bar closed by then
    ok = at >= 0
    usdx_trend, usdx_ret5, usdx_close = (np.full(c.shape, np.nan) for _ in range(3))
    usdx_trend[ok] = usdx["trend"].to_numpy()[at[ok]]
    usdx_ret5[ok] = usdx["ret5"].to_numpy()[at[ok]]
    usdx_close[ok] = usdx["close"].to_numpy()[at[ok]]
    corr = pd.Series(np.diff(np.log(c), prepend=np.nan)).rolling(60).corr(
        pd.Series(np.diff(np.log(usdx_close), prepend=np.nan))).to_numpy()
    return Daily(close_time, adx(o, h, l, c), run_length(c > sma50), run_length(c < sma50), bb_pct,
                 usdx_trend, usdx_ret5, corr)


def usdx_table(manifest: pd.DataFrame) -> pd.DataFrame:
    m1 = c3.load_included("UDXUSD", manifest)
    last, d1 = c3.aggregate(m1, c3.bucket_keys(m1[:, 0], "D1"))
    o, h, l, c = d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]
    atr14 = c3.atr(o, h, l, c, 14)
    return pd.DataFrame({"close_time": m1[last, 0], "close": c, "trend": (c - c3.sma(c, 50)) / atr14,
                         "ret5": (c - pd.Series(c).shift(5).to_numpy()) / atr14})


def daily_features(daily: Daily, j: np.ndarray, sides: np.ndarray, exposure: int) -> dict[str, np.ndarray]:
    ok = j >= 0
    pick = lambda values: np.where(ok, values[np.maximum(j, 0)], np.nan)  # noqa: E731
    return {"adx14": pick(daily.adx14), "trend_age_side": np.where(sides > 0, pick(daily.age_up), pick(daily.age_down)),
            "bb_pct": pick(daily.bb_pct), "usdx_trend_side": pick(daily.usdx_trend) * exposure * sides,
            "usdx_ret5_side": pick(daily.usdx_ret5) * exposure * sides, "corr_usdx60": pick(daily.corr_usdx60)}


def streaks(results: np.ndarray) -> np.ndarray:
    """Signed run of wins (+) or losses (-) of the trades before each one (trades of one strategy and market do not
    overlap, so every earlier trade has closed by the next entry)."""
    out = np.zeros(results.shape)
    run = 0.0
    for i, value in enumerate(results):
        out[i] = run
        run = (run + 1 if run >= 0 else 1) if value > 0 else (run - 1 if run <= 0 else -1)
    return out


# --- building the tables --------------------------------------------------------------------------------------------


def initial_levels(plan: c3.Plan, j: np.ndarray, sides: np.ndarray, entries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stop = np.where(~np.isnan(plan.stop_level[j]), plan.stop_level[j], entries - sides * plan.stop_distance[j])
    target = entries + sides * plan.target_r * np.abs(entries - stop) if not np.isnan(plan.target_r) else np.full(j.shape, np.nan)
    return stop, target


def build(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usdx = usdx_table(manifest)
    trade_frames, point_rows = [], []
    for symbol in SYMBOLS:
        m1 = c3.load_included(symbol, manifest)
        keys = {tf: c3.bucket_keys(m1[:, 0], tf) for tf in c3.TIMEFRAMES}
        d1_last, d1 = c3.aggregate(m1, keys["D1"])
        daily = daily_context(d1, m1[d1_last, 0], usdx)
        cost, exposure = c3.cost_of(symbol), usd_exposure(symbol)
        bars = {tf: c3.aggregate(m1, keys[tf]) for tf in c3.TIMEFRAMES}
        for key, trades in c3.run_instrument(m1, keys, CONFIGS, cost).items():
            if not len(trades):
                continue
            sides, entries_i, exits_i = trades[:, 0], trades[:, 1].astype(np.int64), trades[:, 2].astype(np.int64)
            frame = c6.context(m1, keys, entries_i, sides)
            tf, build_plan = c3.STRATEGIES[key]
            last, tf_bars = bars[tf]
            plan = build_plan(tf_bars[:, 0], tf_bars[:, 1], tf_bars[:, 2], tf_bars[:, 3])
            signal_bar = np.maximum(c6.last_closed(last, entries_i), 0)
            stop, target = initial_levels(plan, signal_bar, sides, trades[:, 3])
            jd = c6.last_closed(d1_last, entries_i)
            for name, values in daily_features(daily, jd, sides, exposure).items():
                frame[name] = values
            frame["session"] = session(frame["hour"].to_numpy())
            frame["stop_dist_atr"] = np.abs(trades[:, 3] - stop) / trades[:, 5]
            frame["target_dist_atr"] = np.abs(target - trades[:, 3]) / trades[:, 5]
            frame["streak"] = streaks(trades[:, 7])
            frame["symbol"], frame["config"], frame["config_id"] = symbol, key, CONFIGS.index(key)
            frame["side"], frame["class_id"] = sides, CLASSES.index(c6.instrument_class(symbol))
            frame["cost_atr"] = cost / trades[:, 5]
            frame["entry_time"], frame["exit_time"] = m1[entries_i, 0], m1[exits_i, 0]
            frame["net"], frame["stress"] = trades[:, 7], trades[:, 8]
            frame["trade_key"] = [f"{symbol}|{key}|{int(t)}" for t in m1[entries_i, 0]]
            trade_frames.append(frame)
            if key in EXIT_CONFIGS:
                point_rows.extend(exit_points(trades, d1_last, d1, daily, cost, exposure, stop, target, frame, key, symbol))
    trades_table = pd.concat(trade_frames, ignore_index=True)
    for (config, symbol), part in trades_table.groupby(["config", "symbol"]):
        ordered = part.sort_values("entry_time")
        trades_table.loc[ordered.index, "recent_same"] = c6.recent_mean(
            ordered["entry_time"].to_numpy(), ordered["exit_time"].to_numpy(), ordered["net"].to_numpy(), 10)
    for config, part in trades_table.groupby("config"):
        trades_table.loc[part.index, "recent_config"] = c6.recent_mean(
            part["entry_time"].to_numpy(), part["exit_time"].to_numpy(), part["net"].to_numpy(), 50)
    return trades_table, pd.DataFrame(point_rows)


def exit_points(trades, d1_last, d1, daily: Daily, cost, exposure, stop, target, frame, key, symbol) -> list[dict]:
    o, h, l, c = d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]
    atr14, atr100 = c3.atr(o, h, l, c, 14), c3.atr(o, h, l, c, 100)
    sma50, sma200, rsi14 = c3.sma(c, 50), c3.sma(c, 200), c3.rsi(c, 14)
    slope = (sma50 - pd.Series(sma50).shift(5).to_numpy()) / atr14
    rows = []
    for number, (side, ei, xi, entry, _exit, unit, _reason, net, stress) in enumerate(trades):
        first = int(np.searchsorted(d1_last, int(ei), side="left"))
        last_bar = int(np.searchsorted(d1_last, int(xi), side="left"))
        best = -np.inf
        for j in range(first, last_bar):
            best = max(best, (((h[j] - entry) if side == 1 else (entry - l[j])) - cost) / unit)
            profit = (side * (c[j] - entry) - cost) / unit
            if profit <= 0:
                continue
            fill = o[j + 1] if j + 1 < len(o) else c[j]
            realized = (side * (fill - entry) - cost) / unit
            context = daily_features(daily, np.array([j]), np.array([side]), exposure)
            rows.append({"trade_key": frame["trade_key"].iat[number], "bar": j, "profit": profit, "realized": realized,
                         "realized_stress": (side * (fill - entry) - 2 * cost) / unit, "best": best,
                         "giveback": best - profit, "days": j - first + 1, "vol_d1": atr14[j] / atr100[j],
                         "trend50_side": side * (c[j] - sma50[j]) / atr14[j], "trend200_side": side * (c[j] - sma200[j]) / atr14[j],
                         "slope50_side": side * slope[j], "rsi_side": side * (rsi14[j] - 50),
                         "to_stop_atr": side * (c[j] - stop[number]) / unit,
                         "to_target_atr": side * (target[number] - c[j]) / unit,
                         "streak": frame["streak"].iat[number], "config_id": CONFIGS.index(key), "side": side,
                         "class_id": CLASSES.index(c6.instrument_class(symbol)), "remaining": net - realized,
                         "exit_time": frame["exit_time"].iat[number], "entry_time": frame["entry_time"].iat[number]}
                        | {k: float(v[0]) for k, v in context.items()})
    return rows


# --- walk-forward ---------------------------------------------------------------------------------------------------


def fit(features: list[str], table: pd.DataFrame, target: str, params: dict):
    from xgboost import XGBRegressor

    model = XGBRegressor(**BASE_PARAMS, **params)
    model.fit(table[features].to_numpy(dtype=float), table[target].to_numpy())
    return model


def apply_bancas(trades: pd.DataFrame, points: pd.DataFrame, entry_model, exit_model) -> pd.DataFrame:
    """The fold's trades with: taken by the entry banca, and their result with the exit banca."""
    out = trades.copy()
    out["taken"] = entry_model.predict(out[ENTRY_FEATURES].to_numpy(dtype=float)) > 0
    out["net_both"], out["stress_both"] = out["net"], out["stress"]
    mine = points[points["trade_key"].isin(out["trade_key"])].copy()
    if len(mine):
        mine["prediction"] = exit_model.predict(mine[EXIT_FEATURES].to_numpy(dtype=float))
        banked = mine[mine["prediction"] < 0].sort_values("bar").groupby("trade_key").first()
        index = out["trade_key"].map(lambda k: k in banked.index).to_numpy()
        keys = out.loc[index, "trade_key"]
        out.loc[index, "net_both"] = banked.loc[keys, "realized"].to_numpy()
        out.loc[index, "stress_both"] = banked.loc[keys, "realized_stress"].to_numpy()
    return out


def walk_forward(trades: pd.DataFrame, points: pd.DataFrame, params: dict) -> pd.DataFrame:
    folds = []
    for year in VALIDATION_YEARS:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp()
        end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC").timestamp(), c6.TEST_START)
        entry_model = fit(ENTRY_FEATURES, trades[trades["exit_time"] < start], "net", params)
        exit_model = fit(EXIT_FEATURES, points[points["exit_time"] < start], "remaining", params)
        fold = trades[(trades["entry_time"] >= start) & (trades["entry_time"] < end)]
        folds.append(apply_bancas(fold, points, entry_model, exit_model).assign(year=year))
    return pd.concat(folds, ignore_index=True)


def scores(frame: pd.DataFrame) -> dict:
    taken = frame[frame["taken"]]
    both = c6.summary(taken.assign(net=taken["net_both"], stress=taken["stress_both"]))
    return {"all": c6.summary(frame)["mean_atr"], "entry_only": c6.summary(taken)["mean_atr"],
            "both": both["mean_atr"], "both_stress": both["mean_atr_stress"], "both_t": both["t"],
            "taken": both["trades"], "win_rate_both": both["win_rate"]}


def run() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607
    if subprocess.run(["git", "status", "--porcelain", "--", "scripts", "app"], capture_output=True, text=True, check=True).stdout:  # noqa: S603, S607
        raise RuntimeError("commit the code before running")
    trades, points = build(pd.read_csv(c3.MANIFEST))
    rows = []
    for params in GRID:
        frame = walk_forward(trades, points, params)
        rows.append(params | scores(frame) | {"by_year": frame[frame["taken"]].groupby("year")["net_both"].mean().round(3).to_dict()})
        sys.stdout.write(f"{params}: {rows[-1]['both']:+.4f}\n")
        sys.stdout.flush()
    table = pd.DataFrame(rows)
    best = table.loc[table["both"].idxmax()]
    chosen = {k: int(best[k]) for k in ("max_depth", "n_estimators", "min_child_weight")}
    # informative only: the last year, already used by C6-C8, with models trained as C6/C8 did
    last_year = trades[(trades["entry_time"] >= c6.TEST_START) & (trades["entry_time"] < c6.TEST_END)]
    informative = scores(apply_bancas(last_year, points, fit(ENTRY_FEATURES, trades[trades["exit_time"] < c6.TEST_START], "net", chosen),
                                      fit(EXIT_FEATURES, points[points["exit_time"] < c6.TEST_START], "remaining", chosen)))
    # frozen for the demo: everything closed before the end of the sample
    entry_model = fit(ENTRY_FEATURES, trades[trades["exit_time"] < FINAL_END], "net", chosen)
    exit_model = fit(EXIT_FEATURES, points[points["exit_time"] < FINAL_END], "remaining", chosen)
    entry_path, exit_path = MODEL_DIR / "c9_entry_model.json", MODEL_DIR / "c9_exit_model.json"
    entry_model.save_model(str(entry_path))
    exit_model.save_model(str(exit_path))
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (entry_path, exit_path)}
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    report = c3.DOCS / "c9-report.md"
    report.write_text(
        f"# C9: as duas bancas com avaliadores novos (walk-forward 2019 a 2025-08)\n\nCommit do código: {commit}. "
        f"Operações: {len(trades):,}; pontos de decisão de saída: {len(points):,}.\n\n"
        f"**Escolhida:** {chosen} (maior média das duas bancas juntas no walk-forward). Modelos congelados: {digests}.\n\n"
        "Walk-forward (cada ano validado com modelo treinado só com o passado), resultado em ATR por operação:\n\n"
        f"```\n{table.to_string(index=False, float_format=fmt)}\n```\n\n"
        "Último ano (2025-09 a 2026-08), **só informativo** (já usado por C6, C7 e C8):\n\n"
        f"```\n{pd.Series(informative).to_string(float_format=fmt)}\n```\n\n"
        "A escolha de 1 entre 8 combinações pela melhor média infla um pouco o número do walk-forward; o juiz é a demo.\n",
        encoding="utf-8")
    doc = c3.DOCS / "c9-preregistration.md"
    with (c3.DOCS / "c9-registry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"date": date.today().isoformat(), "event": "c9_report", "code_commit": commit,
                                 "preregistration_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                                 "chosen": chosen, "walk_forward_best": float(best["both"]), "models": digests,
                                 "last_year_informative": informative}, sort_keys=True, default=float) + "\n")
    sys.stdout.write(report.read_text())
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["run"]:
        sys.stdout.write("usage: python -m scripts.research.fx_c9_lab run\n")
        raise SystemExit(2)
    raise SystemExit(run())

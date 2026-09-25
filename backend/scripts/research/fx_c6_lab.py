"""Round C6: the "banca" (meta-labeling) over the 14 published strategies of C3.

See docs/forex/c6-preregistration.md. The strategies raise their hand exactly as in C3; a gradient-boosting model
reads the context of every signal (volatility, trend, time, the strategy's own recent results, cost) and decides to
take it or skip it. Trained on entries up to 2025-08, tested once on 2025-09 to 2026-08.

    python -m scripts.research.fx_c6_lab run

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_fast_stats as stats

DOCS = c3.DOCS
TEST_START = pd.Timestamp("2025-09-01", tz="UTC").timestamp()
TEST_END = pd.Timestamp("2026-09-01", tz="UTC").timestamp()
SEEN_ON_B = ("RSI2_D1", "TSMOM")
MIN_ACCEPTED = 100
MODEL_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=50, random_state=20260925)
CLASSES = {"XAUUSD": "metal", "XAGUSD": "metal", "SPXUSD": "index", "NSXUSD": "index", "GRXEUR": "index",
           "UKXGBP": "index", "JPXJPY": "index", "FRXEUR": "index", "AUXAUD": "index", "HKXHKD": "index",
           "WTIUSD": "energy", "BCOUSD": "energy", "UDXUSD": "dollar_index"}
MAJORS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF")
FEATURES = ["config_id", "side", "class_id", "cost_atr", "vol_d1", "vol_h1", "trend50", "trend200", "slope50",
            "trend50_side", "trend200_side", "slope50_side", "rsi14", "ret4h_side", "hour", "weekday",
            "recent_same", "recent_config"]


def instrument_class(symbol: str) -> str:
    return CLASSES.get(symbol, "major" if symbol in MAJORS else "cross")


def last_closed(last: np.ndarray, minute_index: np.ndarray) -> np.ndarray:
    """Index of the last bar whose last minute is before each entry minute (-1 if none)."""
    return np.searchsorted(last, minute_index, side="left") - 1


def context(m1: np.ndarray, keys: dict[str, np.ndarray], entries: np.ndarray, sides: np.ndarray) -> pd.DataFrame:
    """Context features at each entry minute, from D1 and H1 bars closed before it."""
    last_d, d1 = c3.aggregate(m1, keys["D1"])
    last_h, h1 = c3.aggregate(m1, keys["H1"])
    return features_from_bars(d1, h1, last_closed(last_d, entries), last_closed(last_h, entries), m1[entries, 0], sides)


def features_from_bars(d1: np.ndarray, h1: np.ndarray, jd: np.ndarray, jh: np.ndarray, entry_times: np.ndarray,
                       sides: np.ndarray) -> pd.DataFrame:
    """The banca's context: D1 and H1 bars (open, high, low, close), the index of the last closed bar of each at every
    entry, the entry times (seconds, UTC) and the sides. The backtest and the live runner both call this."""
    od, hd, ld, cd = d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]
    oh, hh, lh, ch = h1[:, 0], h1[:, 1], h1[:, 2], h1[:, 3]
    atr14, atr100 = c3.atr(od, hd, ld, cd, 14), c3.atr(od, hd, ld, cd, 100)
    atr14h, atr100h = c3.atr(oh, hh, lh, ch, 14), c3.atr(oh, hh, lh, ch, 100)
    sma50, sma200, rsi14 = c3.sma(cd, 50), c3.sma(cd, 200), c3.rsi(cd, 14)
    slope = (sma50 - pd.Series(sma50).shift(5).to_numpy()) / atr14
    jd, jh = np.asarray(jd), np.asarray(jh)
    ok_d, ok_h = jd >= 0, jh >= 4

    def at(values, index, ok):
        out = np.full(index.shape, np.nan)
        out[ok] = values[index[ok]]
        return out

    trend50 = at((cd - sma50) / atr14, jd, ok_d)
    trend200 = at((cd - sma200) / atr14, jd, ok_d)
    slope50 = at(slope, jd, ok_d)
    ret4h = np.full(jh.shape, np.nan)
    ret4h[ok_h] = (ch[jh[ok_h]] - ch[jh[ok_h] - 4]) / atr14h[jh[ok_h]]
    local = pd.to_datetime(np.asarray(entry_times), unit="s", utc=True).tz_convert("America/New_York")
    return pd.DataFrame({
        "vol_d1": at(atr14 / atr100, jd, ok_d), "vol_h1": at(atr14h / atr100h, jh, ok_h),
        "trend50": trend50, "trend200": trend200, "slope50": slope50,
        "trend50_side": trend50 * sides, "trend200_side": trend200 * sides, "slope50_side": slope50 * sides,
        "rsi14": at(rsi14, jd, ok_d), "ret4h_side": ret4h * sides,
        "hour": local.hour.to_numpy(), "weekday": local.weekday.to_numpy(),
    })


def recent_mean(entry_times: np.ndarray, exit_times: np.ndarray, net: np.ndarray, window: int) -> np.ndarray:
    """Mean result of the last `window` trades already closed at each entry (nan if none)."""
    order = np.argsort(exit_times, kind="stable")
    closed_at, values = exit_times[order], net[order]
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    k = np.searchsorted(closed_at, entry_times, side="left")  # trades with exit strictly before the entry
    start = np.maximum(k - window, 0)
    count = k - start
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cumulative[k] - cumulative[start]) / count, np.nan)


def build_trades(manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    configs = list(c3.STRATEGIES)
    for symbol in c3.GROUP_A + c3.GROUP_B:
        m1 = c3.load_included(symbol, manifest)
        keys = {tf: c3.bucket_keys(m1[:, 0], tf) for tf in c3.TIMEFRAMES}
        cost = c3.cost_of(symbol)
        for key, trades in c3.run_instrument(m1, keys, configs, cost).items():
            if not len(trades):
                continue
            entries, sides = trades[:, 1].astype(np.int64), trades[:, 0]
            frame = context(m1, keys, entries, sides)
            frame["symbol"], frame["group"] = symbol, "A" if symbol in c3.GROUP_A else "B"
            frame["config"], frame["config_id"] = key, configs.index(key)
            frame["side"], frame["class_id"] = sides, ["major", "cross", "metal", "index", "energy",
                                                       "dollar_index"].index(instrument_class(symbol))
            frame["cost_atr"] = cost / trades[:, 5]
            frame["entry_time"], frame["exit_time"] = m1[entries, 0], m1[trades[:, 2].astype(np.int64), 0]
            frame["net"], frame["stress"] = trades[:, 7], trades[:, 8]
            frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["recent_same"] = np.nan
    table["recent_config"] = np.nan
    for (config, symbol), part in table.groupby(["config", "symbol"]):
        table.loc[part.index, "recent_same"] = recent_mean(part["entry_time"].to_numpy(), part["exit_time"].to_numpy(),
                                                           part["net"].to_numpy(), 10)
    for config, part in table.groupby("config"):
        table.loc[part.index, "recent_config"] = recent_mean(part["entry_time"].to_numpy(), part["exit_time"].to_numpy(),
                                                             part["net"].to_numpy(), 50)
    return table


def summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0}
    days = (frame["entry_time"] // 86400).to_numpy(dtype=np.int64)
    unique = np.unique(days)
    sums, counts = stats.daily_tables(np.searchsorted(unique, days), frame["net"].to_numpy(),
                                      np.zeros(len(frame), dtype=np.int64), len(unique), 1)
    mean, _, t = stats.cluster_t(sums, counts)
    from scipy.stats import norm

    wins = frame["net"] > 0
    return {"trades": len(frame), "mean_atr": float(mean[0]), "mean_atr_stress": float(frame["stress"].mean()),
            "t": float(t[0]), "p_one_sided": float(norm.sf(t[0])) if np.isfinite(t[0]) else 1.0,
            "win_rate": float(wins.mean()), "avg_win": float(frame.loc[wins, "net"].mean()) if wins.any() else np.nan,
            "avg_loss": float(frame.loc[~wins, "net"].mean()) if (~wins).any() else np.nan}


def run() -> int:
    from xgboost import XGBRegressor

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607
    if subprocess.run(["git", "status", "--porcelain", "--", "scripts", "app"], capture_output=True, text=True, check=True).stdout:  # noqa: S603, S607
        raise RuntimeError("commit the code before running")
    table = build_trades(pd.read_csv(c3.MANIFEST))
    train = table[table["exit_time"] < TEST_START]  # purged: no training label is realised inside the test year
    test = table[(table["entry_time"] >= TEST_START) & (table["entry_time"] < TEST_END)].copy()
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(train[FEATURES].to_numpy(dtype=float), train["net"].to_numpy())
    test["prediction"] = model.predict(test[FEATURES].to_numpy(dtype=float))
    test["accepted"] = test["prediction"] > 0
    main = test[(test["group"] == "B") & ~test["config"].isin(SEEN_ON_B)]
    accepted_main = main[main["accepted"]]
    s_all, s_acc = summary(main), summary(accepted_main)
    approved = bool(s_acc["trades"] >= MIN_ACCEPTED and s_acc["mean_atr"] > 0 and s_acc["mean_atr_stress"] > 0
                    and s_acc["p_one_sided"] <= 0.05 and s_acc["mean_atr"] > s_all["mean_atr"])
    rows = [{"conjunto": "principal: grupo B, 12 estratégias, sem banca"} | s_all,
            {"conjunto": "principal: grupo B, 12 estratégias, aceitas pela banca"} | s_acc,
            {"conjunto": "informativo: 41 mercados, 14 estratégias, sem banca"} | summary(test),
            {"conjunto": "informativo: 41 mercados, 14 estratégias, aceitas"} | summary(test[test["accepted"]]),
            {"conjunto": "informativo: treino (até 2025-08), sem banca"} | summary(train)]
    result = pd.DataFrame(rows)
    by_config = accepted_main.groupby("config")["net"].agg(["count", "mean"]).join(
        main.groupby("config")["net"].agg(total="count", mean_all="mean"))
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    report = DOCS / "c6-report.md"
    report.write_text(
        f"# C6: a banca (meta-labeling) no último ano\n\nCommit do código: {commit}. Treino: {len(train):,} operações "
        f"até 2025-08; teste: {len(test):,} operações de 2025-09 a 2026-08. **Aprovada: {approved}.**\n\n"
        f"Resultado em ATR por operação, líquido do custo.\n\n```\n{result.to_string(index=False, float_format=fmt)}\n```\n\n"
        f"Prova principal por estratégia (aceitas / todas):\n\n```\n{by_config.to_string(float_format=fmt)}\n```\n\n"
        f"Peso de cada avaliador no modelo:\n\n```\n{importance.to_string(float_format=fmt)}\n```\n", encoding="utf-8")
    doc = DOCS / "c6-preregistration.md"
    with (DOCS / "c6-registry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"date": date.today().isoformat(), "event": "c6_report", "code_commit": commit,
                                 "preregistration_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                                 "approved": approved, "main_accepted": s_acc, "main_all": s_all}, sort_keys=True) + "\n")
    LAB = c3.LAB
    LAB.mkdir(parents=True, exist_ok=True)
    model.save_model(str(LAB / "c6_model.json"))
    sys.stdout.write(report.read_text())
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["run"]:
        sys.stdout.write("usage: python -m scripts.research.fx_c6_lab run\n")
        raise SystemExit(2)
    raise SystemExit(run())

"""Round C8: the exit banca. At each daily close of a winning position, bank now or hold to the original exit?

See docs/forex/c8-preregistration.md.

    python -m scripts.research.fx_c8_exit_banca run

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

from scripts.research import fx_banca_demo as banca
from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c6_lab as c6

CONFIGS = ["TSMOM", "TURTLE20", "TURTLE55", "SQUEEZE_D1", "NR7_D1", "RSI2_D1", "CHANDELIER_D1", "CHANDELIER_H4"]
FEATURES = ["config_id", "side", "class_id", "profit", "best", "giveback", "days", "vol_d1", "trend50_side",
            "trend200_side", "slope50_side", "rsi_side"]
CLASSES = ["major", "cross", "metal", "index", "energy", "dollar_index"]
MIN_TRADES = 100


def decision_points(trades: np.ndarray, d1_last: np.ndarray, d1: np.ndarray, cost: float) -> list[dict]:
    """One row per daily close strictly after the entry and before the exit bar where the trade is in profit."""
    o, h, l, c = d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]
    atr14, atr100 = c3.atr(o, h, l, c, 14), c3.atr(o, h, l, c, 100)
    sma50, sma200, rsi14 = c3.sma(c, 50), c3.sma(c, 200), c3.rsi(c, 14)
    slope = (sma50 - pd.Series(sma50).shift(5).to_numpy()) / atr14
    rows = []
    for number, (side, ei, xi, entry, _exit, unit, _reason, net, stress) in enumerate(trades):
        first = int(np.searchsorted(d1_last, int(ei), side="left"))  # the daily bar in which the entry happened
        last_bar = int(np.searchsorted(d1_last, int(xi), side="left"))  # the bar in which the exit happens
        best = -np.inf
        for j in range(first, last_bar):
            favourable = (h[j] - entry) if side == 1 else (entry - l[j])
            best = max(best, (favourable - cost) / unit)
            profit = (side * (c[j] - entry) - cost) / unit
            if profit <= 0:
                continue
            fill = o[j + 1] if j + 1 < len(o) else c[j]  # banking executes at the next open, like every exit in C3
            realized = (side * (fill - entry) - cost) / unit
            realized_stress = (side * (fill - entry) - 2 * cost) / unit
            rows.append({"trade": number, "bar": j, "profit": profit, "realized": realized,
                         "best": best, "giveback": best - profit,
                         "days": j - first + 1, "vol_d1": atr14[j] / atr100[j], "trend50_side": side * (c[j] - sma50[j]) / atr14[j],
                         "trend200_side": side * (c[j] - sma200[j]) / atr14[j], "slope50_side": side * slope[j],
                         "rsi_side": side * (rsi14[j] - 50), "remaining": net - realized,
                         "remaining_stress": stress - realized_stress})
    return rows


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(c3.MANIFEST)
    trade_rows, point_rows = [], []
    for symbol in c3.GROUP_A + c3.GROUP_B:
        if symbol in banca.EXCLUDED:
            continue
        m1 = c3.load_included(symbol, manifest)
        keys = {tf: c3.bucket_keys(m1[:, 0], tf) for tf in c3.TIMEFRAMES}
        d1_last, d1 = c3.aggregate(m1, keys["D1"])
        cost = c3.cost_of(symbol)
        for key, trades in c3.run_instrument(m1, keys, CONFIGS, cost).items():
            base = len(trade_rows)
            for side, ei, xi, *_rest, net, stress in trades:
                trade_rows.append({"symbol": symbol, "config": key, "side": side, "entry_time": m1[int(ei), 0],
                                   "exit_time": m1[int(xi), 0], "net": net, "stress": stress})
            for point in decision_points(trades, d1_last, d1, cost):
                point["trade"] += base
                point["config_id"], point["side"] = CONFIGS.index(key), trades[point["trade"] - base, 0]
                point["class_id"] = CLASSES.index(c6.instrument_class(symbol))
                point["time"] = m1[d1_last[point["bar"]], 0]
                point_rows.append(point)
    return pd.DataFrame(trade_rows), pd.DataFrame(point_rows)


def run() -> int:
    from xgboost import XGBRegressor

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607
    if subprocess.run(["git", "status", "--porcelain", "--", "scripts", "app"], capture_output=True, text=True, check=True).stdout:  # noqa: S603, S607
        raise RuntimeError("commit the code before running")
    trades, points = build()
    points = points.join(trades[["exit_time", "entry_time"]], on="trade")
    train = points[points["exit_time"] < c6.TEST_START]
    model = XGBRegressor(**c6.MODEL_PARAMS)
    model.fit(train[FEATURES].to_numpy(dtype=float), train["remaining"].to_numpy())
    test_trades = trades[(trades["entry_time"] >= c6.TEST_START) & (trades["entry_time"] < c6.TEST_END)].copy()
    test_points = points[points["trade"].isin(test_trades.index)].copy()
    test_points["prediction"] = model.predict(test_points[FEATURES].to_numpy(dtype=float))
    banked = test_points[test_points["prediction"] < 0].sort_values("bar").groupby("trade").first()
    test_trades["net_exit"] = test_trades["net"]
    test_trades["stress_exit"] = test_trades["stress"]
    test_trades.loc[banked.index, "net_exit"] = banked["realized"]
    test_trades.loc[banked.index, "stress_exit"] = test_trades.loc[banked.index, "stress"] - banked["remaining_stress"]
    test_trades["gain"] = test_trades["net_exit"] - test_trades["net"]
    improvement = c6.summary(test_trades.assign(net=test_trades["gain"], stress=test_trades["gain"]))
    with_exit = c6.summary(test_trades.assign(net=test_trades["net_exit"], stress=test_trades["stress_exit"]))
    original = c6.summary(test_trades)
    approved = bool(with_exit["trades"] >= MIN_TRADES and with_exit["mean_atr"] > 0 and with_exit["mean_atr_stress"] > 0
                    and improvement["mean_atr"] > 0 and improvement["p_one_sided"] <= 0.05)
    # the entry banca of C6, trained exactly as there, on the same test trades
    table = c6.build_trades(pd.read_csv(c3.MANIFEST))
    entry_model = XGBRegressor(**c6.MODEL_PARAMS)
    fit = table[table["exit_time"] < c6.TEST_START]
    entry_model.fit(fit[c6.FEATURES].to_numpy(dtype=float), fit["net"].to_numpy())
    entry_test = table[(table["entry_time"] >= c6.TEST_START) & (table["entry_time"] < c6.TEST_END)].copy()
    entry_test["accepted"] = entry_model.predict(entry_test[c6.FEATURES].to_numpy(dtype=float)) > 0
    keys = ["symbol", "config", "entry_time"]
    test_trades = test_trades.reset_index().merge(entry_test[keys + ["accepted"]], on=keys, how="left").set_index("index")
    test_trades["accepted"] = test_trades["accepted"].fillna(False).astype(bool)
    both = test_trades[test_trades["accepted"]]
    s_entry = c6.summary(both)
    s_both = c6.summary(both.assign(net=both["net_exit"], stress=both["stress_exit"]))
    approved_both = bool(s_both["trades"] >= MIN_TRADES and s_both["mean_atr"] > 0 and s_both["mean_atr_stress"] > 0
                         and s_both["p_one_sided"] <= 0.05)
    by_config = test_trades.groupby("config").agg(operacoes=("net", "size"), original=("net", "mean"),
                                                  com_banca=("net_exit", "mean"))
    by_config["realizadas_antes"] = test_trades.assign(b=test_trades.index.isin(banked.index)).groupby("config")["b"].sum()
    rows = pd.DataFrame([{"conjunto": "saída original"} | original, {"conjunto": "com a banca de saída"} | with_exit,
                         {"conjunto": "melhora por operação"} | improvement,
                         {"conjunto": "só a banca de entrada"} | s_entry,
                         {"conjunto": "banca de entrada + banca de saída"} | s_both])
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    report = c3.DOCS / "c8-report.md"
    report.write_text(
        f"# C8: a banca de saída (último ano, 37 mercados)\n\nCommit do código: {commit}. **Banca de saída aprovada: "
        f"{approved}. Entrada + saída aprovadas: {approved_both}.** "
        f"Treino: {len(train):,} pontos de decisão; teste: {len(test_trades):,} operações, {len(banked):,} realizadas "
        f"antes pela banca. Resultado em ATR por operação, líquido do custo.\n\n"
        f"```\n{rows.to_string(index=False, float_format=fmt)}\n```\n\n```\n{by_config.to_string(float_format=fmt)}\n```\n",
        encoding="utf-8")
    doc = c3.DOCS / "c8-preregistration.md"
    with (c3.DOCS / "c8-registry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"date": date.today().isoformat(), "event": "c8_report", "code_commit": commit,
                                 "preregistration_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                                 "approved": approved, "approved_both": approved_both, "with_exit": with_exit,
                                 "improvement": improvement, "entry_and_exit": s_both},
                                sort_keys=True, default=float) + "\n")
    sys.stdout.write(report.read_text())
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["run"]:
        sys.stdout.write("usage: python -m scripts.research.fx_c8_exit_banca run\n")
        raise SystemExit(2)
    raise SystemExit(run())

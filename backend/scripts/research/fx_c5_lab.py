"""Round C5: TSMOM and RSI2 (rules of C3, unchanged) on spot equity indices, daily bars.

See docs/forex/c5-preregistration.md. Sample X: the indices C3/C4 saw, before 2012. Sample Y: indices the project
never used, 2000-2026. Signals at the daily close, fills at the next open, costs and financing charged per trade.

    python -m scripts.research.fx_c5_lab download
    python -m scripts.research.fx_c5_lab run

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_fast_stats as stats

DOCS = Path(__file__).resolve().parents[3] / "docs" / "forex"
DATA = Path("data/c5")
SAMPLE_X = {"symbols": ("^GSPC", "^NDX", "^GDAXI", "^N225", "^FCHI", "^AXJO", "^HSI", "^FTSE"),
            "start": "2000-01-01", "end": "2011-12-31"}
SAMPLE_Y = {"symbols": ("^RUT", "^IBEX", "^SSMI", "^AEX", "^GSPTSE", "^KS11", "^TWII", "^BVSP", "^MXX", "^STI"),
            "start": "2000-01-01", "end": "2026-08-31"}
CONFIGS = ("TSMOM", "RSI2_D1")
COST = {"base": 0.0005, "stress": 0.0010}
FINANCING = {"base": 0.03, "stress": 0.06}
MIN_YEARS, MAX_FLAT_OPEN_SHARE, MIN_TRADES, MIN_SHARE = 5, 0.20, 150, 0.55
YEAR_SECONDS = 365.0 * 86400


def download(symbol: str, out: Path = DATA) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(symbol)}"
           "?period1=915148800&period2=1790000000&interval=1d")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
        result = json.load(response)["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({"t": result["timestamp"], "o": quote["open"], "h": quote["high"], "l": quote["low"],
                          "c": quote["close"]})
    path = out / f"{symbol.lstrip('^')}.csv"
    frame.to_csv(path, index=False)
    return path


def load(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Valid days of the sample and the quality verdict for the index."""
    frame = pd.read_csv(DATA / f"{symbol.lstrip('^')}.csv")
    frame["day"] = pd.to_datetime(frame["t"], unit="s").dt.normalize()
    frame = frame[(frame["day"] >= start) & (frame["day"] <= end)]
    valid = frame[["o", "h", "l", "c"]].gt(0).all(axis=1) & (frame["h"] >= frame["l"])
    frame = frame[valid].drop_duplicates("day").sort_values("day").reset_index(drop=True)
    flat_open = float((frame["o"].to_numpy()[1:] == frame["c"].to_numpy()[:-1]).mean()) if len(frame) > 1 else 1.0
    years = (frame["day"].iloc[-1] - frame["day"].iloc[0]).days / 365.25 if len(frame) else 0.0
    verdict = {"symbol": symbol, "days": len(frame), "years": round(years, 2), "flat_open_share": round(flat_open, 4),
               "included": bool(years >= MIN_YEARS and flat_open <= MAX_FLAT_OPEN_SHARE)}
    return frame, verdict


def simulate_daily(frame: pd.DataFrame, plan: c3.Plan) -> list[tuple]:
    """(side, entry index, exit index) of every trade; decisions at a close, fills at the next open."""
    trades, side, entry = [], 0, 0
    n = len(frame)
    for i in range(n - 1):
        fill = i + 1
        if side != 0 and ((side == 1 and (plan.exit_long[i] or plan.enter[i] == -1))
                          or (side == -1 and (plan.exit_short[i] or plan.enter[i] == 1))):
            trades.append((side, entry, fill))
            side = 0
        if side == 0 and plan.enter[i] != 0 and not np.isnan(plan.unit[i]):
            side, entry = int(plan.enter[i]), fill
    if side != 0:
        trades.append((side, entry, n - 1))
    return trades


def trade_results(frame: pd.DataFrame, plan: c3.Plan, scenario: str) -> pd.DataFrame:
    o, c, t = frame["o"].to_numpy(), frame["c"].to_numpy(), frame["t"].to_numpy(dtype=float)
    rows = []
    for side, entry, exit_ in simulate_daily(frame, plan):
        entry_price = o[entry]
        exit_price = o[exit_] if exit_ > entry else c[exit_]
        held = (t[exit_] - t[entry]) / YEAR_SECONDS
        net = side * (exit_price - entry_price) - COST[scenario] * entry_price - FINANCING[scenario] * held * entry_price
        rows.append({"entry_time": t[entry], "side": side, "net": net / plan.unit[entry - 1]})
    return pd.DataFrame(rows, columns=["entry_time", "side", "net"])


def evaluate(sample: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verdicts, per_config = [], {k: [] for k in CONFIGS}
    for symbol in sample["symbols"]:
        frame, verdict = load(symbol, sample["start"], sample["end"])
        verdicts.append(verdict)
        if not verdict["included"]:
            continue
        o, h, l, cl = (frame[k].to_numpy(dtype=float) for k in ("o", "h", "l", "c"))
        for key in CONFIGS:
            plan = c3.STRATEGIES[key][1](o, h, l, cl)
            base, stress = trade_results(frame, plan, "base"), trade_results(frame, plan, "stress")
            base["stress"] = stress["net"].to_numpy()
            base["symbol"] = symbol
            per_config[key].append(base)
    frames = {k: pd.concat(v, ignore_index=True) for k, v in per_config.items()}
    days = np.unique(np.concatenate([(f["entry_time"] // 86400).to_numpy(dtype=np.int64) for f in frames.values()]))
    ids = np.concatenate([np.full(len(frames[k]), i) for i, k in enumerate(CONFIGS)])
    day_index = np.concatenate([np.searchsorted(days, (frames[k]["entry_time"] // 86400).to_numpy(dtype=np.int64))
                                for k in CONFIGS])
    values = np.concatenate([frames[k]["net"].to_numpy() for k in CONFIGS])
    sums, counts = stats.daily_tables(day_index, values, ids, len(days), len(CONFIGS))
    mean, _, t = stats.cluster_t(sums, counts)
    from scipy.stats import norm

    p = c3.holm(norm.sf(np.where(np.isfinite(t), t, -np.inf)))
    rows, by_symbol = [], {}
    for k, key in enumerate(CONFIGS):
        f = frames[key]
        means = f.groupby("symbol")["net"].mean()
        share = float((means > 0).mean())
        wins = f["net"] > 0
        approved = bool(mean[k] > 0 and f["stress"].mean() > 0 and p[k] <= 0.05 and share >= MIN_SHARE
                        and len(f) >= MIN_TRADES)
        rows.append({"config": key, "trades": len(f), "mean_atr": mean[k], "mean_atr_stress": f["stress"].mean(),
                     "t": t[k], "p_holm": p[k], "win_rate": wins.mean(), "avg_win": f.loc[wins, "net"].mean(),
                     "avg_loss": f.loc[~wins, "net"].mean(), "indices_positive": share, "approved": approved})
        by_symbol[key] = f.groupby("symbol")["net"].agg(["count", "mean"]).apply(
            lambda r: f"{int(r['count'])} / {r['mean']:+.3f}", axis=1)
    return pd.DataFrame(rows), pd.DataFrame(verdicts), pd.DataFrame(by_symbol)


def run() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607
    if subprocess.run(["git", "status", "--porcelain", "--", "scripts"], capture_output=True, text=True, check=True).stdout:  # noqa: S603, S607
        raise RuntimeError("commit the code before running")
    sections, approved = [], {}
    for name, sample in (("X (índices já vistos, 2000-2011)", SAMPLE_X), ("Y (índices novos, 2000-2026)", SAMPLE_Y)):
        table, verdicts, by_symbol = evaluate(sample)
        approved[name] = set(table.loc[table["approved"], "config"])
        fmt = lambda v: f"{v:.4g}"  # noqa: E731
        sections.append(f"## Amostra {name}\n\n```\n{table.to_string(index=False, float_format=fmt)}\n```\n\n"
                        f"Qualidade:\n\n```\n{verdicts.to_string(index=False)}\n```\n\n"
                        f"Por índice (operações / média em ATR):\n\n```\n{by_symbol.to_string()}\n```\n")
    passed = sorted(set.intersection(*approved.values()))
    report = DOCS / "c5-report.md"
    report.write_text(f"# C5: TSMOM e RSI2 em índices de ações\n\nCommit do código: {commit}. Aprovadas nas duas "
                      f"amostras: {passed or 'nenhuma'}.\n\n" + "\n".join(sections), encoding="utf-8")
    doc = DOCS / "c5-preregistration.md"
    with (DOCS / "c5-registry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"date": date.today().isoformat(), "event": "c5_report", "code_commit": commit,
                                 "preregistration_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                                 "approved": passed}, sort_keys=True) + "\n")
    sys.stdout.write(report.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["download", "run"])
    args = parser.parse_args(argv)
    if args.command == "download":
        for symbol in SAMPLE_X["symbols"] + SAMPLE_Y["symbols"]:
            path = download(symbol)
            sys.stdout.write(f"{symbol}: {sum(1 for _ in path.open()) - 1} dias\n")
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

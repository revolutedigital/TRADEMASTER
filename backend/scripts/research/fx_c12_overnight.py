"""Round C12: out-of-sample confirmation of the US overnight drift on SPY and QQQ, 2000-2011 (Yahoo daily).
See docs/forex/c12-preregistration.md.

    python -m scripts.research.fx_c12_overnight confirm

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
from scipy.stats import norm

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c5_lab as c5
from scripts.research import fx_fast_stats as stats

PREREGISTRATION = c3.DOCS / "c12-preregistration.md"
REGISTRY = c3.DOCS / "c12-registry.jsonl"
SYMBOLS = ("SPY", "QQQ")
START, END = "2000-01-03", "2011-12-30"
COSTS = {"base": 0.0001, "stress": 0.0002}  # round trip per night, as a fraction of price
MAX_ABS_RETURN, MIN_POSITIVE_YEARS, SIGNIFICANCE = 0.20, 7, 0.05


def overnight(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Each day's open over the previous row's close, inside the sample; data errors (|r| > 20%) dropped."""
    frame = frame.dropna(subset=["o", "c"])
    frame = frame[(frame["o"] > 0) & (frame["c"] > 0)].copy()
    frame["day"] = pd.to_datetime(frame["t"], unit="s").dt.normalize()
    frame = frame.drop_duplicates("day").sort_values("day")
    frame["r"] = frame["o"] / frame["c"].shift(1) - 1
    frame = frame[(frame["day"] >= START) & (frame["day"] <= END)].dropna(subset=["r"])
    return frame.loc[frame["r"].abs() <= MAX_ABS_RETURN, ["day", "r"]].assign(symbol=symbol)


def summary(nights: pd.DataFrame, cost: float) -> dict:
    net = nights["r"].to_numpy() - cost
    days, index = np.unique(nights["day"].to_numpy(), return_inverse=True)
    sums, counts = stats.daily_tables(index, net, np.zeros(len(net)), len(days), 1)
    mean, _, t = stats.cluster_t(sums, counts)
    return {"nights": len(net), "mean_bp": float(mean[0]) * 1e4, "t": float(t[0]), "p_one_sided": float(norm.sf(t[0])),
            "win_rate": float((net > 0).mean())}


def confirm() -> int:
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731, S603, S607
    if git("status", "--porcelain", "--", "scripts", "app", str(PREREGISTRATION), str(REGISTRY)):
        raise RuntimeError("commit the code, the pre-registration and the registry before running")
    lines = REGISTRY.read_text(encoding="utf-8").splitlines()
    if any(json.loads(line).get("event") == "c12_confirmation" for line in lines):
        raise RuntimeError("the confirmation runs once: it is already in the registry")
    commit, prereg = git("rev-parse", "HEAD"), hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    nights = pd.concat([overnight(pd.read_csv(c5.download(symbol)), symbol) for symbol in SYMBOLS], ignore_index=True)
    base, stress = summary(nights, COSTS["base"]), summary(nights, COSTS["stress"])
    by_symbol = {s: summary(nights[nights["symbol"] == s], COSTS["base"]) for s in SYMBOLS}
    by_year = ((nights.assign(year=nights["day"].dt.year).groupby("year")["r"].mean() - COSTS["base"]) * 1e4).round(2)
    confirmed = bool(stress["p_one_sided"] <= SIGNIFICANCE and base["mean_bp"] > 0 and stress["mean_bp"] > 0
                     and all(v["mean_bp"] > 0 for v in by_symbol.values())
                     and int((by_year > 0).sum()) >= MIN_POSITIVE_YEARS)
    table = pd.DataFrame([{"conjunto": "custo 1 pb"} | base, {"conjunto": "custo 2 pb"} | stress]
                         + [{"conjunto": f"{s}, custo 1 pb"} | v for s, v in by_symbol.items()])
    text = (f"# C12: deriva noturna, SPY e QQQ, 2000 a 2011\n\nCommit do código: {commit}. **Confirmada: {confirmed}.** "
            f"Retorno noturno (abertura sobre o fechamento da véspera), líquido, em pontos-base por noite.\n\n"
            f"```\n{table.to_string(index=False, float_format=lambda v: f'{v:.4g}')}\n```\n\n"
            f"Por ano (custo 1 pb, pontos-base por noite):\n\n```\n{by_year.to_string()}\n```\n")
    (c3.DOCS / "c12-report.md").write_text(text, encoding="utf-8")
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"date": date.today().isoformat(), "event": "c12_confirmation", "code_commit": commit,
                                 "preregistration_sha256": prereg, "confirmed": confirmed, "base": base,
                                 "stress": stress, "by_symbol": by_symbol,
                                 "positive_years": int((by_year > 0).sum())}, sort_keys=True) + "\n")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["confirm"]:
        sys.stdout.write("usage: python -m scripts.research.fx_c12_overnight confirm\n")
        raise SystemExit(2)
    raise SystemExit(confirm())

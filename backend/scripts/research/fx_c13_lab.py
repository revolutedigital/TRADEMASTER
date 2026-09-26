"""Round C13: the shared, one-time run of both fronts' candidates on the untouched 2003-2011 data, with Holm over all
of them. See docs/forex/c13-preregistration.md.

    python -m scripts.research.fx_c13_banca discover       (front A)
    python -m scripts.research.fx_c13_carry discover       (front C)
    python -m scripts.research.fx_c13_lab holdout [--workers N]

With no candidate from either front, nothing runs and the untouched data stays untouched.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_c13_banca as banca
from scripts.research import fx_c13_carry as carry

REPORT = c3.DOCS / "c13-holdout.md"
SIGNIFICANCE, MIN_TRADES, MIN_MARKET_SHARE, MIN_POSITIVE_YEARS = 0.05, 300, 0.55, 5
# the code the discoveries ran: it may not change before the holdout (runners and checks added later may)
FROZEN = ("scripts/research/fx_c13_banca.py", "scripts/research/fx_c13_carry.py", "scripts/research/fx_c10_core.py",
          "scripts/research/fx_c10_families", "scripts/research/fx_c10_lab.py", "scripts/research/fx_c3_lab.py",
          "scripts/research/fx_c6_lab.py", "scripts/research/fx_fast_stats.py", "app")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)  # noqa: S603, S607


def guards() -> tuple[str, str]:
    """Clean committed tree, the holdout manifest tracked, the holdout never run, both discoveries registered and
    the code unchanged since each of them. Returns (commit, pre-registration sha256)."""
    commit, prereg = banca._registered()
    if _git("ls-files", "--error-unmatch", str(c10.HOLDOUT_MANIFEST)).returncode:
        raise RuntimeError("the holdout manifest is not committed")
    if banca._events("c13_holdout"):
        raise RuntimeError("the untouched period is used once: the C13 holdout already ran")
    for kind in ("c13_a_discovery", "c13_c_discovery"):
        events = banca._events(kind)
        if not events:
            raise RuntimeError(f"{kind} has not run")
        if _git("diff", "--quiet", events[-1]["code_commit"], "HEAD", "--", *FROZEN).returncode:
            raise RuntimeError(f"the code changed after {kind}: the holdout must run the discovered code")
    return commit, prereg


def daily_t(series: pd.Series) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    t = values.mean() / (values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 and values.std() > 0 else np.nan
    return float(t), float(norm.sf(t)) if np.isfinite(t) else 1.0


def holdout(workers: int) -> int:
    commit, prereg = guards()
    carried_c = list(banca._events("c13_c_discovery")[-1]["candidates"])
    carry_a = banca.passed()
    if carry_a:  # amendment 2: front A only goes on if the pipeline found no edge on data without one
        placebo = banca._events("c13_a_placebo")
        if not placebo:
            raise RuntimeError("run the front A placebo first (python -m scripts.research.fx_c13_placebo run)")
        carry_a = bool(placebo[-1]["clean"])
    if not carry_a and not carried_c:
        sys.stdout.write("no candidate from either front: the untouched 2003-2011 data stays untouched\n")
        return 0
    rows = []
    if carry_a:
        accepted, days = banca.holdout_accepted(workers)
        rows.append({"front": "A", "candidate": "banca de day trade"} | banca.score(accepted, days)
                    | banca.money_path(accepted))
    if carried_c:
        returns = carry.holdout_returns(carried_c)
        for name in carried_c:
            base, stress = returns[("base", name)], returns[("stress", name)]
            t, p = daily_t(base)
            years = base.groupby(base.index.year).sum()
            rows.append({"front": "C", "candidate": name, "days": len(base), "mean_net": float(base.mean()),
                         "mean_stress": float(stress.mean()), "t": t, "p_one_sided": p,
                         "positive_years": int((years > 0).sum()), "years": len(years),
                         "annual_return": float(base.mean() * carry.days_per_year(base.index))})
    table = pd.DataFrame(rows)
    table["p_holm"] = c3.holm(table["p_one_sided"].fillna(1.0).to_numpy())
    common = (table["p_holm"] <= SIGNIFICANCE) & (table["mean_net"] > 0) & (table["mean_stress"] > 0)
    front_a = (table["front"] == "A") & (table.get("trades", 0) >= MIN_TRADES) \
        & (table.get("market_share", 0.0) >= MIN_MARKET_SHARE)
    front_c = (table["front"] == "C") & (table.get("positive_years", 0) >= MIN_POSITIVE_YEARS)
    table["approved"] = common & (front_a | front_c)
    approved = list(table.loc[table["approved"], "candidate"])
    text = (f"# C13: o período intocado (2003-01 a 2011-12), uma rodada para as duas frentes\n\n"
            f"Commit do código: {commit}. **Aprovadas: {len(approved)} de {len(table)}**"
            + (f": {', '.join(approved)}" if approved else "") + f". Holm sobre as {len(table)} levadas.\n\n"
            f"```\n{c10._fmt(table)}\n```\n")
    REPORT.write_text(text, encoding="utf-8")
    banca._append({"date": date.today().isoformat(), "event": "c13_holdout", "code_commit": commit,
                   "preregistration_sha256": prereg, "approved": approved,
                   "results": json.loads(table.to_json(orient="records"))})
    sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c13_lab")
    parser.add_argument("command", choices=["holdout"])
    parser.add_argument("--workers", type=int, default=9)
    return holdout(parser.parse_args(argv).workers)


if __name__ == "__main__":
    raise SystemExit(main())

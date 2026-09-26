"""Round C13, front A, amendment 2: the leakage check before the untouched period. The whole banca pipeline (candidate
trades, features, recent results, walk-forward with the chosen model) runs again on the discovery minutes with the
sign of every minute's move flipped at random. Such data have no edge by construction: every stopping rule on a
martingale has zero expected gross result, so the accepted trades must not be significantly positive net of cost.
If they are, something in the pipeline sees the future, and front A does not go to the untouched period.

    python -m scripts.research.fx_c13_placebo run [--workers N]

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_c13_banca as banca

SEED = 20261026
MAX_T = 2.0  # the placebo's accepted trades may not reach t >= 2 with a positive mean
REPORT = c3.DOCS / "c13-a-placebo.md"


def flipped_rows(job: tuple[str, int]) -> pd.DataFrame:
    symbol, seed = job
    m1 = c3.load_included(symbol, pd.read_csv(c3.MANIFEST))
    coins = np.random.default_rng([seed, c10.SYMBOLS.index(symbol)]).integers(0, 2, len(m1)) * 2 - 1
    return banca.trade_rows(symbol, c3.sign_flip(m1, coins.astype(np.int8)))


def run(workers: int) -> int:
    commit, prereg = banca._registered()
    if banca._events("c13_a_placebo"):
        raise RuntimeError("the placebo already ran: its result is in the registry")
    event = banca._events("c13_a_discovery")[-1]
    chosen = event["chosen"]
    depth, weight, threshold = int(chosen["max_depth"]), int(chosen["min_child_weight"]), float(chosen["threshold"])
    manifest = pd.read_csv(c3.MANIFEST)
    included = set(manifest.loc[manifest["symbol_included"], "symbol"])
    jobs = [(s, SEED) for s in c10.SYMBOLS if s in included]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frame = banca.add_recent(pd.concat([f for f in pool.map(flipped_rows, jobs) if len(f)], ignore_index=True))
    predictions = banca.walk_forward(frame, grid=((depth, weight),))
    decided = frame.iloc[predictions["row"].to_numpy()]
    predicted = predictions[banca.config_name(depth, weight)].to_numpy()
    days = decided["fx_day"].nunique()
    rows = [{"threshold": tau} | banca.score(decided[predicted > tau], days) for tau in (0.0, threshold)]
    rows.append({"threshold": "todas"} | banca.score(decided, days))
    table = pd.DataFrame(rows)
    at_chosen = table.iloc[1]
    clean = bool(not (at_chosen["mean_net"] > 0 and at_chosen["t"] >= MAX_T))
    text = (f"# C13, frente A: placebo antes do período intocado (emenda 2)\n\nCommit do código: {commit}. Os minutos da "
            f"descoberta com o sinal de cada movimento trocado ao acaso (semente {SEED}), e a mesma banca "
            f"(profundidade {depth}, peso {weight}, τ = {threshold:g}) com o mesmo walk-forward. Sem vantagem possível, a "
            f"média aceita não pode ser positiva com t ≥ {MAX_T:g}. **Limpo: {'sim' if clean else 'NÃO'}.**\n\n"
            f"```\n{c10._fmt(table)}\n```\n")
    REPORT.write_text(text, encoding="utf-8")
    banca._append({"date": date.today().isoformat(), "event": "c13_a_placebo", "code_commit": commit,
                   "preregistration_sha256": prereg, "clean": clean, "results": table.to_dict(orient="records")})
    sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c13_placebo")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--workers", type=int, default=8)
    return run(parser.parse_args(argv).workers)


if __name__ == "__main__":
    raise SystemExit(main())

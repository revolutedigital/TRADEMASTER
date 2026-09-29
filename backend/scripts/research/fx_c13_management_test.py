"""C13-A, a quick side test Igor asked for (exploratory, not pre-registered): tighter stops (1, 0.5, 0.25 ATR) and a
tighter target (0.5R instead of 3R), against the frozen stop 1 ATR / target 3R, on the exact same candidates — how
much each shape would have won and lost. This is a one-shot read of the trailing year, not the choose-on-2016-2022,
read-on-2023-2026 discipline the frozen shape itself was held to — a good result here is not a green light to switch,
only something worth taking to that honest split before it goes live.

    python -m scripts.research.fx_c13_management_test
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_c13_banca as banca
from scripts.research.fx_c10_families import FAMILIES

REPORT = c3.DOCS / "c13-a-management-test.md"
CONFIG = banca.config_name(3, 2000)
THRESHOLDS = (0.02, 0.05)  # the threshold live now, and the frozen discovery's own
# (stop ATR, first target R, fraction banked there, runner target R, trailing R, max minutes — core.py's columns)
GRID = np.array([
    (1.0, 0.0, 0.0, 3.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.5, 0.0, 0.0),
    (0.5, 0.0, 0.0, 3.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
    (0.25, 0.0, 0.0, 3.0, 0.0, 0.0), (0.25, 0.0, 0.0, 0.5, 0.0, 0.0)])
NAMES = ("stop 1 ATR · alvo 3R (congelada)", "stop 1 ATR · alvo 0,5R",
        "stop 0,5 ATR · alvo 3R", "stop 0,5 ATR · alvo 0,5R (os dois apertados)",
        "stop 0,25 ATR · alvo 3R", "stop 0,25 ATR · alvo 0,5R (os dois bem apertados)")
LAST_DAY = banca.SAMPLE_END  # the discovery data ends 2026-09-01; Igor asked for the trailing 12 months of it
WINDOW_DAYS = 365


def candidates() -> pd.DataFrame:
    predictions = pd.read_parquet(banca.PREDICTIONS, columns=["row", "year", CONFIG])
    frame = pd.read_parquet(banca.DATASET, columns=["symbol", "set_id", "entry_time", "fx_day"])
    table = frame.iloc[predictions["row"].to_numpy()].reset_index(drop=True)
    table = table.assign(pred=predictions[CONFIG].to_numpy(), year=predictions["year"].to_numpy())
    table = table[(table["pred"] > min(THRESHOLDS)) & (table["entry_time"] >= LAST_DAY - WINDOW_DAYS * 86400)]
    table = table.reset_index(drop=True)
    return table.assign(cid=np.arange(len(table)))


def market_worker(job: tuple[str, pd.DataFrame]) -> np.ndarray:
    """Every candidate of one market, re-managed under both grid rows, at both thresholds: (threshold, grid, cid,
    net R, net R at double cost)."""
    symbol, wanted = job
    m1 = c3.load_included(symbol, pd.read_csv(c3.MANIFEST))
    day_end = core.day_end_minutes(m1)
    cost = c3.cost_of(symbol)
    lookup = {(int(s), float(t)): (int(c), float(p)) for s, t, c, p in
              wanted[["set_id", "entry_time", "cid", "pred"]].itertuples(index=False)}
    sets = set(wanted["set_id"].astype(int))
    contexts: dict[str, core.Context] = {}
    rows = []
    for set_id, (name, params) in enumerate(c10.family_sets()):
        if set_id not in sets:
            continue
        timeframe = params["timeframe"]
        if timeframe not in contexts:
            contexts[timeframe] = core.build_context(symbol, m1, timeframe)
        side, stop_ref = FAMILIES[name].signals(contexts[timeframe], params)
        entries, sides, stops, atrs = core.entry_arrays(contexts[timeframe], side, stop_ref, len(m1))
        found = [lookup.get((set_id, float(t))) for t in m1[entries, 0]]
        keep = np.array([f is not None for f in found])
        if not keep.any():
            continue
        entries, sides, stops, atrs = entries[keep], sides[keep], stops[keep], atrs[keep]
        cids = np.array([f[0] for f in found if f is not None])
        preds = np.array([f[1] for f in found if f is not None])
        cid_at = dict(zip(entries.tolist(), cids.tolist()))
        for tau_i, tau in enumerate(THRESHOLDS):
            chosen = preds > tau
            if not chosen.any():
                continue
            for g in range(len(GRID)):
                trades = core.manage(m1, day_end, entries[chosen], sides[chosen], stops[chosen], atrs[chosen], cost,
                                     GRID, g)
                if len(trades):
                    rows.append(np.column_stack([np.full(len(trades), tau_i), np.full(len(trades), g),
                                                 [cid_at[int(e)] for e in trades[:, 1]], trades[:, 3], trades[:, 4]]))
    return np.concatenate(rows) if rows else np.empty((0, 5))


def run(workers: int = 8) -> pd.DataFrame:
    table = candidates()
    jobs = [(symbol, group) for symbol, group in table.groupby("symbol")]
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(market_worker, jobs), 1):
            parts.append(result)
            sys.stdout.write(f"\r{i}/{len(jobs)} mercados")
            sys.stdout.flush()
    sys.stdout.write("\n")
    trades = pd.DataFrame(np.concatenate(parts), columns=["tau_i", "g", "cid", "net", "stress"])
    trades = trades.astype({"tau_i": np.int8, "g": np.int16, "cid": np.int64})
    return table.merge(trades, on="cid")


def score(rows: pd.DataFrame, days: int) -> dict:
    base = banca.score(rows, days)
    wins, losses = rows["net"] > 0, rows["net"] <= 0
    return {**base, "media_ganho_R": float(rows.loc[wins, "net"].mean()) if wins.any() else np.nan,
            "media_perda_R": float(rows.loc[losses, "net"].mean()) if losses.any() else np.nan,
            "total_R": float(rows["net"].sum())}


def main() -> int:
    merged = run()
    days = merged["fx_day"].nunique()
    rows = []
    for tau_i, tau in enumerate(THRESHOLDS):
        for g, name in enumerate(NAMES):
            subset = merged[(merged["tau_i"] == tau_i) & (merged["g"] == g)]
            rows.append({"limiar": tau, "gestao": name} | score(subset, days))
    table = pd.DataFrame(rows)
    cols = ["limiar", "gestao", "trades", "win_rate", "media_ganho_R", "media_perda_R", "mean_net", "mean_stress",
           "t", "total_R", "trades_per_day"]
    text = (
        "# C13-A: stops e alvo mais apertados contra a gestão congelada (teste do Igor, exploratório)\n\n"
        f"Último ano de dado ({pd.Timestamp(LAST_DAY - WINDOW_DAYS * 86400, unit='s', tz='UTC').date()} a "
        f"{pd.Timestamp(LAST_DAY, unit='s', tz='UTC').date()}) — leitura única, não é a disciplina de escolher em "
        "2016-2022 e ler em 2023-2026 que a gestão congelada passou; um número bom aqui não é aval pra trocar, é "
        "candidata pra passar por esse teste honesto antes de ir pro robô. As mesmas candidatas do modelo, geridas "
        "por 6 regras: a congelada (stop 1 ATR, alvo 3R) e 5 combinações de stop mais apertado (1 / 0,5 / 0,25 ATR) "
        "com o alvo de sempre ou o alvo mais perto (0,5R). `media_ganho_R`/`media_perda_R`: a média de R só das "
        "operações que ganharam / só das que perderam. Nos dois limiares em uso: 0,05 (a descoberta) e 0,02 (o que "
        "está rodando ao vivo).\n\n```\n"
        + table[cols].to_string(index=False, float_format=lambda v: f"{v:,.4g}") + "\n```\n")
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

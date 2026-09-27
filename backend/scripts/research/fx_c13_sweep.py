"""C13 front A, exploratory sweep — NOT a pre-registered test, nothing here approves anything. The candidates the
walk-forward already scored in 2016-2026 (each year decided by a model trained only on trades that exited before it)
are re-run under other thresholds, stops, targets and trailing rules, and turned into money from a US$ 1,000 account
with the demo broker's real minimum volumes (data/c13/specs.json, read from the account on 2026-09-26).

What it can say: how much each knob moves the result and the drawdown. What it cannot say: that the best row keeps
winning — the best of hundreds of rows on the data that scores them is the winner's curse. The honest check is at the
end of the report: choose on 2016-2022 only, then look at 2023-2026, which the choice never saw.

Deviation from the live runner, declared: a trade that a longer management keeps open blocks later candidates of its
own (market, set) only among the candidates of that threshold (live, every candidate the family raised blocks it).
The frozen management at the frozen threshold reproduces the discovery's trades exactly (checked at run time).

    python -m scripts.research.fx_c13_sweep [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from scripts.research import fx_banca_demo as old_banca
from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core
from scripts.research import fx_c10_lab as c10
from scripts.research import fx_c13_banca as banca
from scripts.research.fx_c10_families import FAMILIES

SPECS = banca.LAB / "specs.json"
TRADES = banca.LAB / "sweep_trades.parquet"
REPORT = c3.DOCS / "c13-a-sweep.md"
CONFIG = banca.config_name(3, 2000)
FROZEN_TAU = 0.05
THRESHOLDS = (0.02, 0.05, 0.08, 0.12, 0.16)
STOPS = (0.5, 1.0, 1.5, 2.0)
EXITS = (  # name, first target R, fraction banked there, runner target R, trailing R (after the first target)
    ("alvo 1R", 0.0, 0.0, 1.0, 0.0), ("alvo 2R", 0.0, 0.0, 2.0, 0.0), ("alvo 3R", 0.0, 0.0, 3.0, 0.0),
    ("alvo 5R", 0.0, 0.0, 5.0, 0.0),
    ("empate em 1R + trailing 0,25R", 1.0, 0.0, 0.0, 0.25), ("empate em 1R + trailing 0,5R", 1.0, 0.0, 0.0, 0.5),
    ("empate em 1R + trailing 1R", 1.0, 0.0, 0.0, 1.0),
    ("metade em 1R + trailing 0,5R", 1.0, 0.5, 0.0, 0.5), ("metade em 1R + trailing 1R", 1.0, 0.5, 0.0, 1.0),
    ("metade em 1R + resto em 3R", 1.0, 0.5, 3.0, 0.0),
)
GRID = np.array([(stop, tp1, fraction, runner, trail, 0.0) for stop in STOPS
                 for _, tp1, fraction, runner, trail in EXITS], dtype=float)
NAMES = [f"stop {stop:g} ATR · {name}" for stop in STOPS for name, *_ in EXITS]
FROZEN_G = NAMES.index("stop 1 ATR · alvo 3R")
SUSPECT_R = 15.0  # |R| beyond this in a same-day trade is a bad tick in the data, not a trade: counted and dropped
RISKS = (0.001, 0.0025, 0.005, 0.01, 0.02)
POLICIES = {"pula": False, "mínimo": True}  # when the minimum volume risks more than the budget: skip, or take it
HEATS = (0.02, 0.05, 0.10)  # the most stop money open at once, as a share of the balance
START_USD = 1000.0
CANDIDATES_FILE = banca.LAB / "sweep_candidates.parquet"
PERIODS = {"2016-2026": (2016, 2026), "2023-2026": (2023, 2026), "2016-2022": (2016, 2022)}
LAST_DAY = pd.Timestamp("2026-09-01", tz="UTC").timestamp()
DD_CAPS = (0.2, 0.35, 0.5)
USD_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
HKD_PER_USD = 7.8  # the peg


# --- the candidates, re-managed ---------------------------------------------------------------------------------------


def candidates() -> pd.DataFrame:
    predictions = pd.read_parquet(banca.PREDICTIONS, columns=["row", "year", CONFIG])
    frame = pd.read_parquet(banca.DATASET, columns=["symbol", "set_id", "entry_time", "fx_day", "net"])
    table = frame.iloc[predictions["row"].to_numpy()].reset_index(drop=True)
    table = table.assign(pred=predictions[CONFIG].to_numpy(), year=predictions["year"].to_numpy())
    table = table[table["pred"] > min(THRESHOLDS)].reset_index(drop=True)
    return table.assign(cid=np.arange(len(table)))


def market_worker(job: tuple[str, pd.DataFrame]) -> dict:
    """Every candidate of one market re-run under every (threshold, management); its daily ATR and entry price."""
    symbol, wanted = job
    m1 = c3.load_included(symbol, pd.read_csv(c3.MANIFEST))
    day_end = core.day_end_minutes(m1)
    cost = c3.cost_of(symbol)
    lookup = {(int(s), float(t)): (int(c), float(p)) for s, t, c, p in
              wanted[["set_id", "entry_time", "cid", "pred"]].itertuples(index=False)}
    sets = set(wanted["set_id"].astype(int))
    contexts: dict[str, core.Context] = {}
    atr_of, price_of, rows = {}, {}, []
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
        atr_of.update(zip(cids.tolist(), atrs.tolist()))
        price_of.update(zip(cids.tolist(), m1[entries, 1].tolist()))
        cid_at = dict(zip(entries.tolist(), cids.tolist()))
        for tau_i, tau in enumerate(THRESHOLDS):
            chosen = preds > tau
            if not chosen.any():
                continue
            for g in (range(len(GRID)) if tau >= FROZEN_TAU else (FROZEN_G,)):
                trades = core.manage(m1, day_end, entries[chosen], sides[chosen], stops[chosen], atrs[chosen], cost,
                                     GRID, g)
                if len(trades):
                    rows.append(np.column_stack([np.full(len(trades), tau_i), np.full(len(trades), g),
                                                 [cid_at[int(e)] for e in trades[:, 1]], m1[trades[:, 2].astype(np.int64), 0],
                                                 trades[:, 3], trades[:, 4]]))
    return {"atr": atr_of, "price": price_of,
            "trades": np.concatenate(rows) if rows else np.empty((0, 6))}


def rerun(table: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(symbol, group) for symbol, group in table.groupby("symbol")]
    atr, price, parts = {}, {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(market_worker, jobs):
            atr |= result["atr"]
            price |= result["price"]
            parts.append(result["trades"])
            sys.stdout.write(f"re-managed {len(atr):,} candidates\n")
            sys.stdout.flush()
    trades = pd.DataFrame(np.concatenate(parts), columns=["tau_i", "g", "cid", "exit_time", "net", "stress"])
    trades = trades.astype({"tau_i": np.int8, "g": np.int16, "cid": np.int64})
    table = table.assign(atr=table["cid"].map(atr), price=table["cid"].map(price))
    return table, trades


def check_reproduction(table: pd.DataFrame, trades: pd.DataFrame) -> None:
    """The frozen management at the frozen threshold must give back the discovery's own trades, R for R."""
    frozen = trades[(trades["tau_i"] == THRESHOLDS.index(FROZEN_TAU)) & (trades["g"] == FROZEN_G)]
    expected = table[table["pred"] > FROZEN_TAU]
    merged = expected.merge(frozen, on="cid", how="left")
    if len(frozen) != len(expected) or merged["net_y"].isna().any() \
            or not np.allclose(merged["net_x"], merged["net_y"], atol=1e-9):
        raise RuntimeError(f"the re-run does not reproduce the discovery: {len(frozen)} vs {len(expected)} trades")


# --- money from US$ 1,000 ---------------------------------------------------------------------------------------------


def usd_per_quote(table: pd.DataFrame) -> np.ndarray:
    """US$ per unit of each candidate's quote currency, at the last daily close before its entry."""
    closes = {}
    for pair in USD_PAIRS:
        m1 = np.load(c3.ARRAYS / f"{pair}.npy", mmap_mode="r")
        days = pd.Series(m1[:, 4], index=m1[:, 0] // 86400).groupby(level=0).last()
        closes[pair] = days
    day = table["entry_time"].to_numpy() // 86400 - 1
    rate = {"USD": np.ones(len(table)), "HKD": np.full(len(table), 1 / HKD_PER_USD)}
    for pair in USD_PAIRS:
        series = closes[pair]
        values = series.to_numpy()[np.clip(np.searchsorted(series.index.to_numpy(), day, side="right") - 1, 0, None)]
        rate[pair[:3] if pair.endswith("USD") else pair[3:]] = values if pair.endswith("USD") else 1 / values
    quote = table["symbol"].map(lambda s: old_banca.QUOTE.get(old_banca.FUSION.get(s, s), s[3:]))
    return np.array([rate[q][i] for i, q in enumerate(quote)])


@njit(cache=True)
def simulate(entry_t, exit_t, net, per_unit, min_vol, step, risk, force_min, heat, compound, start):
    """Each position gets the largest multiple of the step whose stop is worth at most `risk` of the base (the
    realized balance at entry if `compound`, else the starting balance for good); when even the minimum is worth
    more, it is skipped, or taken at the minimum if `force_min`. A position is also skipped when the stops already
    open plus its own would pass `heat` of the same base (the first version measured the cap on the balance while
    sizing on the start, which let an account that had grown hold ever more trades: its yearly profits were inflated). Results are realized at exit. Entries must come sorted by time.
    Returns: final balance; worst drawdown of the realized balance; worst drawdown had every open stop been hit at
    once; trades taken; skipped for the minimum volume; skipped for the heat cap; the most positions open at once;
    whether it went broke."""
    n = entry_t.shape[0]
    order = np.argsort(exit_t)
    pnl = np.zeros(n)
    at_risk = np.zeros(n)
    taken = np.zeros(n, dtype=np.bool_)
    balance, peak, worst, worst_open = start, start, 0.0, 0.0
    open_risk, open_count, max_count = 0.0, 0, 0
    n_taken, n_small, n_heat, j, broke = 0, 0, 0, 0, False
    for i in range(n + 1):
        now = entry_t[i] if i < n else np.inf
        while j < n and exit_t[order[j]] < now:
            k = order[j]
            if taken[k]:
                balance += pnl[k]
                open_risk -= at_risk[k]
                open_count -= 1
                peak = max(peak, balance)
                worst = max(worst, (peak - balance) / peak)
            j += 1
        if i == n:
            break
        if balance <= 0:
            broke = True
            break
        base = balance if compound else start
        volume = np.floor(risk * base / per_unit[i] / step[i] + 1e-9) * step[i]
        if volume < min_vol[i] - 1e-12:
            if not force_min:
                n_small += 1
                continue
            volume = min_vol[i]
        if open_risk + volume * per_unit[i] > heat * base:
            n_heat += 1
            continue
        taken[i] = True
        n_taken += 1
        at_risk[i] = volume * per_unit[i]
        pnl[i] = net[i] * at_risk[i]
        open_risk += at_risk[i]
        open_count += 1
        max_count = max(max_count, open_count)
        worst_open = max(worst_open, (peak - (balance - open_risk)) / peak)
    return balance, worst, worst_open, n_taken, n_small, n_heat, max_count, broke or balance <= 0


def period_bounds(name: str) -> tuple[float, float, float]:
    first, last = PERIODS[name]
    start = pd.Timestamp(f"{first}-01-01", tz="UTC").timestamp()
    end = min(pd.Timestamp(f"{last + 1}-01-01", tz="UTC").timestamp(), LAST_DAY)
    return start, end, (end - start) / (365.25 * 86400)


def r_stats(rows: pd.DataFrame, days: int) -> dict:
    score = banca.score(rows, days)
    path = rows.sort_values("exit_time")["net"].cumsum().to_numpy()
    drawdown = float((np.maximum.accumulate(np.maximum(path, 0)) - path).max()) if len(path) else 0.0
    return {"trades": score["trades"], "por_dia": score["trades_per_day"], "media_R": score["mean_net"],
            "media_R_estresse": score["mean_stress"], "acerto": score["win_rate"], "t": score["t"],
            "total_R": float(rows["net"].sum()), "pior_queda_R": drawdown}


def sweep(table: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = json.loads(SPECS.read_text())
    table = table.assign(usd=usd_per_quote(table), min_vol=table["symbol"].map(lambda s: specs[s]["minVolume"]),
                         step=table["symbol"].map(lambda s: specs[s]["volumeStep"]))
    rows = trades.merge(table[["cid", "symbol", "entry_time", "fx_day", "atr", "usd", "min_vol", "step"]], on="cid")
    days_in = {}
    for period in PERIODS:
        start, end, _ = period_bounds(period)
        days_in[period] = int(table.loc[(table["entry_time"] >= start) & (table["entry_time"] < end), "fx_day"].nunique())
    stats, money = [], []
    for (tau_i, g), group in rows.groupby(["tau_i", "g"]):
        suspect = group["net"].abs() > SUSPECT_R
        group = group[~suspect].sort_values("entry_time")
        per_unit = (GRID[g, core.STOP_ATR] * group["atr"] * group["usd"]).to_numpy()
        for period in PERIODS:
            start, end, years = period_bounds(period)
            inside = ((group["entry_time"] >= start) & (group["entry_time"] < end)).to_numpy()
            if not inside.any():
                continue
            part = group[inside]
            base = {"limiar": THRESHOLDS[tau_i], "gestao": NAMES[g], "periodo": period}
            stats.append(base | r_stats(part, days_in[period]) | {"descartadas_dado_ruim": int(suspect.sum())})
            times = [part[c].to_numpy(dtype=np.float64) for c in ("entry_time", "exit_time")]
            results = {c: part[c].to_numpy(dtype=np.float64) for c in ("net", "stress")}
            costs = [per_unit[inside], part["min_vol"].to_numpy(dtype=np.float64),
                     part["step"].to_numpy(dtype=np.float64)]
            for risk in RISKS:
                for policy, force in POLICIES.items():
                    for heat in HEATS:
                        for compound in (False, True):
                            settings = (risk, force, heat, compound, START_USD)
                            final, worst, worst_open, taken, small, capped, most, broke = simulate(
                                *times, results["net"], *costs, *settings)
                            stress = simulate(*times, results["stress"], *costs, *settings)[0]
                            money.append(base | {
                                "risco": risk, "volume_minimo": policy, "teto_aberto": heat, "compostos": compound,
                                "final_usd": final, "final_estresse_usd": stress,
                                "lucro_ano_usd": (final - START_USD) / years,
                                "lucro_ano_estresse_usd": (stress - START_USD) / years,
                                "pior_queda": worst, "pior_queda_todos_stops": worst_open, "operacoes": taken,
                                "puladas_volume": small, "puladas_teto": capped, "posicoes_max": most,
                                "quebrou": broke})
    return pd.DataFrame(stats), pd.DataFrame(money)


# --- the honest check and the report ------------------------------------------------------------------------------------


KEYS = ["limiar", "gestao", "risco", "volume_minimo", "teto_aberto"]


def out_of_sample(money: pd.DataFrame) -> pd.DataFrame:
    """For each cap on the worst drawdown (every open stop hit at once): the setting that made the most money in
    2016-2022 with double cost and no compounding, without going broke; then what that very setting did in 2023-2026,
    which the choice never saw. The frozen live setting is the last row, for reference."""
    flat = money[~money["compostos"]]
    early = flat[(flat["periodo"] == "2016-2022") & ~flat["quebrou"]]
    late = flat[flat["periodo"] == "2023-2026"].set_index(KEYS)
    picks = []
    for cap in DD_CAPS:
        fit = early[early["pior_queda_todos_stops"] <= cap]
        if len(fit):
            picks.append((f"queda ≤ {cap:.0%}", fit.loc[fit["final_estresse_usd"].idxmax()]))
    frozen = early[(early["limiar"] == FROZEN_TAU) & (early["gestao"] == NAMES[FROZEN_G]) & (early["risco"] == 0.005)
                   & (early["volume_minimo"] == "pula") & (early["teto_aberto"] == 0.05)]
    if len(frozen):
        picks.append(("a congelada (0,5%, teto 5%)", frozen.iloc[0]))
    rows = []
    for label, best in picks:
        key = tuple(best[k] for k in KEYS)
        after = late.loc[key] if key in late.index else None
        rows.append({"escolha": label, **{k: best[k] for k in KEYS},
                     "16-22_lucro_ano": best["lucro_ano_usd"], "16-22_estresse_ano": best["lucro_ano_estresse_usd"],
                     "16-22_queda": best["pior_queda_todos_stops"],
                     "23-26_lucro_ano": np.nan if after is None else after["lucro_ano_usd"],
                     "23-26_estresse_ano": np.nan if after is None else after["lucro_ano_estresse_usd"],
                     "23-26_queda": np.nan if after is None else after["pior_queda_todos_stops"],
                     "23-26_quebrou": None if after is None else bool(after["quebrou"])})
    return pd.DataFrame(rows)


def leaderboard(money: pd.DataFrame, cap: float = 0.25, size: int = 12) -> pd.DataFrame:
    """The `size` settings that made the most in 2016-2022 with double cost, no compounding and at most `cap` of
    worst drawdown, each beside its own 2023-2026: whether the good ones stay good, not just the single best."""
    flat = money[~money["compostos"]]
    early = flat[(flat["periodo"] == "2016-2022") & ~flat["quebrou"] & (flat["pior_queda_todos_stops"] <= cap)]
    late = flat[flat["periodo"] == "2023-2026"].set_index(KEYS)
    rows = []
    for _, best in early.sort_values("lucro_ano_estresse_usd", ascending=False).head(size).iterrows():
        after = late.loc[tuple(best[k] for k in KEYS)]
        rows.append({**{k: best[k] for k in KEYS}, "16-22_lucro_ano": best["lucro_ano_usd"],
                     "16-22_estresse_ano": best["lucro_ano_estresse_usd"], "16-22_queda": best["pior_queda_todos_stops"],
                     "23-26_lucro_ano": after["lucro_ano_usd"], "23-26_estresse_ano": after["lucro_ano_estresse_usd"],
                     "23-26_queda": after["pior_queda_todos_stops"]})
    return pd.DataFrame(rows)


def _table(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string(index=False, float_format=lambda v: f"{v:,.4g}") + "\n```\n"


def write_report(stats: pd.DataFrame, money: pd.DataFrame, oos: pd.DataFrame) -> str:
    frozen_name = NAMES[FROZEN_G]
    by_tau = stats[stats["gestao"] == frozen_name].sort_values(["periodo", "limiar"])
    at_frozen_tau = stats[(stats["limiar"] == FROZEN_TAU) & (stats["periodo"] != "2016-2022")] \
        .sort_values(["periodo", "media_R"], ascending=[True, False])
    flat = money[~money["compostos"] & (money["periodo"] != "2016-2022")]
    columns = ["periodo", "limiar", "gestao", "risco", "volume_minimo", "teto_aberto", "lucro_ano_usd",
               "lucro_ano_estresse_usd", "pior_queda", "pior_queda_todos_stops", "operacoes", "puladas_volume",
               "puladas_teto", "posicoes_max", "quebrou"]
    frozen_money = flat[(flat["gestao"] == frozen_name) & (flat["teto_aberto"] == 0.05)] \
        .sort_values(["periodo", "limiar", "volume_minimo", "risco"])
    heat_effect = flat[(flat["gestao"] == frozen_name) & (flat["limiar"] == FROZEN_TAU)] \
        .sort_values(["periodo", "volume_minimo", "risco", "teto_aberto"])
    text = (
        "# C13, frente A: varredura exploratória de limiar, gestão e risco (não é teste pré-registrado)\n\n"
        "**Corrigido em 2026-09-27.** A primeira versão dimensionava cada operação sobre os US$ 1.000 iniciais mas "
        "media o teto de stops abertos sobre o saldo, que crescia: com a conta maior cabiam cada vez mais operações, e o "
        "lucro por ano \"sem juros compostos\" saiu muito inflado. Agora o teto usa a mesma base do tamanho.\n\n"
        f"As candidatas que o walk-forward já decidiu em 2016-2026 (cada ano por um modelo que só viu operações "
        f"encerradas antes dele), refeitas com {len(THRESHOLDS)} limiares, {len(GRID)} gestões e {len(RISKS)} riscos, "
        f"em dinheiro a partir de US$ {START_USD:,.0f} com o volume mínimo real de cada mercado na demo. A gestão "
        f"congelada no limiar congelado reproduz a descoberta operação por operação (checado na execução). "
        f"Operações com |R| > {SUSPECT_R:g} num mesmo dia são vela ruim no dado e ficam de fora (contadas).\n\n"
        "**Isto não aprova nada.** A melhor linha entre centenas, medida no mesmo dado que a escolhe, é otimista por "
        "construção. O teste honesto é a última tabela: escolhida só com 2016-2022, olhada em 2023-2026.\n\n"
        f"## 1. O limiar do modelo, com a gestão congelada ({frozen_name})\n\n{_table(by_tau)}\n"
        f"## 2. As {len(GRID)} gestões no limiar congelado ({FROZEN_TAU:g})\n\n{_table(at_frozen_tau)}\n"
        f"## 3. Dinheiro a partir de US$ {START_USD:,.0f} (sem juros compostos), gestão congelada, teto aberto 5%\n\n"
        "Sem um teto para a soma dos stops abertos, a banca quebra US$ 1.000: em dias de rajada o modelo aceita "
        "dezenas de entradas no mesmo movimento, e com o volume mínimo isso chegou a 5-90 vezes o saldo em risco ao "
        "mesmo tempo (primeira rodada desta varredura). Daqui em diante toda simulação tem teto. `risco`: o stop de "
        "cada operação como fração do saldo inicial. `volume_minimo`: quando o menor volume da corretora arrisca mais "
        "que isso, `pula` a operação ou entra com o `mínimo`. `teto_aberto`: a soma máxima de stops abertos. "
        "`pior_queda_todos_stops`: a pior queda se todo stop aberto batesse junto (a pior queda possível, não a "
        "vista). Lucro por ano em US$ sobre os US$ 1.000.\n\n"
        f"{_table(frozen_money[columns])}\n"
        f"### O teto aberto, no limiar {FROZEN_TAU:g}\n\n{_table(heat_effect[columns])}\n"
        "## 4. O teste honesto: escolhida em 2016-2022, olhada em 2023-2026\n\n"
        "Para cada teto de queda (todos os stops abertos batendo juntos), a combinação (limiar, gestão, risco, volume "
        "mínimo, teto aberto) que mais rendeu em 2016-2022 com custo dobrado e sem juros compostos, sem quebrar; ao "
        "lado, o que essa mesma combinação fez em 2023-2026, período que a escolha não viu.\n\n"
        f"{_table(oos)}\n"
        "### As 12 melhores de 2016-2022 (queda ≤ 25%), cada uma em 2023-2026\n\n"
        f"{_table(leaderboard(money))}\n")
    REPORT.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c13_sweep")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--reuse", action="store_true", help="skip the re-management, reuse its saved trades")
    args = parser.parse_args(argv)
    if args.reuse and TRADES.exists() and CANDIDATES_FILE.exists():
        table, trades = pd.read_parquet(CANDIDATES_FILE), pd.read_parquet(TRADES)
    else:
        table = candidates()
        sys.stdout.write(f"{len(table):,} candidates above {min(THRESHOLDS):g}\n")
        table, trades = rerun(table, args.workers)
        check_reproduction(table, trades)
        trades.to_parquet(TRADES)
        table.to_parquet(CANDIDATES_FILE)
    stats, money = sweep(table, trades)
    stats.to_parquet(banca.LAB / "sweep_stats.parquet")
    money.to_parquet(banca.LAB / "sweep_money.parquet")
    sys.stdout.write(write_report(stats, money, out_of_sample(money)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

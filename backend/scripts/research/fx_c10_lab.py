"""Round C10: the day-trade search. Every family and parameter set of fx_c10_families, crossed with the 144-row
management grid of fx_c10_core, in the 37 banca markets; discovery on 2012-2026 with the max-t bootstrap over every
configuration, then one run on untouched 2003-2011 data. See docs/forex/c10-preregistration.md.

    python -m scripts.research.fx_c10_lab discover [--workers N]
    python -m scripts.research.fx_c10_lab prepare-holdout
    python -m scripts.research.fx_c10_lab holdout [--workers N]

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
from scripts.research import fx_c9_lab as c9
from scripts.research import fx_c10_core as core
from scripts.research import fx_fast_stats as stats
from scripts.research.fx_c10_families import FAMILIES

PREREGISTRATION = c3.DOCS / "c10-preregistration.md"
REGISTRY = c3.DOCS / "c10-registry.jsonl"
HOLDOUT_MANIFEST = c3.DOCS / "c10-holdout-manifest.csv"
LAB = Path("data/c10")
HOLDOUT_RAW = Path("data/raw/histdata_m1_holdout")
HOLDOUT_ARRAYS = LAB / "holdout_m1"
DISCOVERY_TABLE = LAB / "discovery.parquet"
CANDIDATES_FILE = LAB / "candidates.json"
HOLDOUT_SAMPLE = ("2003-01", "2011-12")
SYMBOLS = c9.SYMBOLS

MIN_TRADES, MIN_MARKET_SHARE, MIN_MARKET_TRADES = 500, 0.55, 20
HOLDOUT_MIN_TRADES, CARRY, SIGNIFICANCE = 300, 10, 0.05
BOOTSTRAP_SEED, PLACEBO_SEED, PLACEBO_COPIES = 20261001, 20261002, 20
RISK_PER_TRADE = 0.005  # money translation of the report: 0.5% of equity at risk per trade (1 R)
# per (family set, grid row): trades, winners, sum R, sum R stress, sum of winning R, sum of losing R, sum %, minutes
STAT_COLUMNS = ("trades", "wins", "net", "stress", "win_sum", "loss_sum", "pct", "minutes")


def family_sets() -> list[tuple[str, dict]]:
    return [(name, params) for name, module in FAMILIES.items() for params in module.PARAMS]


def set_label(name: str, params: dict) -> str:
    rest = ", ".join(f"{k}={v}" for k, v in params.items() if k != "timeframe")
    return f"{name} {params['timeframe']}" + (f" ({rest})" if rest else "")


def fx_day_ids(m1: np.ndarray) -> np.ndarray:
    return core.minute_keys(m1[:, 0], 1440)


def run_trades(symbol: str, m1: np.ndarray, sets: list[tuple[str, dict]], grid: np.ndarray, rows=None):
    """Yield (set index, trades) for every family set; trades rows are core.manage's plus the grid row."""
    day_end = core.day_end_minutes(m1)
    contexts: dict[str, core.Context] = {}
    cost = c3.cost_of(symbol)
    for f, (name, params) in enumerate(sets):
        timeframe = params["timeframe"]
        if timeframe not in contexts:
            contexts[timeframe] = core.build_context(symbol, m1, timeframe)
        side, stop_ref = FAMILIES[name].signals(contexts[timeframe], params)
        entries, sides, stops, atrs = core.entry_arrays(contexts[timeframe], side, stop_ref, len(m1))
        structural = bool(np.isfinite(stops).any())
        wanted = range(len(grid)) if rows is None else rows[f]
        parts = [core.manage(m1, day_end, entries, sides, stops, atrs, cost, grid, g) for g in wanted
                 if grid[g, core.STOP_ATR] > 0 or structural]
        yield f, (np.concatenate(parts) if parts else np.empty((0, 6)))


def market_worker(job: tuple[str, str, int]) -> dict:
    """Discovery for one market: sparse daily sums per configuration and per-configuration totals."""
    symbol, manifest_path, origin = job
    m1 = c3.load_included(symbol, pd.read_csv(manifest_path))
    grid, _ = core.management_grid()
    sets = family_sets()
    n_rows = len(grid)
    days = fx_day_ids(m1) - origin
    totals = np.zeros((len(sets), n_rows, len(STAT_COLUMNS)))
    keys, sums, counts = [], [], []
    for f, trades in run_trades(symbol, m1, sets, grid):
        if not len(trades):
            continue
        g, entry, exit_, net, stress, pct = (trades[:, j] for j in range(6))
        g = g.astype(np.int64)
        minutes = (m1[exit_.astype(np.int64), 0] - m1[entry.astype(np.int64), 0]) / 60 + 1
        win = (net > 0).astype(float)
        for column, values in enumerate((np.ones_like(net), win, net, stress, win * net, (1 - win) * net, pct, minutes)):
            totals[f, :, column] = np.bincount(g, weights=values, minlength=n_rows)
        flat = (f * n_rows + g) * 100_000 + days[entry.astype(np.int64)]  # configuration and day in one key
        unique, inverse = np.unique(flat, return_inverse=True)
        keys.append(unique)
        sums.append(np.bincount(inverse, weights=net))
        counts.append(np.bincount(inverse))
    return {"symbol": symbol, "totals": totals,
            "keys": np.concatenate(keys) if keys else np.empty(0, dtype=np.int64),
            "sums": np.concatenate(sums) if sums else np.empty(0), "counts": np.concatenate(counts) if counts else np.empty(0)}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def _require_clean_and_registered() -> tuple[str, str]:
    if _git("status", "--porcelain", "--", "scripts", "app", str(PREREGISTRATION)):
        raise RuntimeError("commit the code and the pre-registration before running")
    return _git("rev-parse", "HEAD"), hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()


def _append(event: dict) -> None:
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=float) + "\n")


def _origin(manifest: pd.DataFrame) -> int:
    first = pd.Timestamp(f"{manifest['month'].min()}-01", tz="UTC").timestamp() - 86400
    return int(core.minute_keys(np.array([first]), 1440)[0])


def discover(workers: int) -> int:
    commit, prereg = _require_clean_and_registered()
    manifest = pd.read_csv(c3.MANIFEST)
    included = set(manifest.loc[manifest["symbol_included"], "symbol"])
    symbols = [s for s in SYMBOLS if s in included]
    origin = _origin(manifest)
    grid, names = core.management_grid()
    sets = family_sets()
    n_configs = len(sets) * len(grid)
    n_days = 10_000
    sums, counts = np.zeros((n_days, n_configs)), np.zeros((n_days, n_configs))
    totals = np.zeros((len(sets), len(grid), len(STAT_COLUMNS)))
    market_net = np.zeros((len(symbols), n_configs))
    market_trades = np.zeros((len(symbols), n_configs))
    jobs = [(s, str(c3.MANIFEST), origin) for s in symbols]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(market_worker, jobs):
            m = symbols.index(out["symbol"])
            config, day = out["keys"] // 100_000, out["keys"] % 100_000
            np.add.at(sums, (day, config), out["sums"])
            np.add.at(counts, (day, config), out["counts"])
            totals += out["totals"]
            market_net[m] = out["totals"][:, :, 2].reshape(-1)
            market_trades[m] = out["totals"][:, :, 0].reshape(-1)
            sys.stdout.write(f"{out['symbol']}: {int(out['totals'][:, :, 0].sum()):,} trades\n")
            sys.stdout.flush()
    flat_totals = totals.reshape(n_configs, -1)
    active = np.flatnonzero(flat_totals[:, 0] > 0)
    trading_days = np.flatnonzero(counts.sum(axis=1) > 0)
    table_sums, table_counts = sums[trading_days][:, active], counts[trading_days][:, active]
    mean, _, t = stats.cluster_t(table_sums, table_counts)
    _, p_maxt = stats.bootstrap_adjusted_p_values(table_sums, table_counts, seed=BOOTSTRAP_SEED)
    with np.errstate(divide="ignore", invalid="ignore"):
        market_mean = market_net[:, active] / market_trades[:, active]
    eligible = market_trades[:, active] >= MIN_MARKET_TRADES
    share = np.where(eligible, market_mean > 0, False).sum(axis=0) / np.maximum(eligible.sum(axis=0), 1)
    rows = []
    for j, k in enumerate(active):
        f, g = divmod(int(k), len(grid))
        n, wins, net, stress, win_sum, loss_sum, pct, minutes = flat_totals[k]
        rows.append({"config": int(k), "family": sets[f][0], "set": set_label(*sets[f]), "management": names[g],
                     "trades": int(n), "trades_per_day": n / len(trading_days), "win_rate": wins / n,
                     "mean_r": net / n, "mean_r_stress": stress / n, "t": t[j], "p_maxt": p_maxt[j],
                     "market_share": share[j], "markets": int(eligible[:, j].sum()),
                     "avg_win_r": win_sum / wins if wins else np.nan,
                     "avg_loss_r": loss_sum / (n - wins) if n > wins else np.nan,
                     "mean_pct": pct / n, "mean_minutes": minutes / n})
    table = pd.DataFrame(rows)
    table["passes_rest"] = ((table["trades"] >= MIN_TRADES) & (table["mean_r"] > 0) & (table["mean_r_stress"] > 0)
                            & (table["market_share"] >= MIN_MARKET_SHARE))
    table["stage_a"] = table["passes_rest"] & (table["p_maxt"] <= SIGNIFICANCE)
    candidates = carry(table)
    LAB.mkdir(parents=True, exist_ok=True)
    table.to_parquet(DISCOVERY_TABLE)
    CANDIDATES_FILE.write_text(json.dumps([int(k) for k in candidates["config"]], indent=2) + "\n", encoding="utf-8")
    years = pd.to_datetime((trading_days + origin) * 86400, unit="s").year  # the FX day's date, close enough for years
    by_year = {}
    for k in candidates["config"]:
        j = int(np.searchsorted(active, k))
        frame = pd.DataFrame({"year": years, "sum": table_sums[:, j], "count": table_counts[:, j]}).groupby("year").sum()
        by_year[int(k)] = (frame["sum"] / frame["count"]).round(3).to_dict()
    write_discovery_report(table, candidates, by_year, commit, len(trading_days), len(symbols))
    _append({"date": date.today().isoformat(), "event": "c10_discovery", "code_commit": commit,
             "preregistration_sha256": prereg, "configs": len(table), "trading_days": len(trading_days),
             "stage_a": int(table["stage_a"].sum()), "candidates": [int(k) for k in candidates["config"]]})
    return 0


def carry(table: pd.DataFrame) -> pd.DataFrame:
    """Every configuration that passes stage A, best t first, at most CARRY; the free places go to the best
    configuration of each other family set that passes everything except the p-value."""
    chosen = table[table["stage_a"]].sort_values("t", ascending=False).head(CARRY)
    used = set(chosen["set"])
    rest = table[table["passes_rest"] & ~table["stage_a"] & ~table["set"].isin(used)]
    best = rest.sort_values("t", ascending=False).drop_duplicates("set").head(CARRY - len(chosen))
    return pd.concat([chosen, best])


COLUMNS = ["set", "management", "trades", "trades_per_day", "win_rate", "mean_r", "mean_r_stress", "t", "p_maxt",
           "market_share", "avg_win_r", "avg_loss_r", "mean_minutes"]


def _fmt(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda v: f"{v:.4g}", max_colwidth=70)


def write_discovery_report(table, candidates, by_year, commit, trading_days, markets) -> None:
    best_per_set = table.sort_values("t", ascending=False).drop_duplicates("set")
    money = RISK_PER_TRADE * 1000
    text = (
        f"# C10: descoberta (2012-01 a 2026-08, {markets} mercados, {trading_days:,} dias de pregão)\n\n"
        f"Commit do código: {commit}. Configurações testadas: {len(table):,}. **Aprovadas na etapa A "
        f"(p máx-t ≤ 0,05 e o resto): {int(table['stage_a'].sum())}.** Unidade: R (a distância do stop inicial), "
        f"líquido do custo; com 0,5% de risco por operação numa conta de US$ 1.000, 1 R = US$ {money:.0f}.\n\n"
        f"## As 25 melhores por t\n\n```\n{_fmt(table.sort_values('t', ascending=False).head(25)[COLUMNS])}\n```\n\n"
        f"## A melhor de cada família e parâmetro\n\n```\n{_fmt(best_per_set[COLUMNS])}\n```\n\n"
        f"## Levadas ao dado intocado (2003 a 2011)\n\n```\n{_fmt(candidates[['config'] + COLUMNS])}\n```\n\n"
        f"Média por ano (R por operação):\n\n```\n{json.dumps(by_year, indent=1)}\n```\n")
    (c3.DOCS / "c10-discovery.md").write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def prepare_holdout() -> int:
    c3.prepare(SYMBOLS, raw=HOLDOUT_RAW, out=HOLDOUT_ARRAYS)
    manifest = c3.build_manifest(SYMBOLS, arrays=HOLDOUT_ARRAYS, sample=HOLDOUT_SAMPLE)
    manifest.to_csv(HOLDOUT_MANIFEST, index=False)
    included = manifest[manifest["symbol_included"]].groupby("symbol")["included"].sum()
    sys.stdout.write(f"holdout markets: {len(included)}\n{included.to_string()}\n")
    return 0


def candidate_worker(job: tuple) -> dict:
    """One market of the holdout (or of a placebo copy): the trades of every candidate configuration."""
    symbol, manifest_path, arrays, configs, coin_seed = job
    manifest = pd.read_csv(manifest_path)
    m1 = c3.load_included(symbol, manifest, arrays=Path(arrays))
    if coin_seed is not None:
        coins = np.random.default_rng([coin_seed, SYMBOLS.index(symbol)]).integers(0, 2, len(m1)) * 2 - 1
        m1 = c3.sign_flip(m1, coins.astype(np.int8))
    grid, _ = core.management_grid()
    sets = family_sets()
    wanted = sorted({k // len(grid) for k in configs})
    rows = [[k % len(grid) for k in configs if k // len(grid) == f] for f in wanted]
    days = fx_day_ids(m1)
    out = []
    for position, trades in run_trades(symbol, m1, [sets[f] for f in wanted], grid, rows):
        for trade in trades:
            config = wanted[position] * len(grid) + int(trade[0])
            out.append((config, int(days[int(trade[1])]), trade[3], trade[4]))
    return {"symbol": symbol, "trades": np.array(out, dtype=float).reshape(-1, 4)}


def evaluate(trades: pd.DataFrame, configs: list[int]) -> pd.DataFrame:
    rows = []
    all_days = np.unique(trades["day"]) if len(trades) else np.array([0])
    for k in configs:
        frame = trades[trades["config"] == k]
        if not len(frame):
            rows.append({"config": k, "trades": 0})
            continue
        day = np.searchsorted(all_days, frame["day"].to_numpy())
        sums, counts = stats.daily_tables(day, frame["net"].to_numpy(), np.zeros(len(frame)), len(all_days), 1)
        mean, _, t = stats.cluster_t(sums, counts)
        by_market = frame.groupby("symbol")["net"].agg(["mean", "size"])
        eligible = by_market[by_market["size"] >= MIN_MARKET_TRADES]
        rows.append({"config": k, "trades": len(frame), "win_rate": float((frame["net"] > 0).mean()),
                     "mean_r": float(mean[0]), "mean_r_stress": float(frame["stress"].mean()), "t": float(t[0]),
                     "p_one_sided": float(norm.sf(t[0])), "markets": len(eligible),
                     "market_share": float((eligible["mean"] > 0).mean()) if len(eligible) else 0.0})
    return pd.DataFrame(rows)


def run_candidates(symbols, manifest_path, arrays, configs, workers, coin_seed=None) -> pd.DataFrame:
    jobs = [(s, str(manifest_path), str(arrays), configs, coin_seed) for s in symbols]
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out in pool.map(candidate_worker, jobs):
            frame = pd.DataFrame(out["trades"], columns=["config", "day", "net", "stress"])
            frames.append(frame.assign(symbol=out["symbol"], config=frame["config"].astype(int)))
    return pd.concat(frames, ignore_index=True)


def holdout(workers: int) -> int:
    commit, prereg = _require_clean_and_registered()
    configs = json.loads(CANDIDATES_FILE.read_text())
    if not configs:
        sys.stdout.write("no candidates from discovery: nothing to confirm\n")
        return 0
    table = pd.read_parquet(DISCOVERY_TABLE).set_index("config")
    manifest = pd.read_csv(HOLDOUT_MANIFEST)
    symbols = [s for s in SYMBOLS if s in set(manifest.loc[manifest["symbol_included"], "symbol"])]
    result = evaluate(run_candidates(symbols, HOLDOUT_MANIFEST, HOLDOUT_ARRAYS, configs, workers), configs)
    result["p_holm"] = c3.holm(result["p_one_sided"].fillna(1.0).to_numpy())
    result["approved"] = ((result["p_holm"] <= SIGNIFICANCE) & (result["mean_r"] > 0) & (result["mean_r_stress"] > 0)
                          & (result["trades"] >= HOLDOUT_MIN_TRADES) & (result["market_share"] >= MIN_MARKET_SHARE))
    approved = [int(k) for k in result.loc[result["approved"], "config"]]
    placebo_rows = []
    if approved:
        discovery_symbols = [s for s in SYMBOLS if s in set(pd.read_csv(c3.MANIFEST).query("symbol_included")["symbol"])]
        for copy in range(PLACEBO_COPIES):
            flipped = evaluate(run_candidates(discovery_symbols, c3.MANIFEST, c3.ARRAYS, approved, workers,
                                              coin_seed=PLACEBO_SEED + copy), approved)
            placebo_rows.append(flipped.assign(copy=copy))
        placebo = pd.concat(placebo_rows)
        for k in approved:
            reached = int((placebo.loc[placebo["config"] == k, "mean_r"] >= table.loc[k, "mean_r"]).sum())
            result.loc[result["config"] == k, "placebo_reached"] = reached
            if reached > 1:
                result.loc[result["config"] == k, "approved"] = False
    result.insert(1, "set", [table.loc[k, "set"] for k in result["config"]])
    result.insert(2, "management", [table.loc[k, "management"] for k in result["config"]])
    result["discovery_mean_r"] = [table.loc[k, "mean_r"] for k in result["config"]]
    text = (f"# C10: o dado intocado (2003-01 a 2011-12, {len(symbols)} mercados)\n\nCommit do código: {commit}. "
            f"**Aprovadas: {int(result['approved'].sum())} de {len(configs)}.** Holm sobre as {len(configs)} levadas; "
            f"placebo com {PLACEBO_COPIES} cópias de sinal trocado só para as que passaram.\n\n"
            f"```\n{_fmt(result)}\n```\n")
    (c3.DOCS / "c10-holdout.md").write_text(text, encoding="utf-8")
    _append({"date": date.today().isoformat(), "event": "c10_holdout", "code_commit": commit,
             "preregistration_sha256": prereg, "approved": [int(k) for k in result.loc[result["approved"], "config"]],
             "results": result.to_dict(orient="records")})
    sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx_c10_lab")
    parser.add_argument("command", choices=["discover", "prepare-holdout", "holdout"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    if args.command == "discover":
        return discover(args.workers)
    if args.command == "prepare-holdout":
        return prepare_holdout()
    return holdout(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())

"""Lab of round 2: the grid of every family, in three samples (docs/forex/fast2-preregistration.md).

Order enforced by the registry (`fx_fast2_registry`): `calibrate` (200 placebo datasets of S0, measures the
false-approval rate of the whole criterion over the 212 combinations), `discover` (S0, 2011-01 to 2018-11),
`replicate` (S1, 2019-01 to 2024-08, only what passed discovery) and `confirm` (S2, 2024-09 to 2026-08,
frozen, only what was replicated). The machinery (simulation, costs, statistics) is round 1's.

    python -m scripts.research.fx_fast2_lab prepare | calibrate | discover | replicate | confirm

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx import strategy as fx
from scripts.research import fx_fast2_grid as grid
from scripts.research import fx_fast2_manifest as manifest2
from scripts.research import fx_fast2_registry as registry
from scripts.research import fx_fast_lab as lab
from scripts.research import fx_fast_placebo as placebo
from scripts.research import fx_fast_stats as stats
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast_manifest import load_m1

PRE_2019_DATA = Path("data/raw/fx_m1_hd_pre")
PRE_2019_LAB = Path("data/lab_pre")
PLACEBO_REPLICATES = 200
PLACEBO_SEED = 20260923
BOOTSTRAP_SEED = 20260922
MIN_TRADES = {"discovery": stats.MIN_TRADES_DISCOVERY, "replication": stats.MIN_TRADES_DISCOVERY,
              "confirmation": stats.MIN_TRADES_CONFIRMATION}
SAMPLES = {  # stage: (months, directory of the matrices, data manifest)
    "discovery": ((manifest2.S0_FIRST_MONTH, "2018-11"), PRE_2019_LAB, manifest2.DEFAULT_MANIFEST),
    "replication": (lab.DISCOVERY, lab.LAB_DIR, lab.DEFAULT_MANIFEST),
    "confirmation": (lab.CONFIRMATION, lab.LAB_DIR, lab.DEFAULT_MANIFEST),
}
NULL_T_FILE = PRE_2019_LAB / "null_t_discovery.npy"


def configurations() -> dict[str, lab.Configuration]:
    return grid.build_configurations(grid.load_grid())


def setup(stage: str):
    months, lab_dir, manifest_path = SAMPLES[stage]
    table = pd.read_csv(manifest_path)
    matrices = lab.open_matrices(lab_dir)
    windows = {pair: lab.included_windows(table, pair, months) for pair in (*ALL_PAIRS, lab.SYNTHETIC)}
    windows = {pair: w for pair, w in windows.items() if w}
    return matrices, windows, lab.day_universe(matrices, windows), lab.median_rates(matrices)


def _placebo_worker(arguments):
    seed, stage, keys = arguments
    matrices, windows, universe, rates = setup(stage)
    configs = {key: config for key, config in configurations().items() if key in keys}
    first = int(min(m[0, fx.BAR_TIME] for m in matrices.values()) // 60)
    last = int(max(m[-1, fx.BAR_TIME] for m in matrices.values()) // 60)
    return lab.run_sample(matrices, windows, universe, rates, configs,
                          coins=(placebo.coin_table(first, last, seed), first))


def run_placebos(stage: str, keys: list[str], replicates: int, workers: int) -> list[lab.SampleResult]:
    jobs = [(PLACEBO_SEED + i, stage, tuple(keys)) for i in range(replicates)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_placebo_worker, jobs))


def _report(name: str, title: str, table: pd.DataFrame, extra: str = "") -> Path:
    ordered = table.sort_values("t", ascending=False)
    return lab._write_report(name, title, ordered, extra)  # noqa: SLF001


def prepare() -> int:
    """The M1 matrices of S0 only: the months before it are reserved and stay out of the lab."""
    PRE_2019_LAB.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(f"{manifest2.S0_FIRST_MONTH}-01", tz="UTC")
    for pair in ALL_PAIRS:
        frame = load_m1(pair, PRE_2019_DATA)
        np.save(PRE_2019_LAB / f"{pair}.npy", fx.bars_to_matrix(frame[frame.index >= start]))
    return 0


def calibrate(replicates: int, workers: int) -> int:
    commit = lab.code_commit()
    configs = configurations()
    results = run_placebos("discovery", list(configs), replicates, workers)
    rate = lab.false_approval_rate(results, configs, stats.BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    null_t = lab.null_t_matrix(results)
    np.save(NULL_T_FILE, null_t)
    best = np.nanmax(np.where(np.isfinite(null_t), null_t, -np.inf), axis=1)
    summary = pd.DataFrame({"statistic": ["median of the best t", "p95 of the best t", "p99 of the best t"],
                            "t": [np.median(best), np.percentile(best, 95), np.percentile(best, 99)]})
    report = lab._write_report("fast2-calibration-report.md", "Calibração da rodada 2 em dados sem vantagem", summary,  # noqa: SLF001
                               f"Commit do código: {commit}. Conjuntos sintéticos de S0: {replicates}. Combinações: {len(configs)}. "
                               f"Taxa de falsa aprovação do critério inteiro: {rate:.1%} (limite {lab.FALSE_APPROVAL_LIMIT:.0%}).")
    sys.stdout.write(f"false approval rate {rate:.3f}; report {report}\n")
    if rate > lab.FALSE_APPROVAL_LIMIT:
        sys.stdout.write("criterion is miscalibrated: fix it before any real-data run; nothing registered\n")
        return 2
    registry.append_event({"event": "calibration_report", "false_pass_rate": rate, "replicates": replicates,
                           "code_commit": commit})
    return 0


def _run_stage(stage: str, keys: list[str], null_t: np.ndarray, report_name: str, title: str, *,
               allow_inconclusive: bool, commit: str) -> pd.DataFrame:
    configs = {key: config for key, config in configurations().items() if key in keys}
    matrices, windows, universe, rates = setup(stage)
    result = lab.run_sample(matrices, windows, universe, rates, configs)
    table = lab.evaluate(result, configs, null_t, draws=stats.BOOTSTRAP_DRAWS, min_trades=MIN_TRADES[stage],
                         allow_inconclusive=allow_inconclusive, bootstrap_seed=BOOTSTRAP_SEED)
    for row in table.to_dict("records"):
        registry.append_event({"event": "run", "configuration": row["config"], "sample": stage,
                               "code_commit": commit, "result": lab._plain(row)})  # noqa: SLF001
    _report(report_name, title, table)
    return table


def discover() -> int:
    commit = lab.code_commit()
    keys = list(configurations())
    table = _run_stage("discovery", keys, np.load(NULL_T_FILE), "fast2-discovery-report.md",
                       "Descoberta da rodada 2 (S0: 2011-01 a 2018-11)", allow_inconclusive=True, commit=commit)
    approved = table.loc[table["approved"], "config"].tolist()
    registry.append_event({"event": "discovery_report", "approved": approved, "report": "fast2-discovery-report.md"})
    sys.stdout.write(f"approved in discovery: {len(approved)} {approved}\n")
    return 0


def _later_stage(stage: str, replicates: int, workers: int) -> int:
    events = registry.read_events()
    report_kind = "discovery_report" if stage == "replication" else "replication_report"
    previous = [e for e in events if e["event"] == report_kind][-1]["approved"]
    if not previous:
        sys.stdout.write(f"nothing passed the stage before {stage}: stop and talk to Igor\n")
        return 0
    commit = lab.code_commit()
    null_t = lab.null_t_matrix(run_placebos(stage, previous, replicates, workers))
    name = f"fast2-{stage}-report.md"
    title = {"replication": "Réplica da rodada 2 (S1: 2019-01 a 2024-08)",
             "confirmation": "Confirmação da rodada 2 (S2: 2024-09 a 2026-08)"}[stage]
    table = _run_stage(stage, previous, null_t, name, title, allow_inconclusive=False, commit=commit)
    approved = table.loc[table["approved"], "config"].tolist()
    if stage == "replication":
        registry.append_event({"event": "replication_report", "approved": approved, "report": name})
    sys.stdout.write(f"passed {stage}: {len(approved)} {approved}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["prepare", "calibrate", "discover", "replicate", "confirm"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=PLACEBO_REPLICATES)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare()
    if args.command == "calibrate":
        return calibrate(args.replicates, args.workers)
    if args.command == "discover":
        return discover()
    return _later_stage("replication" if args.command == "replicate" else "confirmation", args.replicates, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())

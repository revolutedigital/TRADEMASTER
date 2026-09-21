"""Power of the round-2 criterion, measured by positive control on the placebo datasets.

Round 2 calibrates the false-approval rate (no edge in the data). This measures the other error: if a
combination had a true net edge of `nu` R per trade, how often would the whole criterion approve it?

No new simulation is needed for the injected edge. Adding a constant to every trade of one combination
shifts its mean and leaves its cluster standard error untouched (the residuals `sums - mean * counts`
do not change), and the bootstrap statistics are centred, so they do not move either. The shifted
combination has t = nu / SE, and its two p-values come from the unmodified null distributions: the
best t of the bootstrap of the same replicate (key A) and the best t of the other placebo replicates
(key B). The other rules (mean R and stress mean above zero, 60% of the pairs positive, 300 trades) use the
replicate's own tables shifted by the same constant.

Informative only: nothing here enters the criterion or the registry. It uses placebo data, never real results.

    python -m scripts.research.fx_fast2_power --replicates 40 --workers 9

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from scripts.research import fx_fast2_lab as lab2
from scripts.research import fx_fast_lab as lab
from scripts.research import fx_fast_stats as stats

PROBES = (
    "F1a.rr1.5.w0.3-1.2.se11:00.x16:30", "F1b.rr1.5.w0.3-1.2.se11:00.x16:00", "F2a.in15:00.out15:55.atr3.0",
    "F2b.in16:05.out17:00.atr3.0", "F3.th4.0.stop0.5.ret0.5.hold12", "F5a.z2.0.stop3.5.hold120",
)
NET_EDGES = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)  # true net R per trade under the base costs


def bootstrap_best(sums: np.ndarray, counts: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """The best studentised bootstrap t over all combinations, in each resample (the null of key A)."""
    mean, _, t_real = stats.cluster_t(sums, counts)
    weights = stats._stationary_weights(sums.shape[0], draws, 1.0 / stats.MEAN_BLOCK_DAYS, seed)  # noqa: SLF001
    total = weights @ counts
    with np.errstate(divide="ignore", invalid="ignore"):
        resampled_mean = (weights @ sums) / total
        squares = (weights @ sums**2 - 2.0 * resampled_mean * (weights @ (sums * counts))
                   + resampled_mean**2 * (weights @ counts**2))
        error = np.sqrt(np.maximum(squares, 0.0)) / total
        t_star = (resampled_mean - mean) / error
    return np.where(np.isfinite(t_star), t_star, -np.inf).max(axis=1)


def approves(result: lab.SampleResult, k: int, net_edge: float, standard_error: float, mean: float,
             best_bootstrap: np.ndarray, best_placebo: np.ndarray, applicable: Sequence[str]) -> bool:
    """Would the criterion approve combination `k` if its true net mean were `net_edge`?"""
    shift = net_edge - mean
    t = net_edge / standard_error
    p_bootstrap = (1.0 + (best_bootstrap >= t).sum()) / (len(best_bootstrap) + 1.0)
    p_placebo = (1.0 + (best_placebo >= t).sum()) / (len(best_placebo) + 1.0)
    means = result.pair_means[result.keys[k]]
    share = sum(means[p] + shift > 0 for p in applicable if p in means) / len(applicable)
    return bool(
        p_bootstrap <= stats.SIGNIFICANCE and p_placebo <= stats.SIGNIFICANCE and net_edge > 0
        and share >= stats.MIN_PAIR_SHARE and result.trades[k] >= stats.MIN_TRADES_DISCOVERY
        and result.stress_mean[k] + shift > 0
    )


def power_table(results: list[lab.SampleResult], null_t: np.ndarray, configs, probes: Sequence[str] = PROBES,
                net_edges: Sequence[float] = NET_EDGES, draws: int = stats.BOOTSTRAP_DRAWS) -> pd.DataFrame:
    """Share of the replicates in which each probe would be approved, by true net edge."""
    keys = results[0].keys
    null_best = np.where(np.isfinite(null_t), null_t, -np.inf).max(axis=1)
    approved = {(probe, edge): 0 for probe in probes for edge in net_edges}
    trades = {probe: [] for probe in probes}
    for i, result in enumerate(results):
        mean, error, _ = stats.cluster_t(result.sums, result.counts)
        best = bootstrap_best(result.sums, result.counts, draws, lab2.BOOTSTRAP_SEED)
        others = np.delete(null_best, i)
        for probe in probes:
            k = keys.index(probe)
            trades[probe].append(result.trades[k])
            if not np.isfinite(error[k]) or error[k] <= 0:
                continue
            for edge in net_edges:
                approved[(probe, edge)] += approves(result, k, edge, error[k], mean[k], best, others, configs[probe].pairs)
    table = pd.DataFrame({f"{edge:+.2f} R": [approved[(probe, edge)] / len(results) for probe in probes] for edge in net_edges},
                         index=list(probes))
    table.insert(0, "trades (median)", [int(np.median(trades[probe])) for probe in probes])
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--workers", type=int, default=9)
    arguments = parser.parse_args(argv)
    configs = lab2.configurations()
    results = lab2.run_placebos("discovery", list(configs), arguments.replicates, arguments.workers)
    null_t = np.load(lab2.NULL_T_FILE)[: arguments.replicates]  # the same seeds as the first replicates of the calibration
    table = power_table(results, null_t, configs)
    body = table.to_string(float_format=lambda value: f"{value:.2f}")
    report = lab.REPORT_DIR / "fast2-power-report.md"
    report.write_text(
        "# Poder da rodada 2 por controle positivo (informativo)\n\n"
        f"Probabilidade de o critério inteiro aprovar uma combinação cujo R líquido verdadeiro por trade seja o da coluna, "
        f"medida em {arguments.replicates} conjuntos placebo de S0 (as mesmas sementes dos primeiros da calibração). "
        "O estresse (spread ×2) também precisa ficar acima de zero, o que impõe um piso acima do custo base.\n\n"
        f"```\n{body}\n```\n", encoding="utf-8")
    sys.stdout.write(body + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

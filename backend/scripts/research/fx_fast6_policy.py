"""Evaluate and freeze top-ranked Q2 runner policies for round 6."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.fx.instruments import ConversionRates
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_materialize import _median_rates
from scripts.research.fx_fast5_events import TRAILING_DISTANCES_R
from scripts.research.fx_fast5_model import DEFAULT_OUTPUT as MODEL_ROOT
from scripts.research.fx_fast5_policy import (
    _add_outcomes,
    _costs,
    _rebuild_ticks_and_validate,
    enforce_common_non_overlap,
    policy_metrics,
    select_candidate_side,
    selection_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast6" / "policies"
TOP_FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)


def top_ranked_candidates(candidates: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if candidates.empty:
        return candidates.copy()
    count = math.ceil(fraction * len(candidates))
    ranked = candidates.assign(decision_time=candidates.index).sort_values(
        ["score", "probability_base", "decision_time"],
        ascending=[False, False, True],
        kind="stable",
    )
    return ranked.iloc[:count].drop(columns="decision_time").sort_index(kind="stable")


def evaluate_pair(
    pair: str, model_root: Path, output: Path, rates: ConversionRates
) -> dict[str, object]:
    predictions = pd.read_parquet(model_root / f"{pair}-q2-probabilities.parquet")
    directional = select_candidate_side(predictions, float("-inf"))
    pool = top_ranked_candidates(directional, max(TOP_FRACTIONS))
    cleaned, instrument = _rebuild_ticks_and_validate(pair, predictions)
    costs = _costs(pair, cleaned, rates)
    attempts: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)
    for trail in TRAILING_DISTANCES_R:
        outcomes = _add_outcomes(pool, cleaned, instrument, costs, trail)
        for fraction in TOP_FRACTIONS:
            ranked = top_ranked_candidates(directional, fraction)
            eligible = outcomes.loc[ranked.index]
            trades = enforce_common_non_overlap(eligible)
            metrics = policy_metrics(trades)
            fraction_label = str(fraction * 100).replace(".", "p")
            trail_label = str(trail).replace(".", "p")
            attempt_id = f"{pair}-top{fraction_label}-t{trail_label}"
            trades.to_parquet(
                output / f"{attempt_id}-trades.parquet", compression="zstd", index=True
            )
            attempts.append(
                {
                    "attempt": attempt_id,
                    "top_fraction": fraction,
                    "authorized_events": len(ranked),
                    "score_cutoff": float(ranked["score"].min()),
                    "trail_distance_r": trail,
                    **asdict(metrics),
                    "selection_gate": selection_gate(metrics),
                }
            )
    del cleaned
    gc.collect()
    passed = [attempt for attempt in attempts if attempt["selection_gate"]]
    selected = max(
        passed,
        key=lambda attempt: (
            attempt["mean_stress_r"],
            attempt["lower_95_base_r"],
            -attempt["trades"],
        ),
        default=None,
    )
    report = {
        "pair": pair,
        "directional_events": len(directional),
        "simulated_top_10_percent": len(pool),
        "attempts": attempts,
        "selected_policy": selected,
        "status": "policy_frozen" if selected else "do_not_trade",
    }
    (output / f"{pair}-q2-ranked-grid.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def evaluate_all(
    pairs: tuple[str, ...],
    model_root: Path = MODEL_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    if set(pairs) - set(ALL_PAIRS):
        raise ValueError("unknown pair")
    rates = _median_rates()
    reports = []
    for pair in pairs:
        report = evaluate_pair(pair, model_root, output, rates)
        reports.append(report)
        selected = report["selected_policy"]
        summary = (
            f"top={selected['top_fraction']}, trail={selected['trail_distance_r']}, "
            f"stress={selected['mean_stress_r']:.4f}R"
            if selected
            else "do_not_trade"
        )
        sys.stdout.write(f"{pair}: {summary}\n")
        sys.stdout.flush()
    manifest = {
        "kind": "round6_q2_ranked_frozen_policies",
        "protected_samples_opened": False,
        "year_2022_opened": False,
        "attempt_count": len(pairs) * len(TOP_FRACTIONS) * len(TRAILING_DISTANCES_R),
        "pairs": reports,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen-policies.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=",".join(ALL_PAIRS))
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    pairs = tuple(value.strip().upper() for value in arguments.pairs.split(",") if value.strip())
    evaluate_all(pairs, arguments.model_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

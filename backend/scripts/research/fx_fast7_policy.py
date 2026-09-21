"""Run round-7 stage A: earlier breakeven activation on the frozen global Top 5%."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_materialize import _median_rates, _sha256
from scripts.research.fx_fast5_events import simulate_trailing_batch
from scripts.research.fx_fast5_policy import _costs, _rebuild_ticks_and_validate
from scripts.research.fx_fast6_model import DEFAULT_OUTPUT as MODEL_ROOT
from scripts.research.fx_fast6_model import strongest_direction
from scripts.research.fx_fast6_policy import (
    as_candidate_multiindex,
    causal_daily_cutoffs,
    enforce_global_portfolio,
    global_policy_metrics,
    qualifying_candidates,
    selection_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast7" / "stage_a"
TOP_FRACTION = 0.05
ACTIVATION_LEVELS_R = (0.1, 0.2, 0.3)
TRAIL_DISTANCE_R = 0.5
ACTIVATION_SECONDS = 600


def stage_a_candidates(model_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(model_root / "reference-probabilities.parquet")
    strongest = strongest_direction(predictions)
    floor = float(manifest["absolute_floors"][str(TOP_FRACTION)])
    cutoffs = causal_daily_cutoffs(strongest, TOP_FRACTION, floor)
    return predictions, qualifying_candidates(strongest, cutoffs)


def _add_management_outcome(
    candidates: pd.DataFrame,
    cleaned: pd.DataFrame,
    instrument,
    costs,
    activation_r: float,
) -> pd.DataFrame:
    enriched = candidates.copy()
    for scenario in ("base", "stress"):
        outcomes = simulate_trailing_batch(
            cleaned,
            candidates,
            instrument,
            costs,
            scenario=scenario,
            activation_r=activation_r,
            activation_seconds=ACTIVATION_SECONDS,
            trail_distance_r=TRAIL_DISTANCE_R,
        )
        for name in outcomes.columns:
            enriched[f"{name}_{scenario}"] = outcomes[name].to_numpy()
    return enriched.rename(columns={"result_r_base": "base_r", "result_r_stress": "stress_r"})


def simulate_stage_a(
    predictions: pd.DataFrame, candidates: pd.DataFrame
) -> dict[float, pd.DataFrame]:
    rates = _median_rates()
    parts: dict[float, list[pd.DataFrame]] = {activation: [] for activation in ACTIVATION_LEVELS_R}
    for pair in ALL_PAIRS:
        pair_candidates = candidates.loc[candidates["pair"] == pair]
        if pair_candidates.empty:
            continue
        pair_predictions = predictions.loc[predictions["pair"] == pair]
        cleaned, instrument = _rebuild_ticks_and_validate(pair, pair_predictions)
        costs = _costs(pair, cleaned, rates)
        for activation in ACTIVATION_LEVELS_R:
            outcome = _add_management_outcome(
                pair_candidates, cleaned, instrument, costs, activation
            )
            parts[activation].append(as_candidate_multiindex(outcome))
        sys.stdout.write(f"{pair}: {len(pair_candidates):,} Top-5% candidates simulated\n")
        sys.stdout.flush()
        del cleaned
        gc.collect()
    return {
        activation: pd.concat(items).sort_index(kind="stable") if items else pd.DataFrame()
        for activation, items in parts.items()
    }


def evaluate_stage_a(
    model_root: Path = MODEL_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    predictions, candidates = stage_a_candidates(model_root)
    outcomes = simulate_stage_a(predictions, candidates)
    output.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    for activation in ACTIVATION_LEVELS_R:
        trades = enforce_global_portfolio(outcomes[activation])
        metrics = global_policy_metrics(trades)
        label = str(activation).replace(".", "p")
        trade_path = output / f"activation-{label}-trades.parquet"
        trades.to_parquet(trade_path, compression="zstd", index=True)
        attempts.append(
            {
                "activation_r": activation,
                "activation_seconds": ACTIVATION_SECONDS,
                "trail_distance_r": TRAIL_DISTANCE_R,
                "qualified_events": len(candidates),
                **asdict(metrics),
                "stage_a_gate": selection_gate(metrics),
                "trades_sha256": _sha256(trade_path),
            }
        )
    passed = [attempt for attempt in attempts if attempt["stage_a_gate"]]
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
        "kind": "round7_q2_stage_a_early_protection",
        "protected_samples_opened": False,
        "year_2022_opened": False,
        "top_fraction": TOP_FRACTION,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "selected_activation": selected,
        "status": "stage_a_pass_requires_stage_b" if selected else "stage_a_rejected",
    }
    report_path = output / "stage-a-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"round7 stage A: {report['status']}, sha256={_sha256(report_path)}\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    evaluate_stage_a(arguments.model_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

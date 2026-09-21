"""Run round 8: fixed early protection with preregistered pre-activation time stops."""

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
from scripts.research.fx_fast6_policy import (
    as_candidate_multiindex,
    enforce_global_portfolio,
    global_policy_metrics,
    selection_gate,
)
from scripts.research.fx_fast7_policy import stage_a_candidates

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast8" / "q2"
ACTIVATION_R = 0.1
TRAIL_DISTANCE_R = 0.5
PREACTIVATION_TIMEOUT_SECONDS = (60, 120, 210)


def _add_time_stop_outcome(
    candidates: pd.DataFrame,
    cleaned: pd.DataFrame,
    instrument,
    costs,
    timeout_seconds: int,
) -> pd.DataFrame:
    enriched = candidates.copy()
    for scenario in ("base", "stress"):
        outcomes = simulate_trailing_batch(
            cleaned,
            candidates,
            instrument,
            costs,
            scenario=scenario,
            activation_r=ACTIVATION_R,
            activation_seconds=timeout_seconds,
            trail_distance_r=TRAIL_DISTANCE_R,
        )
        for name in outcomes.columns:
            enriched[f"{name}_{scenario}"] = outcomes[name].to_numpy()
    return enriched.rename(columns={"result_r_base": "base_r", "result_r_stress": "stress_r"})


def simulate_time_stops(
    predictions: pd.DataFrame, candidates: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    rates = _median_rates()
    parts: dict[int, list[pd.DataFrame]] = {
        timeout: [] for timeout in PREACTIVATION_TIMEOUT_SECONDS
    }
    for pair in ALL_PAIRS:
        pair_candidates = candidates.loc[candidates["pair"] == pair]
        if pair_candidates.empty:
            continue
        pair_predictions = predictions.loc[predictions["pair"] == pair]
        cleaned, instrument = _rebuild_ticks_and_validate(pair, pair_predictions)
        costs = _costs(pair, cleaned, rates)
        for timeout in PREACTIVATION_TIMEOUT_SECONDS:
            outcomes = _add_time_stop_outcome(
                pair_candidates, cleaned, instrument, costs, timeout
            )
            parts[timeout].append(as_candidate_multiindex(outcomes))
        sys.stdout.write(f"{pair}: {len(pair_candidates):,} Top-5% candidates simulated\n")
        sys.stdout.flush()
        del cleaned
        gc.collect()
    return {
        timeout: pd.concat(items).sort_index(kind="stable") if items else pd.DataFrame()
        for timeout, items in parts.items()
    }


def evaluate_time_stops(
    model_root: Path = MODEL_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    predictions, candidates = stage_a_candidates(model_root)
    outcomes = simulate_time_stops(predictions, candidates)
    output.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    for timeout in PREACTIVATION_TIMEOUT_SECONDS:
        trades = enforce_global_portfolio(outcomes[timeout])
        metrics = global_policy_metrics(trades)
        trade_path = output / f"timeout-{timeout}s-trades.parquet"
        trades.to_parquet(trade_path, compression="zstd", index=True)
        attempts.append(
            {
                "attempt": f"global-top5-a0p1-t0p5-timeout-{timeout}s",
                "top_fraction": 0.05,
                "activation_r": ACTIVATION_R,
                "trail_distance_r": TRAIL_DISTANCE_R,
                "preactivation_timeout_seconds": timeout,
                "qualified_events": len(candidates),
                **asdict(metrics),
                "selection_gate": selection_gate(metrics),
                "trades_sha256": _sha256(trade_path),
            }
        )
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
        "kind": "round8_q2_preactivation_time_stop",
        "protected_samples_opened": False,
        "year_2022_opened": False,
        "attempt_count": len(attempts),
        "cumulative_attempt_count": 24,
        "attempts": attempts,
        "selected_policy": selected,
        "status": "policy_frozen" if selected else "architecture_rejected",
    }
    report_path = output / "frozen-policy.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        f"round8: {len(attempts)} attempts, {report['status']}, "
        f"sha256={_sha256(report_path)}\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    evaluate_time_stops(arguments.model_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

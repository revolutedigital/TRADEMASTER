"""Evaluate and freeze the causal global Top-P runner policy in 2021-Q2."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx.instruments import ConversionRates
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_diagnostic import _fx_day, stationary_bootstrap_lower_bound
from scripts.research.fx_fast4_materialize import _median_rates, _sha256
from scripts.research.fx_fast5_events import TRAILING_DISTANCES_R
from scripts.research.fx_fast5_policy import (
    _add_outcomes,
    _costs,
    _profit_factor,
    _rebuild_ticks_and_validate,
)
from scripts.research.fx_fast6_model import DEFAULT_OUTPUT as MODEL_ROOT
from scripts.research.fx_fast6_model import TOP_FRACTIONS, strongest_direction

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast6" / "policies"
Q2_START = pd.Timestamp("2021-04-01T06:00:00Z")
Q2_END = pd.Timestamp("2021-06-30T18:00:00Z")
REFERENCE_DAYS = 60
MAX_PORTFOLIO_POSITIONS = 3
Q2_MONTHS = (
    pd.Period("2021-04", freq="M"),
    pd.Period("2021-05", freq="M"),
    pd.Period("2021-06", freq="M"),
)


@dataclass(frozen=True)
class GlobalPolicyMetrics:
    trades: int
    mean_base_r: float
    mean_stress_r: float
    lower_95_base_r: float
    stress_profit_factor: float
    positive_month_fraction: float
    participating_pairs: int
    maximum_pair_profit_fraction: float


def with_fx_day(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["fx_day"] = _fx_day(enriched.index)
    return enriched


def causal_daily_cutoffs(
    reference: pd.DataFrame,
    fraction: float,
    absolute_floor: float,
    *,
    start: pd.Timestamp = Q2_START,
    end: pd.Timestamp = Q2_END,
) -> dict[pd.Timestamp, float]:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    prepared = with_fx_day(reference)
    days = pd.date_range(
        _fx_day(pd.DatetimeIndex([start]))[0],
        _fx_day(pd.DatetimeIndex([end]))[0],
        freq="D",
    )
    cutoffs: dict[pd.Timestamp, float] = {}
    for day in days:
        prior_start = day - pd.Timedelta(days=REFERENCE_DAYS)
        prior = prepared.loc[
            (prepared["fx_day"] >= prior_start) & (prepared["fx_day"] < day),
            "trusted_score",
        ]
        if prior.empty:
            cutoffs[day] = float("inf")
        else:
            cutoffs[day] = max(float(prior.quantile(1 - fraction)), absolute_floor)
    return cutoffs


def qualifying_candidates(
    reference: pd.DataFrame,
    cutoffs: dict[pd.Timestamp, float],
    *,
    start: pd.Timestamp = Q2_START,
    end: pd.Timestamp = Q2_END,
) -> pd.DataFrame:
    q2 = with_fx_day(reference.loc[(reference.index >= start) & (reference.index < end)])
    threshold = q2["fx_day"].map(cutoffs)
    eligible = threshold.notna() & (q2["trusted_score"] >= threshold)
    return q2.loc[eligible].drop(columns="fx_day")


def _candidate_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [frame.index, frame["pair"].to_numpy()], names=("decision_time", "pair")
    )


def simulate_candidate_pool(
    predictions: pd.DataFrame,
    pool: pd.DataFrame,
    rates: ConversionRates,
) -> dict[float, pd.DataFrame]:
    by_trail: dict[float, list[pd.DataFrame]] = {trail: [] for trail in TRAILING_DISTANCES_R}
    for pair in ALL_PAIRS:
        pair_pool = pool.loc[pool["pair"] == pair]
        if pair_pool.empty:
            continue
        pair_predictions = predictions.loc[predictions["pair"] == pair]
        cleaned, instrument = _rebuild_ticks_and_validate(pair, pair_predictions)
        costs = _costs(pair, cleaned, rates)
        for trail in TRAILING_DISTANCES_R:
            outcomes = _add_outcomes(pair_pool, cleaned, instrument, costs, trail)
            outcomes.index = _candidate_index(outcomes)
            by_trail[trail].append(outcomes)
        del cleaned
        gc.collect()
    return {
        trail: pd.concat(parts).sort_index(kind="stable") if parts else pd.DataFrame()
        for trail, parts in by_trail.items()
    }


def enforce_global_portfolio(
    candidates: pd.DataFrame, *, max_positions: int = MAX_PORTFOLIO_POSITIONS
) -> pd.DataFrame:
    if max_positions < 1:
        raise ValueError("max_positions must be positive")
    if candidates.empty:
        return candidates.copy()
    required = {
        "base_r",
        "stress_r",
        "holding_seconds_base",
        "holding_seconds_stress",
        "trusted_score",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"outcome columns missing: {sorted(missing)}")
    valid = candidates[list(required)].notna().all(axis=1)
    ordered = (
        candidates.loc[valid]
        .reset_index()
        .sort_values(["decision_time", "trusted_score"], ascending=[True, False], kind="stable")
    )
    keep = np.zeros(len(ordered), dtype=bool)
    open_positions: list[tuple[pd.Timestamp, str]] = []
    for position, row in ordered.iterrows():
        timestamp = row["decision_time"]
        open_positions = [item for item in open_positions if item[0] > timestamp]
        if len(open_positions) >= max_positions:
            continue
        pair = str(row["pair"])
        if any(open_pair == pair for _, open_pair in open_positions):
            continue
        keep[position] = True
        holding = max(row["holding_seconds_base"], row["holding_seconds_stress"])
        open_positions.append((timestamp + pd.Timedelta(seconds=float(holding)), pair))
    selected = ordered.loc[keep].set_index(["decision_time", "pair"])
    return selected.sort_index(kind="stable")


def global_policy_metrics(trades: pd.DataFrame) -> GlobalPolicyMetrics:
    if trades.empty:
        return GlobalPolicyMetrics(0, *(float("nan"),) * 5, 0, float("nan"))
    flat = trades.reset_index().set_index("decision_time")
    months = flat.index.tz_localize(None).to_period("M")
    monthly = flat.assign(month=months).groupby("month")["stress_r"].mean()
    positive_months = sum(float(monthly.get(month, float("nan"))) > 0 for month in Q2_MONTHS)
    pair_profit = flat.groupby("pair")["base_r"].sum().clip(lower=0)
    positive_profit = float(pair_profit.sum())
    concentration = (
        float(pair_profit.max() / positive_profit) if positive_profit > 0 else float("inf")
    )
    return GlobalPolicyMetrics(
        trades=len(flat),
        mean_base_r=float(flat["base_r"].mean()),
        mean_stress_r=float(flat["stress_r"].mean()),
        lower_95_base_r=stationary_bootstrap_lower_bound(flat),
        stress_profit_factor=_profit_factor(flat["stress_r"]),
        positive_month_fraction=positive_months / len(Q2_MONTHS),
        participating_pairs=int(flat["pair"].nunique()),
        maximum_pair_profit_fraction=concentration,
    )


def selection_gate(metrics: GlobalPolicyMetrics) -> bool:
    return (
        metrics.trades >= 100
        and metrics.mean_base_r > 0
        and metrics.mean_stress_r > 0
        and metrics.positive_month_fraction >= 2 / 3
        and metrics.stress_profit_factor > 1.05
        and metrics.participating_pairs >= 4
        and metrics.maximum_pair_profit_fraction <= 0.35
    )


def evaluate(
    model_root: Path = MODEL_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(model_root / "reference-probabilities.parquet")
    strongest = strongest_direction(predictions)
    floors = manifest["absolute_floors"]
    cutoffs = {
        fraction: causal_daily_cutoffs(strongest, fraction, float(floors[str(fraction)]))
        for fraction in TOP_FRACTIONS
    }
    qualifying = {
        fraction: qualifying_candidates(strongest, cutoffs[fraction]) for fraction in TOP_FRACTIONS
    }
    pool = qualifying[max(TOP_FRACTIONS)]
    outcomes = simulate_candidate_pool(predictions, pool, _median_rates())
    output.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, object]] = []
    for trail in TRAILING_DISTANCES_R:
        trail_outcomes = outcomes[trail]
        for fraction in TOP_FRACTIONS:
            candidate_index = _candidate_index(qualifying[fraction])
            eligible = trail_outcomes.loc[trail_outcomes.index.isin(candidate_index)]
            trades = enforce_global_portfolio(eligible)
            metrics = global_policy_metrics(trades)
            fraction_label = str(fraction * 100).replace(".", "p")
            trail_label = str(trail).replace(".", "p")
            attempt_id = f"global-top{fraction_label}-t{trail_label}"
            trade_path = output / f"{attempt_id}-trades.parquet"
            trades.to_parquet(trade_path, compression="zstd", index=True)
            attempts.append(
                {
                    "attempt": attempt_id,
                    "top_fraction": fraction,
                    "trail_distance_r": trail,
                    "qualified_events": len(qualifying[fraction]),
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
        "kind": "round6_q2_causal_global_top_p",
        "protected_samples_opened": False,
        "year_2022_opened": False,
        "attempt_count": len(attempts),
        "maximum_positions": MAX_PORTFOLIO_POSITIONS,
        "absolute_floors": floors,
        "attempts": attempts,
        "selected_policy": selected,
        "status": "policy_frozen" if selected else "no_global_policy",
    }
    report_path = output / "frozen-policy.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        f"round6: {len(attempts)} attempts, "
        f"{report['status']}, pool={len(pool):,}, sha256={_sha256(report_path)}\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    evaluate(arguments.model_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

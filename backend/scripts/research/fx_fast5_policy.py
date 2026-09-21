"""Evaluate and freeze round-5 Q2 trailing policies without opening H2 or 2022."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim.costs import FUSION_ZERO, STRESS
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_diagnostic import stationary_bootstrap_lower_bound
from scripts.research.fx_fast4_events import EventCosts, build_feature_frame
from scripts.research.fx_fast4_materialize import _commission_pips, _load_archives, _median_rates
from scripts.research.fx_fast5_events import TRAILING_DISTANCES_R, simulate_trailing_batch
from scripts.research.fx_fast5_materialize import context_archives
from scripts.research.fx_fast5_model import DEFAULT_OUTPUT as MODEL_ROOT

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast5" / "policies"
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
Q2_MONTHS = (
    pd.Period("2021-04", freq="M"),
    pd.Period("2021-05", freq="M"),
    pd.Period("2021-06", freq="M"),
)


@dataclass(frozen=True)
class PolicyMetrics:
    trades: int
    mean_base_r: float
    mean_stress_r: float
    lower_95_base_r: float
    stress_profit_factor: float
    positive_month_fraction: float


def select_candidate_side(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    candidates = predictions.loc[predictions["score"] >= threshold].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values(["score", "probability_base"], kind="stable")
        .groupby(level=0, sort=False)
        .tail(1)
        .sort_index(kind="stable")
    )


def enforce_common_non_overlap(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep one position using the later base/stress exit as the common availability time."""
    if candidates.empty:
        return candidates.copy()
    required = {"base_r", "stress_r", "holding_seconds_base", "holding_seconds_stress"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"outcome columns missing: {sorted(missing)}")
    valid = candidates[list(required)].notna().all(axis=1)
    ordered = candidates.loc[valid].sort_index(kind="stable")
    keep = np.zeros(len(ordered), dtype=bool)
    available_at = pd.Timestamp.min.tz_localize("UTC")
    for position, (timestamp, row) in enumerate(ordered.iterrows()):
        if timestamp < available_at:
            continue
        keep[position] = True
        holding = max(row["holding_seconds_base"], row["holding_seconds_stress"])
        available_at = timestamp + pd.Timedelta(seconds=float(holding))
    return ordered.iloc[keep]


def _profit_factor(values: pd.Series) -> float:
    wins = float(values.clip(lower=0).sum())
    losses = float(-values.clip(upper=0).sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def policy_metrics(trades: pd.DataFrame) -> PolicyMetrics:
    if trades.empty:
        return PolicyMetrics(0, *(float("nan"),) * 5)
    month = trades.index.tz_localize(None).to_period("M")
    monthly = trades.assign(month=month).groupby("month")["stress_r"].mean()
    positive_months = sum(float(monthly.get(period, float("nan"))) > 0 for period in Q2_MONTHS)
    return PolicyMetrics(
        trades=len(trades),
        mean_base_r=float(trades["base_r"].mean()),
        mean_stress_r=float(trades["stress_r"].mean()),
        lower_95_base_r=stationary_bootstrap_lower_bound(trades),
        stress_profit_factor=_profit_factor(trades["stress_r"]),
        positive_month_fraction=positive_months / len(Q2_MONTHS),
    )


def selection_gate(metrics: PolicyMetrics) -> bool:
    return (
        metrics.trades >= 30
        and metrics.mean_base_r > 0
        and metrics.mean_stress_r > 0
        and metrics.positive_month_fraction >= 2 / 3
        and metrics.stress_profit_factor > 1.05
    )


def _costs(pair: str, cleaned: pd.DataFrame, rates: ConversionRates) -> EventCosts:
    median_price = float((cleaned["bid"] + cleaned["ask"]).median() * 0.5)
    return EventCosts(
        base_commission_pips=_commission_pips(pair, median_price, rates, FUSION_ZERO),
        stress_commission_pips=_commission_pips(pair, median_price, rates, STRESS),
    )


def _add_outcomes(
    pool: pd.DataFrame,
    cleaned: pd.DataFrame,
    instrument: Instrument,
    costs: EventCosts,
    trail_distance_r: float,
) -> pd.DataFrame:
    enriched = pool.copy()
    for scenario in ("base", "stress"):
        outcomes = simulate_trailing_batch(
            cleaned,
            pool,
            instrument,
            costs,
            scenario=scenario,
            trail_distance_r=trail_distance_r,
        )
        for name in outcomes.columns:
            enriched[f"{name}_{scenario}"] = outcomes[name].to_numpy()
    enriched = enriched.rename(columns={"result_r_base": "base_r", "result_r_stress": "stress_r"})
    return enriched


def _rebuild_ticks_and_validate(
    pair: str, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, Instrument]:
    ticks = _load_archives(context_archives(pair, 2021))
    instrument = Instrument.from_symbol(pair)
    cleaned, features = build_feature_frame(ticks, instrument)
    expected = features.loc[predictions.index, "decision_index"].to_numpy(dtype=np.int64)
    observed = predictions["decision_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected, observed):
        raise ValueError(f"prediction indices do not match rebuilt ticks for {pair}")
    del ticks, features
    gc.collect()
    return cleaned, instrument


def evaluate_pair(
    pair: str, model_root: Path, output: Path, rates: ConversionRates
) -> dict[str, object]:
    predictions = pd.read_parquet(model_root / f"{pair}-q2-probabilities.parquet")
    pool = select_candidate_side(predictions, min(THRESHOLDS))
    attempts: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)
    if pool.empty:
        for threshold in THRESHOLDS:
            for trail in TRAILING_DISTANCES_R:
                attempts.append(
                    {
                        "threshold": threshold,
                        "trail_distance_r": trail,
                        **asdict(policy_metrics(pool)),
                        "selection_gate": False,
                    }
                )
    else:
        cleaned, instrument = _rebuild_ticks_and_validate(pair, predictions)
        costs = _costs(pair, cleaned, rates)
        for trail in TRAILING_DISTANCES_R:
            outcomes = _add_outcomes(pool, cleaned, instrument, costs, trail)
            for threshold in THRESHOLDS:
                eligible = outcomes.loc[outcomes["score"] >= threshold]
                trades = enforce_common_non_overlap(eligible)
                metrics = policy_metrics(trades)
                attempt_id = f"{pair}-p{int(threshold * 100)}-t{str(trail).replace('.', 'p')}"
                trade_path = output / f"{attempt_id}-trades.parquet"
                trades.to_parquet(trade_path, compression="zstd", index=True)
                attempts.append(
                    {
                        "attempt": attempt_id,
                        "threshold": threshold,
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
        "candidate_events_at_0.55": len(pool),
        "attempts": attempts,
        "selected_policy": selected,
        "status": "policy_frozen" if selected else "do_not_trade",
    }
    (output / f"{pair}-q2-grid.json").write_text(
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
        sys.stdout.write(
            f"{pair}: pool={report['candidate_events_at_0.55']:,}, {report['status']}\n"
        )
        sys.stdout.flush()
    manifest = {
        "kind": "round5_q2_frozen_policies",
        "protected_samples_opened": False,
        "year_2022_opened": False,
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

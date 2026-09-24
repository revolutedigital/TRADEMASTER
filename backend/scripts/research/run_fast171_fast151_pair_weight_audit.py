"""Fast171: pair-risk weight audit for the Fast151 operating policy.

Fast151's remaining blocker is not just signal quality; swap and drawdown
margin are tight. Fast171 asks whether the same frozen entry/exit family can be
made more operable by scaling risk per pair, while keeping:

* the same USD-quoted four-pair universe;
* the same D1 Donchian fade ch100 hold20 SHORT-only entry;
* the same stop2 fixed 1R management;
* the same max-per-pair exposure overlay;
* the same causal drawdown circuit breaker.

Unlike a weak post-processing check, Fast171 starts from the Fast149
max_per_pair_1 overlay trades and reapplies the drawdown circuit breaker after
pair weights scale closed P/L. The weight selection is based only on the
pre-2019 sample; current-period results are audit evidence, not a selector.

Historical research only. No orders, no ledger, no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    TARGET_PAIRS,
    TARGET_POLICY_ID,
    summarize_sample,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import lower_confidence_bound  # noqa: E402
from scripts.research.run_fast150_usd_quoted_basket_circuit_breaker_audit import (  # noqa: E402
    CircuitBreakerPolicy,
    apply_drawdown_circuit_breaker,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)
from scripts.research.run_fast153_fast151_robustness_audit import (  # noqa: E402
    TIMESTAMP_COLUMNS,
)
from scripts.research.run_fast159_fast151_exact_swap_rollover_audit import add_rollover_days  # noqa: E402

DEFAULT_PRE_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast149"
    / "usd-quoted-basket-position-cap-audit"
    / "selected-overlay-pre-trades.csv"
)
DEFAULT_CURRENT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast149"
    / "usd-quoted-basket-position-cap-audit"
    / "selected-overlay-current-trades.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast171" / "fast151-pair-weight-audit"

SELECTED_CIRCUIT_BREAKER = CircuitBreakerPolicy(
    "dd800_cooldown180d",
    closed_equity_drawdown_threshold_pips=800.0,
    cooldown_days=180,
)
PAIR_WEIGHT_VALUES = (0.25, 0.50, 0.75, 1.00)
BASELINE_WEIGHTS = {pair: 1.0 for pair in TARGET_PAIRS}
MIN_PRE_TRADES = 50
MIN_CURRENT_TRADES = 50
MIN_POSITIVE_YEAR_SHARE = 0.65
MAX_PRE_DRAWDOWN_PIPS = 1000.0
MAX_CURRENT_DRAWDOWN_PIPS = 1000.0
MAX_CONCURRENT_TRADES = 4
MIN_MATERIAL_CURRENT_SWAP_UPLIFT = 0.10


@dataclass(frozen=True)
class PairWeightPolicy:
    identifier: str
    weights: dict[str, float]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-input-path", type=Path, default=DEFAULT_PRE_INPUT_PATH)
    parser.add_argument("--current-input-path", type=Path, default=DEFAULT_CURRENT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(
        pre_input_path=arguments.pre_input_path,
        current_input_path=arguments.current_input_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    pre_overlay = add_rollover_days(load_overlay_trades(pre_input_path))
    current_overlay = add_rollover_days(load_overlay_trades(current_input_path))
    evaluations = [
        evaluate_weight_policy(pre_overlay, current_overlay, policy)
        for policy in pair_weight_grid()
    ]
    baseline = next(evaluation for evaluation in evaluations if evaluation["weight_policy_id"] == "equal_1_00")
    pre_selected = select_pre_weight_policy(evaluations)
    strict_survivors = [evaluation for evaluation in evaluations if evaluation_passes_both(evaluation)]
    strict_selected = select_strict_weight_policy(strict_survivors)
    status, status_reasons = fast171_status(
        pre_selected=pre_selected,
        baseline=baseline,
        strict_survivor_count=len(strict_survivors),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "pair-weight-metrics.csv"
    summary_path = output_root / "pair-weight-summary.json"
    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "baseline_policy": baseline,
        "pre_selected_policy": pre_selected,
        "strict_selected_policy": strict_selected,
        "strict_survivor_count": len(strict_survivors),
        "evaluation_count": len(evaluations),
        "pre_gate_pass_count": sum(1 for evaluation in evaluations if evaluation["pre"]["strict_gate_passed"]),
        "current_gate_pass_count": sum(1 for evaluation in evaluations if evaluation["current"]["strict_gate_passed"]),
        "top_pre_swap_candidates": [compact_policy(evaluation) for evaluation in top_pre_swap_candidates(evaluations, 10)],
        "top_strict_swap_candidates": [compact_policy(evaluation) for evaluation in top_strict_swap_candidates(strict_survivors, 10)],
        "policy_constants": {
            "target_pairs": list(TARGET_PAIRS),
            "weight_values": list(PAIR_WEIGHT_VALUES),
            "circuit_breaker": {
                "breaker_id": SELECTED_CIRCUIT_BREAKER.identifier,
                "closed_equity_drawdown_threshold_pips": SELECTED_CIRCUIT_BREAKER.closed_equity_drawdown_threshold_pips,
                "cooldown_days": SELECTED_CIRCUIT_BREAKER.cooldown_days,
            },
        },
        "selection_rule": {
            "selector": "pre_sample_only",
            "primary": "pre_uniform_swap_break_even_lcb_zero_pips_per_rollover_day",
            "secondary": "pre_lower_stress_pips",
            "current_is_audit_only": True,
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast171_is_historical_pair_weight_audit_not_strategy_approval",
            "pair_weights_selected_on_pre_sample_only_and_still_require_preregistration",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast171_fast151_pair_weight_audit",
        "status": status,
        "status_reasons": status_reasons,
        "target_policy_id": TARGET_POLICY_ID,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "pair_weight_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def load_overlay_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values(["entry_timestamp", "pair", "candidate_id"], ignore_index=True)


def pair_weight_grid() -> tuple[PairWeightPolicy, ...]:
    policies: list[PairWeightPolicy] = []
    for values in itertools.product(PAIR_WEIGHT_VALUES, repeat=len(TARGET_PAIRS)):
        weights = dict(zip(TARGET_PAIRS, values, strict=True))
        policies.append(PairWeightPolicy(identifier=weight_policy_id(weights), weights=weights))
    return tuple(sorted(policies, key=lambda policy: policy.identifier))


def weight_policy_id(weights: dict[str, float]) -> str:
    if weights == BASELINE_WEIGHTS:
        return "equal_1_00"
    parts = [f"{pair.lower()}_{str(weights[pair]).replace('.', '_')}" for pair in TARGET_PAIRS]
    return "pair_weight:" + ":".join(parts)


def evaluate_weight_policy(
    pre_overlay: pd.DataFrame,
    current_overlay: pd.DataFrame,
    policy: PairWeightPolicy,
) -> dict[str, Any]:
    return {
        "weight_policy_id": policy.identifier,
        "weights": policy.weights,
        "pre": evaluate_weighted_sample(
            pre_overlay,
            policy,
            min_trades=MIN_PRE_TRADES,
            max_drawdown_pips=MAX_PRE_DRAWDOWN_PIPS,
        ),
        "current": evaluate_weighted_sample(
            current_overlay,
            policy,
            min_trades=MIN_CURRENT_TRADES,
            max_drawdown_pips=MAX_CURRENT_DRAWDOWN_PIPS,
        ),
    }


def evaluate_weighted_sample(
    overlay_trades: pd.DataFrame,
    policy: PairWeightPolicy,
    *,
    min_trades: int,
    max_drawdown_pips: float,
) -> dict[str, Any]:
    weighted_overlay = apply_pair_weights(overlay_trades, policy.weights)
    accepted, skipped, triggers = apply_drawdown_circuit_breaker(weighted_overlay, SELECTED_CIRCUIT_BREAKER)
    summary = summarize_sample(accepted)
    failures = strict_gate_failures(summary, min_trades=min_trades, max_drawdown_pips=max_drawdown_pips)
    return {
        "summary": summary,
        "breaker_skipped_count": int(len(skipped)),
        "breaker_trigger_count": int(len(triggers)),
        "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": estimate_weighted_break_even(accepted),
        "weighted_rollover_day_diagnostics": weighted_rollover_day_diagnostics(accepted),
        "strict_gate_passed": not failures,
        "strict_gate_failures": failures,
    }


def apply_pair_weights(trades: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    missing_pairs = sorted(set(trades["pair"].unique()) - set(weights))
    if missing_pairs:
        raise ValueError(f"weights missing pairs: {missing_pairs}")
    weighted = trades.copy()
    weighted["pair_weight"] = weighted["pair"].map(weights).astype("float64")
    weighted["unweighted_net_base_pips"] = weighted["net_base_pips"].astype("float64")
    weighted["unweighted_net_stress_pips"] = weighted["net_stress_pips"].astype("float64")
    weighted["unweighted_raw_base_pips"] = weighted["raw_base_pips"].astype("float64")
    weighted["unweighted_raw_stress_pips"] = weighted["raw_stress_pips"].astype("float64")
    for column in ("net_base_pips", "net_stress_pips", "raw_base_pips", "raw_stress_pips"):
        weighted[column] = weighted[column].astype("float64") * weighted["pair_weight"]
    weighted["weighted_swap_rollover_days"] = (
        weighted["swap_rollover_days"].astype("float64") * weighted["pair_weight"]
    )
    return weighted


def apply_weighted_uniform_adverse_swap_cost(
    trades: pd.DataFrame,
    adverse_short_pips_per_rollover_day: float,
) -> pd.DataFrame:
    adjusted = trades.copy()
    adjusted["swap_cost_pips"] = (
        adjusted["weighted_swap_rollover_days"].astype("float64") * adverse_short_pips_per_rollover_day
    )
    adjusted["swap_pips"] = -adjusted["swap_cost_pips"]
    adjusted["net_stress_pips"] = adjusted["net_stress_pips"].astype("float64") + adjusted["swap_pips"]
    return adjusted


def estimate_weighted_break_even(trades: pd.DataFrame) -> float:
    if trades.empty or trades["weighted_swap_rollover_days"].sum() <= 0:
        return 0.0
    if weighted_lcb_after_swap(trades, adverse_short_pips_per_rollover_day=0.0) <= 0:
        return 0.0
    low = 0.0
    high = 10.0
    while weighted_lcb_after_swap(trades, adverse_short_pips_per_rollover_day=high) > 0:
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        if weighted_lcb_after_swap(trades, adverse_short_pips_per_rollover_day=mid) > 0:
            low = mid
        else:
            high = mid
    return high


def weighted_lcb_after_swap(trades: pd.DataFrame, *, adverse_short_pips_per_rollover_day: float) -> float:
    adjusted = trades["net_stress_pips"].astype("float64") - (
        trades["weighted_swap_rollover_days"].astype("float64") * adverse_short_pips_per_rollover_day
    )
    value = lower_confidence_bound(adjusted, 0.05)
    return value if pd.notna(value) else -1_000_000.0


def weighted_rollover_day_diagnostics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "weighted_total_rollover_days": 0.0,
            "unweighted_total_rollover_days": 0,
            "pair_rollover_diagnostics": [],
        }
    grouped = (
        trades.groupby("pair", sort=True)
        .agg(
            trade_count=("pair", "size"),
            unweighted_total_rollover_days=("swap_rollover_days", "sum"),
            weighted_total_rollover_days=("weighted_swap_rollover_days", "sum"),
            mean_pair_weight=("pair_weight", "mean"),
        )
        .reset_index()
    )
    return {
        "trade_count": int(len(trades)),
        "weighted_total_rollover_days": float(trades["weighted_swap_rollover_days"].sum()),
        "unweighted_total_rollover_days": int(trades["swap_rollover_days"].sum()),
        "pair_rollover_diagnostics": [
            {
                "pair": str(row["pair"]),
                "trade_count": int(row["trade_count"]),
                "unweighted_total_rollover_days": int(row["unweighted_total_rollover_days"]),
                "weighted_total_rollover_days": float(row["weighted_total_rollover_days"]),
                "mean_pair_weight": float(row["mean_pair_weight"]),
            }
            for row in grouped.to_dict("records")
        ],
    }


def strict_gate_failures(
    summary: dict[str, Any],
    *,
    min_trades: int,
    max_drawdown_pips: float,
) -> list[str]:
    failures: list[str] = []
    if summary["trade_count"] < min_trades:
        failures.append("trade_count_below_minimum")
    if summary["mean_stress_pips"] is None or summary["mean_stress_pips"] <= 0:
        failures.append("mean_non_positive")
    if summary["lower_stress_pips"] is None or summary["lower_stress_pips"] <= 0:
        failures.append("lcb_non_positive")
    if summary["max_drawdown_pips"] is None or abs(summary["max_drawdown_pips"]) > max_drawdown_pips:
        failures.append("drawdown_exceeds_limit")
    if summary["max_concurrent_trades"] > MAX_CONCURRENT_TRADES:
        failures.append("concurrency_exceeds_limit")
    if summary["two_consecutive_negative_years"]:
        failures.append("two_consecutive_negative_years")
    if summary["positive_year_share"] is None or summary["positive_year_share"] < MIN_POSITIVE_YEAR_SHARE:
        failures.append("positive_year_share_below_minimum")
    return failures


def evaluation_passes_both(evaluation: dict[str, Any]) -> bool:
    return bool(evaluation["pre"]["strict_gate_passed"] and evaluation["current"]["strict_gate_passed"])


def select_pre_weight_policy(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [evaluation for evaluation in evaluations if evaluation["pre"]["strict_gate_passed"]]
    if not passed:
        return None
    return sorted(passed, key=pre_selection_key, reverse=True)[0]


def select_strict_weight_policy(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not evaluations:
        return None
    return sorted(evaluations, key=strict_selection_key, reverse=True)[0]


def top_pre_swap_candidates(evaluations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    passed = [evaluation for evaluation in evaluations if evaluation["pre"]["strict_gate_passed"]]
    return sorted(passed, key=pre_selection_key, reverse=True)[:limit]


def top_strict_swap_candidates(evaluations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(evaluations, key=strict_selection_key, reverse=True)[:limit]


def pre_selection_key(evaluation: dict[str, Any]) -> tuple[float, float, float, str]:
    pre = evaluation["pre"]["summary"]
    return (
        evaluation["pre"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        pre["lower_stress_pips"] if pre["lower_stress_pips"] is not None else -1_000_000.0,
        -abs(pre["max_drawdown_pips"]) if pre["max_drawdown_pips"] is not None else -1_000_000.0,
        evaluation["weight_policy_id"],
    )


def strict_selection_key(evaluation: dict[str, Any]) -> tuple[float, float, float, str]:
    pre = evaluation["pre"]["summary"]
    current = evaluation["current"]["summary"]
    pre_lcb = pre["lower_stress_pips"] if pre["lower_stress_pips"] is not None else -1_000_000.0
    current_lcb = current["lower_stress_pips"] if current["lower_stress_pips"] is not None else -1_000_000.0
    return (
        min(
            evaluation["pre"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
            evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        ),
        min(pre_lcb, current_lcb),
        -abs(current["max_drawdown_pips"]) if current["max_drawdown_pips"] is not None else -1_000_000.0,
        evaluation["weight_policy_id"],
    )


def fast171_status(
    *,
    pre_selected: dict[str, Any] | None,
    baseline: dict[str, Any],
    strict_survivor_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "pair_weight_audit"]
    if pre_selected is None:
        return (
            "FAST171_NO_PAIR_WEIGHT_POLICY_PASSES_PRE_GATE",
            reasons + ["fresh_preregistered_prospective_observation_still_required"],
        )
    flags = [
        f"strict_survivor_count:{strict_survivor_count}",
        f"pre_selected_weight_policy:{pre_selected['weight_policy_id']}",
        "pre_sample_selection_only",
        "fresh_preregistered_prospective_observation_still_required",
    ]
    if pre_selected["weight_policy_id"] == baseline["weight_policy_id"]:
        return (
            "FAST171_EQUAL_WEIGHTS_REMAIN_PRE_SELECTED",
            reasons + flags + ["baseline_equal_weights_remain_selected"],
        )
    if not pre_selected["current"]["strict_gate_passed"]:
        return (
            "FAST171_PRE_SELECTED_PAIR_WEIGHT_FAILS_CURRENT_AUDIT",
            reasons + flags + ["do_not_promote_pair_weight_policy"],
        )
    current_swap_uplift = (
        pre_selected["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"]
        - baseline["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"]
    )
    flags.append(f"current_swap_uplift:{current_swap_uplift:.6f}")
    if current_swap_uplift >= MIN_MATERIAL_CURRENT_SWAP_UPLIFT:
        return (
            "FAST171_PAIR_WEIGHT_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons + flags + ["pre_selected_weight_policy_passes_current_audit"],
        )
    return (
        "FAST171_PAIR_WEIGHT_PASSES_CURRENT_BUT_NO_MATERIAL_SWAP_UPLIFT",
        reasons + flags + ["do_not_replace_equal_weight_fast151_policy"],
    )


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row: dict[str, Any] = {
            "weight_policy_id": evaluation["weight_policy_id"],
            "weights": json.dumps(evaluation["weights"], sort_keys=True),
            "pre_strict_gate_passed": evaluation["pre"]["strict_gate_passed"],
            "current_strict_gate_passed": evaluation["current"]["strict_gate_passed"],
            "both_strict_gate_passed": evaluation_passes_both(evaluation),
        }
        for sample_name in ("pre", "current"):
            sample = evaluation[sample_name]
            summary = sample["summary"]
            row[f"{sample_name}_breaker_skipped_count"] = sample["breaker_skipped_count"]
            row[f"{sample_name}_breaker_trigger_count"] = sample["breaker_trigger_count"]
            row[f"{sample_name}_uniform_swap_break_even"] = sample[
                "uniform_swap_break_even_lcb_zero_pips_per_rollover_day"
            ]
            row[f"{sample_name}_strict_gate_failures"] = "|".join(sample["strict_gate_failures"])
            row[f"{sample_name}_weighted_total_rollover_days"] = sample["weighted_rollover_day_diagnostics"][
                "weighted_total_rollover_days"
            ]
            for key, value in summary.items():
                if key in {"yearly", "pair_summary", "exit_reason_summary"}:
                    continue
                row[f"{sample_name}_{key}"] = value
        rows.append(row)
    return rows


def compact_policy(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "weight_policy_id": evaluation["weight_policy_id"],
        "weights": evaluation["weights"],
        "pre": compact_sample(evaluation["pre"]),
        "current": compact_sample(evaluation["current"]),
    }


def compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    summary = sample["summary"]
    return {
        "trade_count": summary["trade_count"],
        "mean_stress_pips": summary["mean_stress_pips"],
        "lower_stress_pips": summary["lower_stress_pips"],
        "max_drawdown_pips": summary["max_drawdown_pips"],
        "two_consecutive_negative_years": summary["two_consecutive_negative_years"],
        "positive_year_share": summary["positive_year_share"],
        "uniform_swap_break_even": sample["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        "weighted_total_rollover_days": sample["weighted_rollover_day_diagnostics"]["weighted_total_rollover_days"],
        "strict_gate_failures": sample["strict_gate_failures"],
    }


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "operational_policy_id": report["operational_policy_id"],
        "evaluation_count": report["summary"]["evaluation_count"],
        "pre_gate_pass_count": report["summary"]["pre_gate_pass_count"],
        "current_gate_pass_count": report["summary"]["current_gate_pass_count"],
        "strict_survivor_count": report["summary"]["strict_survivor_count"],
        "baseline": compact_policy(report["summary"]["baseline_policy"]),
        "pre_selected": compact_policy(report["summary"]["pre_selected_policy"]),
        "strict_selected": compact_policy(report["summary"]["strict_selected_policy"]),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

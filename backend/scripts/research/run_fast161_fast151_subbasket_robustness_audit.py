"""Fast161: sub-basket robustness audit for the Fast151 operating policy.

Fast160 showed pair-level swap sensitivity. Fast161 tests the natural follow-up:
does keeping the same entry, position overlay and circuit breaker but changing
the USD-quoted pair universe improve robustness?

Important guardrail: this is a post-hoc basket-composition audit. The frozen
Fast151 operating policy remains the four-pair basket. A smaller basket can
only become a new candidate after explicit preregistration and prospective
observation. This runner therefore reports both:

* strict gate: keeps the historical sample-size gates from Fast150;
* exploratory ranking: shows promising smaller baskets that fail strict gate,
  usually because the pre sample has too few trades.

Historical research only. No orders, no ledger, no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    CURRENT_SAMPLE_LABEL,
    DEFAULT_CURRENT_INPUT_PATH,
    DEFAULT_PRE_INPUT_PATH,
    ENTRY_POLICY_ID,
    GROUP_ID,
    MANAGEMENT,
    PRE_SAMPLE_LABEL,
    RISK_POLICY_ID,
    SIDE_FILTER,
    TARGET_PAIRS,
    TARGET_POLICY_ID,
    filtered_sample,
    summarize_sample,
)
from scripts.research.run_fast149_usd_quoted_basket_position_cap_audit import (  # noqa: E402
    PositionOverlay,
    apply_position_overlay,
)
from scripts.research.run_fast150_usd_quoted_basket_circuit_breaker_audit import (  # noqa: E402
    CircuitBreakerPolicy,
    apply_drawdown_circuit_breaker,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)
from scripts.research.run_fast159_fast151_exact_swap_rollover_audit import (  # noqa: E402
    add_rollover_days,
    estimate_uniform_break_even_pips_per_rollover_day,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast161" / "fast151-subbasket-robustness-audit"
)
SELECTED_POSITION_OVERLAY = PositionOverlay(
    "max_per_pair_1",
    max_total_open=None,
    max_open_per_pair=1,
)
SELECTED_CIRCUIT_BREAKER = CircuitBreakerPolicy(
    "dd800_cooldown180d",
    closed_equity_drawdown_threshold_pips=800.0,
    cooldown_days=180,
)
MIN_PRE_TRADES = 50
MIN_CURRENT_TRADES = 50
MAX_PRE_DRAWDOWN_PIPS = 1000.0
MAX_CURRENT_DRAWDOWN_PIPS = 1000.0
MIN_POSITIVE_YEAR_SHARE = 0.65
MAX_CONCURRENT_TRADES = 4


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
    pre_source = pd.read_parquet(pre_input_path)
    current_source = pd.read_parquet(current_input_path)
    pre_trades = filtered_sample(pre_source, sample_column="sample", sample_value=PRE_SAMPLE_LABEL)
    current_trades = filtered_sample(current_source, sample_column=None, sample_value=None)
    basket_evaluations = [
        evaluate_subbasket(pre_trades, current_trades, pairs)
        for pairs in pair_subsets(TARGET_PAIRS)
    ]
    strict_selected = select_strict_basket(basket_evaluations)
    exploratory_selected = select_exploratory_basket(basket_evaluations)
    status, status_reasons = fast161_status(
        strict_selected=strict_selected,
        exploratory_selected=exploratory_selected,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "subbasket-robustness-metrics.csv"
    summary_path = output_root / "subbasket-robustness-summary.json"
    pd.DataFrame(flat_metric_rows(basket_evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "strict_selected_basket": strict_selected,
        "exploratory_selected_basket": exploratory_selected,
        "basket_evaluations": basket_evaluations,
        "policy_constants": {
            "target_pairs": list(TARGET_PAIRS),
            "entry_policy_id": ENTRY_POLICY_ID,
            "risk_policy_id": RISK_POLICY_ID,
            "side_filter": SIDE_FILTER,
            "management": MANAGEMENT,
            "position_overlay": {
                "overlay_id": SELECTED_POSITION_OVERLAY.identifier,
                "max_total_open": SELECTED_POSITION_OVERLAY.max_total_open,
                "max_open_per_pair": SELECTED_POSITION_OVERLAY.max_open_per_pair,
            },
            "circuit_breaker": {
                "breaker_id": SELECTED_CIRCUIT_BREAKER.identifier,
                "closed_equity_drawdown_threshold_pips": SELECTED_CIRCUIT_BREAKER.closed_equity_drawdown_threshold_pips,
                "cooldown_days": SELECTED_CIRCUIT_BREAKER.cooldown_days,
            },
        },
        "strict_gates": {
            "min_pre_trades": MIN_PRE_TRADES,
            "min_current_trades": MIN_CURRENT_TRADES,
            "max_pre_drawdown_pips": MAX_PRE_DRAWDOWN_PIPS,
            "max_current_drawdown_pips": MAX_CURRENT_DRAWDOWN_PIPS,
            "min_positive_year_share": MIN_POSITIVE_YEAR_SHARE,
            "max_concurrent_trades": MAX_CONCURRENT_TRADES,
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast161_is_historical_post_hoc_subbasket_audit_not_strategy_approval",
            "smaller_basket_cannot_replace_fast151_without_new_preregistration",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast161_fast151_subbasket_robustness_audit",
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
            "subbasket_robustness_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def pair_subsets(pairs: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(1, len(pairs) + 1)
        for subset in itertools.combinations(pairs, size)
    ]


def evaluate_subbasket(
    pre_trades: pd.DataFrame,
    current_trades: pd.DataFrame,
    pairs: tuple[str, ...],
) -> dict[str, Any]:
    pre_result = evaluate_sample(pre_trades, pairs, min_trades=MIN_PRE_TRADES, max_drawdown_pips=MAX_PRE_DRAWDOWN_PIPS)
    current_result = evaluate_sample(
        current_trades,
        pairs,
        min_trades=MIN_CURRENT_TRADES,
        max_drawdown_pips=MAX_CURRENT_DRAWDOWN_PIPS,
    )
    return {
        "basket_id": "+".join(pairs),
        "pairs": list(pairs),
        "pair_count": len(pairs),
        "pre": pre_result,
        "current": current_result,
        "strict_pre_gate_passed": pre_result["strict_gate_passed"],
        "strict_current_gate_passed": current_result["strict_gate_passed"],
        "strict_both_gate_passed": pre_result["strict_gate_passed"] and current_result["strict_gate_passed"],
    }


def evaluate_sample(
    trades: pd.DataFrame,
    pairs: tuple[str, ...],
    *,
    min_trades: int,
    max_drawdown_pips: float,
) -> dict[str, Any]:
    subbasket = trades.loc[trades["pair"].isin(pairs)].copy()
    overlay_trades, overlay_skipped = apply_position_overlay(subbasket, SELECTED_POSITION_OVERLAY)
    accepted, breaker_skipped, breaker_triggers = apply_drawdown_circuit_breaker(
        overlay_trades,
        SELECTED_CIRCUIT_BREAKER,
    )
    summary = summarize_sample(accepted)
    swap_break_even = (
        estimate_uniform_break_even_pips_per_rollover_day(add_rollover_days(accepted))
        if not accepted.empty and summary["lower_stress_pips"] is not None and summary["lower_stress_pips"] > 0
        else 0.0
    )
    gate_failures = strict_gate_failures(summary, min_trades=min_trades, max_drawdown_pips=max_drawdown_pips)
    return {
        "summary": summary,
        "position_skipped_count": int(len(overlay_skipped)),
        "breaker_skipped_count": int(len(breaker_skipped)),
        "breaker_trigger_count": int(len(breaker_triggers)),
        "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": swap_break_even,
        "strict_gate_passed": not gate_failures,
        "strict_gate_failures": gate_failures,
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


def select_strict_basket(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [evaluation for evaluation in evaluations if evaluation["strict_both_gate_passed"]]
    if not passed:
        return None
    return sorted(passed, key=strict_selection_key, reverse=True)[0]


def strict_selection_key(evaluation: dict[str, Any]) -> tuple[float, float, float, float, int]:
    pre = evaluation["pre"]["summary"]
    current = evaluation["current"]["summary"]
    return (
        min(pre["lower_stress_pips"], current["lower_stress_pips"]),
        evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        -abs(current["max_drawdown_pips"]),
        -abs(pre["max_drawdown_pips"]),
        -evaluation["pair_count"],
    )


def select_exploratory_basket(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [
        evaluation
        for evaluation in evaluations
        if exploratory_viable(evaluation["pre"]["summary"]) and exploratory_viable(evaluation["current"]["summary"])
    ]
    if not viable:
        return None
    return sorted(viable, key=exploratory_selection_key, reverse=True)[0]


def exploratory_viable(summary: dict[str, Any]) -> bool:
    return bool(
        summary["trade_count"] > 0
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and not summary["two_consecutive_negative_years"]
    )


def exploratory_selection_key(evaluation: dict[str, Any]) -> tuple[float, float, float, int]:
    pre = evaluation["pre"]["summary"]
    current = evaluation["current"]["summary"]
    return (
        min(pre["lower_stress_pips"], current["lower_stress_pips"]),
        evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        -abs(current["max_drawdown_pips"]),
        evaluation["pair_count"],
    )


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row: dict[str, Any] = {
            "basket_id": evaluation["basket_id"],
            "pairs": ",".join(evaluation["pairs"]),
            "pair_count": evaluation["pair_count"],
            "strict_pre_gate_passed": evaluation["strict_pre_gate_passed"],
            "strict_current_gate_passed": evaluation["strict_current_gate_passed"],
            "strict_both_gate_passed": evaluation["strict_both_gate_passed"],
        }
        for sample_name in ("pre", "current"):
            sample = evaluation[sample_name]
            summary = sample["summary"]
            row[f"{sample_name}_position_skipped_count"] = sample["position_skipped_count"]
            row[f"{sample_name}_breaker_skipped_count"] = sample["breaker_skipped_count"]
            row[f"{sample_name}_breaker_trigger_count"] = sample["breaker_trigger_count"]
            row[f"{sample_name}_uniform_swap_break_even"] = sample[
                "uniform_swap_break_even_lcb_zero_pips_per_rollover_day"
            ]
            row[f"{sample_name}_strict_gate_failures"] = "|".join(sample["strict_gate_failures"])
            for key, value in summary.items():
                if key in {"yearly", "pair_summary", "exit_reason_summary"}:
                    continue
                row[f"{sample_name}_{key}"] = value
        rows.append(row)
    return rows


def fast161_status(
    *,
    strict_selected: dict[str, Any] | None,
    exploratory_selected: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "post_hoc_subbasket_audit"]
    if strict_selected is None:
        return (
            "FAST161_NO_SUBBASKET_PASSES_STRICT_GATE",
            reasons + ["fresh_preregistered_prospective_observation_still_required"],
        )
    flags = [
        f"strict_selected_basket:{strict_selected['basket_id']}",
        "strict_selection_is_historical_not_approval",
        "fresh_preregistered_prospective_observation_still_required",
    ]
    if exploratory_selected is not None and exploratory_selected["basket_id"] != strict_selected["basket_id"]:
        flags.append(f"exploratory_best_basket:{exploratory_selected['basket_id']}")
        flags.append("exploratory_best_requires_new_preregistration_due_post_hoc_pair_selection")
        return (
            "FAST161_STRICT_GATE_KEEP_CURRENT_BASKET_EXPLORATORY_SUBBASKET_IDENTIFIED",
            reasons + flags,
        )
    return (
        "FAST161_STRICT_GATE_KEEP_CURRENT_BASKET_NO_SUBBASKET_REPLACEMENT",
        reasons + flags,
    )


def basket_summary(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "basket_id": evaluation["basket_id"],
        "pairs": evaluation["pairs"],
        "pre": {
            "trade_count": evaluation["pre"]["summary"]["trade_count"],
            "mean_stress_pips": evaluation["pre"]["summary"]["mean_stress_pips"],
            "lower_stress_pips": evaluation["pre"]["summary"]["lower_stress_pips"],
            "max_drawdown_pips": evaluation["pre"]["summary"]["max_drawdown_pips"],
            "strict_gate_failures": evaluation["pre"]["strict_gate_failures"],
            "uniform_swap_break_even": evaluation["pre"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        },
        "current": {
            "trade_count": evaluation["current"]["summary"]["trade_count"],
            "mean_stress_pips": evaluation["current"]["summary"]["mean_stress_pips"],
            "lower_stress_pips": evaluation["current"]["summary"]["lower_stress_pips"],
            "max_drawdown_pips": evaluation["current"]["summary"]["max_drawdown_pips"],
            "strict_gate_failures": evaluation["current"]["strict_gate_failures"],
            "uniform_swap_break_even": evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        },
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
        "strict_selected": basket_summary(report["summary"]["strict_selected_basket"]),
        "exploratory_selected": basket_summary(report["summary"]["exploratory_selected_basket"]),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

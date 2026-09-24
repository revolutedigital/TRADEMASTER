"""Fast169: expanded USD-quoted policy audit around Fast151.

Fast168 proved that simply compressing the Fast151 hold from 20 days to 5/10
days does not pass the pre-period gate. Fast169 expands that question across the
same safe family:

* USD-quoted major subsets from AUDUSD/EURUSD/GBPUSD/NZDUSD;
* daily Donchian fade channels 20/55/100;
* holds 5/10/20;
* SHORT-only;
* stop2 fixed 1R;
* max one open position per pair;
* the Fast151 drawdown circuit breaker.

The audit asks whether another basket/channel/hold variant in this family beats
the current Fast151 operating candidate without using current-period lookahead.

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
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)
from scripts.research.run_fast168_fast151_hold_compression_audit import (  # noqa: E402
    ENTRY_POLICIES as FAST168_ENTRY_POLICIES,
    evaluate_sample,
    strict_selection_key,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast169" / "fast151-usd-policy-expansion-audit"
)
BENCHMARK_BASKET_ID = "+".join(TARGET_PAIRS)
BENCHMARK_POLICY_ID = ENTRY_POLICY_ID


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
    pre_family = policy_family_sample(pre_source, sample_value=PRE_SAMPLE_LABEL)
    current_family = policy_family_sample(current_source, sample_value=None)
    entry_policy_ids = expanded_entry_policies(pre_family, current_family)
    evaluations = [
        evaluate_basket_policy(pre_family, current_family, basket, entry_policy_id)
        for basket in basket_grid(TARGET_PAIRS)
        for entry_policy_id in entry_policy_ids
    ]
    strict_survivors = [evaluation for evaluation in evaluations if evaluation_passes_both(evaluation)]
    benchmark = next(
        evaluation
        for evaluation in evaluations
        if evaluation["basket_id"] == BENCHMARK_BASKET_ID
        and evaluation["entry_policy_id"] == BENCHMARK_POLICY_ID
    )
    strict_selected = select_strict_survivor(strict_survivors)
    status, status_reasons = fast169_status(
        strict_survivors=strict_survivors,
        strict_selected=strict_selected,
        benchmark=benchmark,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "usd-policy-expansion-metrics.csv"
    summary_path = output_root / "usd-policy-expansion-summary.json"
    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "benchmark_policy": benchmark,
        "strict_selected_policy": strict_selected,
        "strict_survivors": strict_survivors,
        "strict_survivor_count": len(strict_survivors),
        "evaluation_count": len(evaluations),
        "pre_gate_pass_count": sum(1 for evaluation in evaluations if evaluation["pre"]["strict_gate_passed"]),
        "current_gate_pass_count": sum(1 for evaluation in evaluations if evaluation["current"]["strict_gate_passed"]),
        "top_current_only_swap_candidates": top_current_only_swap_candidates(evaluations, limit=10),
        "policy_constants": {
            "target_pairs": list(TARGET_PAIRS),
            "group_id": GROUP_ID,
            "risk_policy_id": RISK_POLICY_ID,
            "side_filter": SIDE_FILTER,
            "management": MANAGEMENT,
            "entry_policy_ids": list(entry_policy_ids),
            "basket_count": len(list(basket_grid(TARGET_PAIRS))),
        },
        "interpretation": {
            "question": "whether another USD-quoted basket/channel/hold variant passes both pre and current gates",
            "selection_rule": "strict survivor must pass pre-2019 and 2019-2026 gates; current-only winners are diagnostics",
            "family_scope": "USD-quoted majors, short-only daily Donchian fade, stop2 fixed 1R",
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast169_is_historical_policy_expansion_audit_not_strategy_approval",
            "expanded_family_uniqueness_is_not_execution_authorization",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast169_fast151_usd_policy_expansion_audit",
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
            "usd_policy_expansion_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def policy_family_sample(source: pd.DataFrame, *, sample_value: str | None) -> pd.DataFrame:
    sample = source.copy()
    if sample_value is not None and "sample" in sample.columns:
        sample = sample.loc[sample["sample"].eq(sample_value)].copy()
    for column in ("signal_timestamp", "entry_timestamp", "exit_base_timestamp", "exit_stress_timestamp"):
        sample[column] = pd.to_datetime(sample[column], utc=True)
    return sample.loc[
        sample["pair"].isin(TARGET_PAIRS)
        & sample["entry_policy_id"].astype(str).str.startswith("d1_donchian_fade_")
        & sample["risk_policy_id"].eq(RISK_POLICY_ID)
        & sample["side"].eq(SIDE_FILTER)
        & sample["management"].eq(MANAGEMENT)
    ].sort_values(["entry_timestamp", "pair", "entry_policy_id"], ignore_index=True)


def expanded_entry_policies(pre_family: pd.DataFrame, current_family: pd.DataFrame) -> tuple[str, ...]:
    pre_policies = set(pre_family["entry_policy_id"].astype(str).unique())
    current_policies = set(current_family["entry_policy_id"].astype(str).unique())
    policy_ids = tuple(sorted(pre_policies.intersection(current_policies), key=entry_policy_sort_key))
    if not set(FAST168_ENTRY_POLICIES).issubset(policy_ids):
        raise ValueError("expanded policy universe is missing Fast168 entry policies")
    return policy_ids


def entry_policy_sort_key(entry_policy_id: str) -> tuple[int, int, str]:
    channel = int(entry_policy_id.split("_ch", maxsplit=1)[1].split("_", maxsplit=1)[0])
    hold = int(entry_policy_id.rsplit("hold", maxsplit=1)[1])
    return channel, hold, entry_policy_id


def basket_grid(pairs: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    baskets: list[tuple[str, ...]] = []
    for basket_size in range(1, len(pairs) + 1):
        baskets.extend(itertools.combinations(pairs, basket_size))
    return tuple(baskets)


def evaluate_basket_policy(
    pre_family: pd.DataFrame,
    current_family: pd.DataFrame,
    basket: tuple[str, ...],
    entry_policy_id: str,
) -> dict[str, Any]:
    pre_policy = sample_for_basket_policy(pre_family, basket=basket, entry_policy_id=entry_policy_id)
    current_policy = sample_for_basket_policy(current_family, basket=basket, entry_policy_id=entry_policy_id)
    return {
        "basket_id": "+".join(basket),
        "pairs": list(basket),
        "pair_count": len(basket),
        "entry_policy_id": entry_policy_id,
        "hold_days": int(pre_policy["hold_days"].iloc[0]),
        "channel_days": int(pre_policy["channel_days"].iloc[0]),
        "pre": evaluate_sample(pre_policy, min_trades=50, max_drawdown_pips=1000.0),
        "current": evaluate_sample(current_policy, min_trades=50, max_drawdown_pips=1000.0),
    }


def sample_for_basket_policy(
    family: pd.DataFrame,
    *,
    basket: tuple[str, ...],
    entry_policy_id: str,
) -> pd.DataFrame:
    sample = family.loc[
        family["pair"].isin(basket)
        & family["entry_policy_id"].eq(entry_policy_id)
    ].copy()
    if sample.empty:
        raise ValueError(f"empty sample for basket={basket!r}, entry_policy_id={entry_policy_id!r}")
    return sample.sort_values(["entry_timestamp", "pair"], ignore_index=True)


def evaluation_passes_both(evaluation: dict[str, Any]) -> bool:
    return bool(evaluation["pre"]["strict_gate_passed"] and evaluation["current"]["strict_gate_passed"])


def select_strict_survivor(strict_survivors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not strict_survivors:
        return None
    return sorted(strict_survivors, key=strict_selection_key, reverse=True)[0]


def top_current_only_swap_candidates(evaluations: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    current_passed = [
        evaluation
        for evaluation in evaluations
        if evaluation["current"]["strict_gate_passed"] and not evaluation_passes_both(evaluation)
    ]
    ordered = sorted(
        current_passed,
        key=lambda evaluation: (
            evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
            evaluation["current"]["summary"]["lower_stress_pips"] or -1_000_000.0,
        ),
        reverse=True,
    )
    return [compact_policy(evaluation) for evaluation in ordered[:limit]]


def fast169_status(
    *,
    strict_survivors: list[dict[str, Any]],
    strict_selected: dict[str, Any] | None,
    benchmark: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "usd_policy_expansion_audit"]
    if strict_selected is None:
        return (
            "FAST169_NO_STRICT_SURVIVORS_IN_USD_POLICY_EXPANSION",
            reasons + ["fresh_preregistered_prospective_observation_still_required"],
        )
    selected_is_benchmark = (
        strict_selected["basket_id"] == benchmark["basket_id"]
        and strict_selected["entry_policy_id"] == benchmark["entry_policy_id"]
    )
    flags = [
        f"strict_survivor_count:{len(strict_survivors)}",
        f"strict_selected:{strict_selected['basket_id']}:{strict_selected['entry_policy_id']}",
        "strict_selection_is_historical_not_approval",
        "fresh_preregistered_prospective_observation_still_required",
    ]
    if len(strict_survivors) == 1 and selected_is_benchmark:
        return (
            "FAST169_FAST151_UNIQUE_STRICT_SURVIVOR_IN_USD_POLICY_EXPANSION",
            reasons + flags + ["fast151_remains_only_strict_survivor"],
        )
    if selected_is_benchmark:
        return (
            "FAST169_FAST151_REMAINS_SELECTED_WITH_ADDITIONAL_STRICT_SURVIVORS",
            reasons + flags + ["fast151_selected_but_alternatives_require_preregistration"],
        )
    return (
        "FAST169_ALTERNATIVE_USD_POLICY_CANDIDATE_REQUIRES_PREREGISTRATION",
        reasons + flags + ["alternative_selected_historically_not_approved"],
    )


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row: dict[str, Any] = {
            "basket_id": evaluation["basket_id"],
            "pairs": ",".join(evaluation["pairs"]),
            "pair_count": evaluation["pair_count"],
            "entry_policy_id": evaluation["entry_policy_id"],
            "channel_days": evaluation["channel_days"],
            "hold_days": evaluation["hold_days"],
            "pre_strict_gate_passed": evaluation["pre"]["strict_gate_passed"],
            "current_strict_gate_passed": evaluation["current"]["strict_gate_passed"],
            "both_strict_gate_passed": evaluation_passes_both(evaluation),
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


def compact_policy(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "basket_id": evaluation["basket_id"],
        "entry_policy_id": evaluation["entry_policy_id"],
        "hold_days": evaluation["hold_days"],
        "channel_days": evaluation["channel_days"],
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
        "strict_gate_failures": sample["strict_gate_failures"],
        "uniform_swap_break_even": sample["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
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
        "strict_survivor_count": report["summary"]["strict_survivor_count"],
        "benchmark": compact_policy(report["summary"]["benchmark_policy"]),
        "strict_selected": compact_policy(report["summary"]["strict_selected_policy"]),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

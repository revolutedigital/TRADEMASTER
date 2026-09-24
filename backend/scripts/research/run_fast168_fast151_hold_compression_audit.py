"""Fast168: hold-compression audit for the Fast151 operating policy.

Fast151's current candidate is a daily Donchian-fade basket with a 20-day
maximum hold. That creates a real swap/rollover blocker. Fast168 asks the
natural follow-up question: can the same frozen channel/pair/side/risk family
use a shorter hold (5 or 10 days) to preserve edge while improving swap
tolerance?

This is historical research only. It writes offline artifacts and never touches
the trading engine, credentials, ledger, Demo, Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
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
from scripts.research.run_fast162_fast151_management_variant_audit import (  # noqa: E402
    MAX_CONCURRENT_TRADES,
    MAX_CURRENT_DRAWDOWN_PIPS,
    MAX_PRE_DRAWDOWN_PIPS,
    MIN_CURRENT_TRADES,
    MIN_POSITIVE_YEAR_SHARE,
    MIN_PRE_TRADES,
    strict_gate_failures,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast168" / "fast151-hold-compression-audit"
)
ENTRY_POLICIES = (
    "d1_donchian_fade_ch100_hold5",
    "d1_donchian_fade_ch100_hold10",
    ENTRY_POLICY_ID,
)
SELECTED_POSITION_OVERLAY = PositionOverlay("max_per_pair_1", max_total_open=None, max_open_per_pair=1)
SELECTED_CIRCUIT_BREAKER = CircuitBreakerPolicy(
    "dd800_cooldown180d",
    closed_equity_drawdown_threshold_pips=800.0,
    cooldown_days=180,
)


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
    evaluations = [
        evaluate_entry_policy(pre_family, current_family, entry_policy_id)
        for entry_policy_id in ENTRY_POLICIES
    ]
    benchmark = next(evaluation for evaluation in evaluations if evaluation["entry_policy_id"] == ENTRY_POLICY_ID)
    strict_selected = select_strict_entry_policy(evaluations)
    best_current_swap = select_best_current_swap_candidate(evaluations)
    status, status_reasons = fast168_status(
        strict_selected=strict_selected,
        benchmark=benchmark,
        best_current_swap=best_current_swap,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "hold-compression-metrics.csv"
    summary_path = output_root / "hold-compression-summary.json"
    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "benchmark_entry_policy": benchmark,
        "strict_selected_entry_policy": strict_selected,
        "best_current_swap_candidate": best_current_swap,
        "evaluations": evaluations,
        "policy_constants": {
            "target_pairs": list(TARGET_PAIRS),
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
        "interpretation": {
            "question": "whether shorter max-hold variants reduce rollover exposure without using current-period lookahead",
            "strict_selection": "candidate must pass both pre-2019 selection and 2019-2026 audit gates",
            "swap_break_even": "uniform adverse short swap pips per rollover day that drives sample LCB to zero",
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast168_is_historical_hold_compression_audit_not_strategy_approval",
            "shorter_hold_selection_is_not_execution_authorization",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast168_fast151_hold_compression_audit",
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
            "hold_compression_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
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
        & sample["entry_policy_id"].isin(ENTRY_POLICIES)
        & sample["risk_policy_id"].eq(RISK_POLICY_ID)
        & sample["side"].eq(SIDE_FILTER)
        & sample["management"].eq(MANAGEMENT)
    ].sort_values(["entry_timestamp", "pair", "entry_policy_id"], ignore_index=True)


def evaluate_entry_policy(
    pre_family: pd.DataFrame,
    current_family: pd.DataFrame,
    entry_policy_id: str,
) -> dict[str, Any]:
    pre_policy = pre_family.loc[pre_family["entry_policy_id"].eq(entry_policy_id)].copy()
    current_policy = current_family.loc[current_family["entry_policy_id"].eq(entry_policy_id)].copy()
    if pre_policy.empty or current_policy.empty:
        raise ValueError(f"entry policy {entry_policy_id!r} missing from one sample")
    return {
        "entry_policy_id": entry_policy_id,
        "hold_days": int(pre_policy["hold_days"].iloc[0]),
        "channel_days": int(pre_policy["channel_days"].iloc[0]),
        "pre": evaluate_sample(pre_policy, min_trades=MIN_PRE_TRADES, max_drawdown_pips=MAX_PRE_DRAWDOWN_PIPS),
        "current": evaluate_sample(
            current_policy,
            min_trades=MIN_CURRENT_TRADES,
            max_drawdown_pips=MAX_CURRENT_DRAWDOWN_PIPS,
        ),
    }


def evaluate_sample(
    trades: pd.DataFrame,
    *,
    min_trades: int,
    max_drawdown_pips: float,
) -> dict[str, Any]:
    overlay_trades, overlay_skipped = apply_position_overlay(trades, SELECTED_POSITION_OVERLAY)
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
    failures = strict_gate_failures(summary, min_trades=min_trades, max_drawdown_pips=max_drawdown_pips)
    return {
        "summary": summary,
        "position_skipped_count": int(len(overlay_skipped)),
        "breaker_skipped_count": int(len(breaker_skipped)),
        "breaker_trigger_count": int(len(breaker_triggers)),
        "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": swap_break_even,
        "strict_gate_passed": not failures,
        "strict_gate_failures": failures,
    }


def entry_policy_passes_both(evaluation: dict[str, Any]) -> bool:
    return bool(evaluation["pre"]["strict_gate_passed"] and evaluation["current"]["strict_gate_passed"])


def select_strict_entry_policy(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [evaluation for evaluation in evaluations if entry_policy_passes_both(evaluation)]
    if not passed:
        return None
    return sorted(passed, key=strict_selection_key, reverse=True)[0]


def strict_selection_key(evaluation: dict[str, Any]) -> tuple[float, float, float, float]:
    pre = evaluation["pre"]["summary"]
    current = evaluation["current"]["summary"]
    return (
        min(pre["lower_stress_pips"], current["lower_stress_pips"]),
        evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        -abs(current["max_drawdown_pips"]),
        -abs(pre["max_drawdown_pips"]),
    )


def select_best_current_swap_candidate(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    current_passed = [evaluation for evaluation in evaluations if evaluation["current"]["strict_gate_passed"]]
    if not current_passed:
        return None
    return sorted(current_passed, key=current_swap_selection_key, reverse=True)[0]


def current_swap_selection_key(evaluation: dict[str, Any]) -> tuple[float, float, float]:
    current = evaluation["current"]["summary"]
    current_lcb = current["lower_stress_pips"] if current["lower_stress_pips"] is not None else -1_000_000.0
    return (
        evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        current_lcb,
        -abs(current["max_drawdown_pips"]) if current["max_drawdown_pips"] is not None else -1_000_000.0,
    )


def fast168_status(
    *,
    strict_selected: dict[str, Any] | None,
    benchmark: dict[str, Any],
    best_current_swap: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "hold_compression_audit"]
    if strict_selected is None:
        return (
            "FAST168_NO_HOLD_POLICY_PASSES_STRICT_GATE",
            reasons + ["fresh_preregistered_prospective_observation_still_required"],
        )
    flags = [
        f"strict_selected_entry_policy:{strict_selected['entry_policy_id']}",
        "strict_selection_is_historical_not_approval",
        "fresh_preregistered_prospective_observation_still_required",
    ]
    if best_current_swap is not None:
        flags.append(f"best_current_swap_entry_policy:{best_current_swap['entry_policy_id']}")
        if not entry_policy_passes_both(best_current_swap):
            flags.append("best_current_swap_policy_not_pre_robust")
    if strict_selected["entry_policy_id"] == benchmark["entry_policy_id"]:
        return (
            "FAST168_HOLD20_REMAINS_SELECTED_COMPRESSED_HOLDS_FAIL_PRE_GATE",
            reasons + flags + ["benchmark_hold20_remains_selected"],
        )
    return (
        "FAST168_COMPRESSED_HOLD_CANDIDATE_REQUIRES_PREREGISTRATION",
        reasons + flags + ["compressed_hold_selected_historically_not_approved"],
    )


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row: dict[str, Any] = {
            "entry_policy_id": evaluation["entry_policy_id"],
            "channel_days": evaluation["channel_days"],
            "hold_days": evaluation["hold_days"],
            "pre_strict_gate_passed": evaluation["pre"]["strict_gate_passed"],
            "current_strict_gate_passed": evaluation["current"]["strict_gate_passed"],
            "both_strict_gate_passed": entry_policy_passes_both(evaluation),
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


def compact_entry_policy(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "entry_policy_id": evaluation["entry_policy_id"],
        "hold_days": evaluation["hold_days"],
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
        "benchmark": compact_entry_policy(report["summary"]["benchmark_entry_policy"]),
        "strict_selected": compact_entry_policy(report["summary"]["strict_selected_entry_policy"]),
        "best_current_swap": compact_entry_policy(report["summary"]["best_current_swap_candidate"]),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

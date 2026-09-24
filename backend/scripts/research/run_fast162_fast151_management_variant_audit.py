"""Fast162: management-variant audit for the Fast151 operating policy family.

The user originally wanted breakeven/trailing style trade management. Fast151's
current candidate uses fixed 1R exits. Fast162 revisits that decision within
the same daily Donchian-fade family, keeping:

* the same USD-quoted four-pair universe;
* the same SHORT-only entry policy;
* the same max-per-pair position overlay;
* the same causal drawdown circuit breaker.

It compares stop/management variants already produced by the historical grid
(`fixed_1r`, `fixed_2r`, `stop_only`, `trail_1r_1r` across stop multiples).

Historical research only. No orders, no ledger, no Demo/Testnet/LIVE.
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
    REPO_ROOT / "backend" / "data" / "lab_fast162" / "fast151-management-variant-audit"
)
SELECTED_POSITION_OVERLAY = PositionOverlay("max_per_pair_1", max_total_open=None, max_open_per_pair=1)
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
    pre_family = management_family_sample(pre_source, sample_value=PRE_SAMPLE_LABEL)
    current_family = management_family_sample(current_source, sample_value=None)
    variants = sorted(set(pre_family["risk_policy_id"].unique()).intersection(current_family["risk_policy_id"].unique()))
    evaluations = [
        evaluate_management_variant(pre_family, current_family, risk_policy_id)
        for risk_policy_id in variants
    ]
    strict_selected = select_strict_variant(evaluations)
    benchmark = next(evaluation for evaluation in evaluations if evaluation["risk_policy_id"] == RISK_POLICY_ID)
    best_trailing = select_best_management(evaluations, management_name="trail_1r_1r")
    status, status_reasons = fast162_status(
        strict_selected=strict_selected,
        benchmark=benchmark,
        best_trailing=best_trailing,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "management-variant-metrics.csv"
    summary_path = output_root / "management-variant-summary.json"
    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "benchmark_variant": benchmark,
        "strict_selected_variant": strict_selected,
        "best_trailing_variant": best_trailing,
        "evaluations": evaluations,
        "policy_constants": {
            "target_pairs": list(TARGET_PAIRS),
            "entry_policy_id": ENTRY_POLICY_ID,
            "side_filter": SIDE_FILTER,
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
            "fast162_is_historical_management_variant_audit_not_strategy_approval",
            "management_variant_selection_is_not_execution_authorization",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast162_fast151_management_variant_audit",
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
            "management_variant_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def management_family_sample(source: pd.DataFrame, *, sample_value: str | None) -> pd.DataFrame:
    sample = source.copy()
    if sample_value is not None:
        sample = sample.loc[sample["sample"].eq(sample_value)].copy()
    for column in ("signal_timestamp", "entry_timestamp", "exit_base_timestamp", "exit_stress_timestamp"):
        sample[column] = pd.to_datetime(sample[column], utc=True)
    return sample.loc[
        sample["pair"].isin(TARGET_PAIRS)
        & sample["entry_policy_id"].eq(ENTRY_POLICY_ID)
        & sample["side"].eq(SIDE_FILTER)
    ].sort_values(["entry_timestamp", "pair", "risk_policy_id"], ignore_index=True)


def evaluate_management_variant(
    pre_family: pd.DataFrame,
    current_family: pd.DataFrame,
    risk_policy_id: str,
) -> dict[str, Any]:
    pre_variant = pre_family.loc[pre_family["risk_policy_id"].eq(risk_policy_id)].copy()
    current_variant = current_family.loc[current_family["risk_policy_id"].eq(risk_policy_id)].copy()
    if pre_variant.empty or current_variant.empty:
        raise ValueError(f"risk policy {risk_policy_id!r} missing from one sample")
    management = str(pre_variant["management"].iloc[0])
    stop_atr_multiple = float(pre_variant["stop_atr_multiple"].iloc[0])
    return {
        "risk_policy_id": risk_policy_id,
        "management": management,
        "stop_atr_multiple": stop_atr_multiple,
        "pre": evaluate_sample(pre_variant, min_trades=MIN_PRE_TRADES, max_drawdown_pips=MAX_PRE_DRAWDOWN_PIPS),
        "current": evaluate_sample(
            current_variant,
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


def variant_passes_both(evaluation: dict[str, Any]) -> bool:
    return bool(evaluation["pre"]["strict_gate_passed"] and evaluation["current"]["strict_gate_passed"])


def select_strict_variant(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [evaluation for evaluation in evaluations if variant_passes_both(evaluation)]
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


def select_best_management(evaluations: list[dict[str, Any]], *, management_name: str) -> dict[str, Any] | None:
    candidates = [evaluation for evaluation in evaluations if evaluation["management"] == management_name]
    if not candidates:
        return None
    return sorted(candidates, key=diagnostic_selection_key, reverse=True)[0]


def diagnostic_selection_key(evaluation: dict[str, Any]) -> tuple[float, float, float, float]:
    pre = evaluation["pre"]["summary"]
    current = evaluation["current"]["summary"]
    pre_lcb = pre["lower_stress_pips"] if pre["lower_stress_pips"] is not None else -1_000_000.0
    current_lcb = current["lower_stress_pips"] if current["lower_stress_pips"] is not None else -1_000_000.0
    return (
        min(pre_lcb, current_lcb),
        current_lcb,
        evaluation["current"]["uniform_swap_break_even_lcb_zero_pips_per_rollover_day"],
        -abs(current["max_drawdown_pips"]) if current["max_drawdown_pips"] is not None else -1_000_000.0,
    )


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row: dict[str, Any] = {
            "risk_policy_id": evaluation["risk_policy_id"],
            "management": evaluation["management"],
            "stop_atr_multiple": evaluation["stop_atr_multiple"],
            "pre_strict_gate_passed": evaluation["pre"]["strict_gate_passed"],
            "current_strict_gate_passed": evaluation["current"]["strict_gate_passed"],
            "both_strict_gate_passed": variant_passes_both(evaluation),
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


def fast162_status(
    *,
    strict_selected: dict[str, Any] | None,
    benchmark: dict[str, Any],
    best_trailing: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "management_variant_audit"]
    if strict_selected is None:
        return (
            "FAST162_NO_MANAGEMENT_VARIANT_PASSES_STRICT_GATE",
            reasons + ["fresh_preregistered_prospective_observation_still_required"],
        )
    flags = [
        f"strict_selected_variant:{strict_selected['risk_policy_id']}",
        "strict_selection_is_historical_not_approval",
        "fresh_preregistered_prospective_observation_still_required",
    ]
    if strict_selected["risk_policy_id"] == benchmark["risk_policy_id"]:
        flags.append("benchmark_fixed_1r_remains_selected")
    if best_trailing is not None:
        flags.append(f"best_trailing_variant:{best_trailing['risk_policy_id']}")
        if strict_selection_key(best_trailing) < strict_selection_key(benchmark):
            flags.append("best_trailing_weaker_than_benchmark")
    return (
        "FAST162_FIXED_1R_REMAINS_PREFERRED_OVER_TRAILING",
        reasons + flags,
    )


def compact_variant(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "risk_policy_id": evaluation["risk_policy_id"],
        "management": evaluation["management"],
        "stop_atr_multiple": evaluation["stop_atr_multiple"],
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
        "benchmark": compact_variant(report["summary"]["benchmark_variant"]),
        "strict_selected": compact_variant(report["summary"]["strict_selected_variant"]),
        "best_trailing": compact_variant(report["summary"]["best_trailing_variant"]),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

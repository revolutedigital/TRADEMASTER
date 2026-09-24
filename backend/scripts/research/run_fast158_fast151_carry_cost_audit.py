"""Fast158: carry/rollover sensitivity audit for Fast151.

Fast151 holds trades for several calendar days. Even if entry/exit edge is
positive after spread/commission stress, adverse swap/financing can erode the
edge. This audit keeps the frozen Fast151 historical operating-policy trades
and subtracts a conservative fixed carry cost per billed exposure day.

It does not assume a broker-specific swap table. It answers: how many adverse
pips per exposure day can the candidate tolerate before the LCB turns
non-positive?

Historical research only. No orders, no ledger, no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import json
import math
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
    TARGET_POLICY_ID,
    summarize_sample,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)
from scripts.research.run_fast156_fast151_bootstrap_risk_audit import (  # noqa: E402
    DEFAULT_CURRENT_INPUT_PATH,
    DEFAULT_PRE_INPUT_PATH,
    load_trades,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast158" / "fast151-carry-cost-audit"


@dataclass(frozen=True)
class CarryScenario:
    identifier: str
    adverse_pips_per_billed_day: float


def carry_scenarios() -> tuple[CarryScenario, ...]:
    return (
        CarryScenario("no_extra_carry", 0.0),
        CarryScenario("carry_0_10_pip_per_day", 0.10),
        CarryScenario("carry_0_25_pip_per_day", 0.25),
        CarryScenario("carry_0_50_pip_per_day", 0.50),
        CarryScenario("carry_0_75_pip_per_day", 0.75),
        CarryScenario("carry_1_00_pip_per_day", 1.00),
        CarryScenario("carry_1_50_pip_per_day", 1.50),
        CarryScenario("carry_2_00_pip_per_day", 2.00),
        CarryScenario("carry_3_00_pip_per_day", 3.00),
        CarryScenario("carry_5_00_pip_per_day", 5.00),
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
    pre = add_carry_days(load_trades(pre_input_path))
    current = add_carry_days(load_trades(current_input_path))
    evaluations = [evaluate_carry_scenario(pre, current, scenario) for scenario in carry_scenarios()]
    baseline = next(evaluation for evaluation in evaluations if evaluation["scenario_id"] == "no_extra_carry")
    break_even = {
        "pre_lcb_zero_pips_per_day": estimate_lcb_break_even_pips_per_day(pre),
        "current_lcb_zero_pips_per_day": estimate_lcb_break_even_pips_per_day(current),
    }
    status, status_reasons = fast158_status(evaluations=evaluations, break_even=break_even)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "carry-cost-metrics.csv"
    summary_path = output_root / "carry-cost-summary.json"
    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "baseline": baseline,
        "evaluations": evaluations,
        "break_even": break_even,
        "carry_day_diagnostics": {
            "pre": carry_day_diagnostics(pre),
            "current": carry_day_diagnostics(current),
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast158_is_historical_carry_cost_sensitivity_audit_not_strategy_approval",
            "broker_specific_swap_must_be_measured_before_shadow_or_testnet",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast158_fast151_carry_cost_audit",
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
            "carry_cost_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def add_carry_days(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    elapsed_days = (frame["exit_stress_timestamp"] - frame["entry_timestamp"]).dt.total_seconds() / 86_400
    frame["carry_elapsed_days"] = elapsed_days.astype("float64")
    frame["carry_billed_days"] = elapsed_days.map(lambda value: max(1, math.ceil(float(value)))).astype("int64")
    return frame


def apply_carry_cost(trades: pd.DataFrame, adverse_pips_per_billed_day: float) -> pd.DataFrame:
    adjusted = trades.copy()
    adjusted["carry_cost_pips"] = adjusted["carry_billed_days"].astype("float64") * adverse_pips_per_billed_day
    adjusted["net_stress_pips"] = adjusted["net_stress_pips"].astype("float64") - adjusted["carry_cost_pips"]
    return adjusted


def evaluate_carry_scenario(
    pre: pd.DataFrame,
    current: pd.DataFrame,
    scenario: CarryScenario,
) -> dict[str, Any]:
    pre_adjusted = apply_carry_cost(pre, scenario.adverse_pips_per_billed_day)
    current_adjusted = apply_carry_cost(current, scenario.adverse_pips_per_billed_day)
    pre_summary = summarize_sample(pre_adjusted)
    current_summary = summarize_sample(current_adjusted)
    return {
        "scenario_id": scenario.identifier,
        "adverse_pips_per_billed_day": scenario.adverse_pips_per_billed_day,
        "pre_total_billed_days": int(pre["carry_billed_days"].sum()),
        "current_total_billed_days": int(current["carry_billed_days"].sum()),
        "pre": pre_summary,
        "current": current_summary,
        "pre_lcb_positive": bool(pre_summary["lower_stress_pips"] is not None and pre_summary["lower_stress_pips"] > 0),
        "current_lcb_positive": bool(
            current_summary["lower_stress_pips"] is not None and current_summary["lower_stress_pips"] > 0
        ),
    }


def estimate_lcb_break_even_pips_per_day(trades: pd.DataFrame) -> float:
    low = 0.0
    high = 10.0
    while summarize_sample(apply_carry_cost(trades, high))["lower_stress_pips"] > 0:
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        if summarize_sample(apply_carry_cost(trades, mid))["lower_stress_pips"] > 0:
            low = mid
        else:
            high = mid
    return high


def carry_day_diagnostics(trades: pd.DataFrame) -> dict[str, Any]:
    return {
        "trade_count": int(len(trades)),
        "total_billed_days": int(trades["carry_billed_days"].sum()),
        "mean_billed_days": float(trades["carry_billed_days"].mean()),
        "median_billed_days": float(trades["carry_billed_days"].median()),
        "max_billed_days": int(trades["carry_billed_days"].max()),
        "mean_elapsed_days": float(trades["carry_elapsed_days"].mean()),
        "median_elapsed_days": float(trades["carry_elapsed_days"].median()),
    }


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row = {
            key: value
            for key, value in evaluation.items()
            if key not in {"pre", "current"}
        }
        for prefix in ("pre", "current"):
            summary = evaluation[prefix]
            for key, value in summary.items():
                if key in {"yearly", "pair_summary", "exit_reason_summary"}:
                    continue
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def fast158_status(
    *,
    evaluations: list[dict[str, Any]],
    break_even: dict[str, float],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "carry_cost_audit"]
    scenario_by_id = {evaluation["scenario_id"]: evaluation for evaluation in evaluations}
    flags: list[str] = []
    if not scenario_by_id["carry_0_75_pip_per_day"]["current_lcb_positive"]:
        flags.append("current_lcb_fails_at_0_75_pip_per_day")
    if not scenario_by_id["carry_1_00_pip_per_day"]["current_lcb_positive"]:
        flags.append("current_lcb_fails_at_1_00_pip_per_day")
    if break_even["current_lcb_zero_pips_per_day"] < 1.0:
        flags.append("current_lcb_break_even_below_1_pip_per_day")
    if flags:
        return (
            "FAST158_FAST151_CARRY_MARGIN_POSITIVE_BUT_TIGHT_REQUIRES_SWAP_MEASUREMENT",
            reasons + flags + ["fresh_preregistered_prospective_observation_still_required"],
        )
    return (
        "FAST158_FAST151_CARRY_SENSITIVITY_CLEAN_REQUIRES_PROSPECTIVE_OBSERVATION",
        reasons + ["fresh_preregistered_prospective_observation_required_before_any_execution"],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    baseline = report["summary"]["baseline"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "operational_policy_id": report["operational_policy_id"],
        "baseline_pre": {
            key: baseline["pre"][key]
            for key in ("trade_count", "mean_stress_pips", "lower_stress_pips", "max_drawdown_pips")
        },
        "baseline_current": {
            key: baseline["current"][key]
            for key in ("trade_count", "mean_stress_pips", "lower_stress_pips", "max_drawdown_pips")
        },
        "break_even": report["summary"]["break_even"],
        "carry_day_diagnostics": report["summary"]["carry_day_diagnostics"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

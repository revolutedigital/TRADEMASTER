"""Fast154: cost-sensitivity audit for the Fast151 operating policy.

Fast151 remains research-only and still lacks prospective observations. This
audit keeps the frozen operating-policy trades from Fast150 and asks how much
extra per-trade friction the candidate can tolerate before the lower confidence
bound stops being positive.

No entries, exits, pairs, position caps, or circuit breakers are changed. The
script only rewrites `net_stress_pips` in copies of the selected historical
trade rows to model harsher cost scenarios.
"""

from __future__ import annotations

import argparse
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
    TARGET_POLICY_ID,
    summarize_sample,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)

DEFAULT_PRE_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast150"
    / "usd-quoted-basket-circuit-breaker-audit"
    / "selected-breaker-pre-trades.csv"
)
DEFAULT_CURRENT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast150"
    / "usd-quoted-basket-circuit-breaker-audit"
    / "selected-breaker-current-trades.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast154" / "fast151-cost-sensitivity-audit"
TIMESTAMP_COLUMNS = ("signal_timestamp", "entry_timestamp", "exit_base_timestamp", "exit_stress_timestamp")


@dataclass(frozen=True)
class CostScenario:
    identifier: str
    extra_fixed_pips: float = 0.0
    base_to_stress_drag_multiplier: float | None = None


def cost_scenarios() -> tuple[CostScenario, ...]:
    return (
        CostScenario("stress_as_recorded"),
        CostScenario("stress_plus_1pip", extra_fixed_pips=1.0),
        CostScenario("stress_plus_2pips", extra_fixed_pips=2.0),
        CostScenario("stress_plus_5pips", extra_fixed_pips=5.0),
        CostScenario("stress_plus_10pips", extra_fixed_pips=10.0),
        CostScenario("stress_plus_15pips", extra_fixed_pips=15.0),
        CostScenario("double_base_to_stress_drag", base_to_stress_drag_multiplier=2.0),
        CostScenario("triple_base_to_stress_drag", base_to_stress_drag_multiplier=3.0),
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
    pre = load_trades(pre_input_path)
    current = load_trades(current_input_path)
    evaluations = [evaluate_scenario(pre, current, scenario) for scenario in cost_scenarios()]
    baseline = next(evaluation for evaluation in evaluations if evaluation["scenario_id"] == "stress_as_recorded")
    status, status_reasons = fast154_status(evaluations)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "cost-sensitivity-metrics.csv"
    summary_path = output_root / "cost-sensitivity-summary.json"
    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "baseline": baseline,
        "evaluations": evaluations,
        "break_even_extra_fixed_pips": {
            "pre_lcb_to_zero": baseline["pre"]["lower_stress_pips"],
            "current_lcb_to_zero": baseline["current"]["lower_stress_pips"],
            "pre_mean_to_zero": baseline["pre"]["mean_stress_pips"],
            "current_mean_to_zero": baseline["current"]["mean_stress_pips"],
        },
        "cost_drag_diagnostics": {
            "pre": cost_drag_diagnostics(pre),
            "current": cost_drag_diagnostics(current),
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast154_is_historical_cost_sensitivity_audit_not_strategy_approval",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast154_fast151_cost_sensitivity_audit",
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
            "cost_sensitivity_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values(["entry_timestamp", "pair", "candidate_id"], ignore_index=True)


def apply_cost_scenario(trades: pd.DataFrame, scenario: CostScenario) -> pd.DataFrame:
    adjusted = trades.copy()
    if scenario.base_to_stress_drag_multiplier is not None:
        base_to_stress_drag = adjusted["net_base_pips"].astype("float64") - adjusted["net_stress_pips"].astype(
            "float64"
        )
        adjusted["net_stress_pips"] = adjusted["net_base_pips"].astype("float64") - (
            base_to_stress_drag * scenario.base_to_stress_drag_multiplier
        )
    else:
        adjusted["net_stress_pips"] = adjusted["net_stress_pips"].astype("float64") - scenario.extra_fixed_pips
    return adjusted


def evaluate_scenario(
    pre: pd.DataFrame,
    current: pd.DataFrame,
    scenario: CostScenario,
) -> dict[str, Any]:
    pre_adjusted = apply_cost_scenario(pre, scenario)
    current_adjusted = apply_cost_scenario(current, scenario)
    pre_summary = summarize_sample(pre_adjusted)
    current_summary = summarize_sample(current_adjusted)
    return {
        "scenario_id": scenario.identifier,
        "extra_fixed_pips": scenario.extra_fixed_pips,
        "base_to_stress_drag_multiplier": scenario.base_to_stress_drag_multiplier,
        "pre": pre_summary,
        "current": current_summary,
        "pre_lcb_positive": bool(pre_summary["lower_stress_pips"] is not None and pre_summary["lower_stress_pips"] > 0),
        "current_lcb_positive": bool(
            current_summary["lower_stress_pips"] is not None and current_summary["lower_stress_pips"] > 0
        ),
        "both_lcb_positive": bool(
            pre_summary["lower_stress_pips"] is not None
            and pre_summary["lower_stress_pips"] > 0
            and current_summary["lower_stress_pips"] is not None
            and current_summary["lower_stress_pips"] > 0
        ),
    }


def cost_drag_diagnostics(trades: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_base_execution_cost_pips": float((trades["raw_base_pips"] - trades["net_base_pips"]).mean()),
        "mean_stress_execution_cost_pips": float((trades["raw_stress_pips"] - trades["net_stress_pips"]).mean()),
        "mean_quote_stress_delta_pips": float((trades["raw_base_pips"] - trades["raw_stress_pips"]).mean()),
        "mean_total_base_to_stress_delta_pips": float((trades["net_base_pips"] - trades["net_stress_pips"]).mean()),
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


def fast154_status(evaluations: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "cost_sensitivity_audit"]
    scenario_by_id = {evaluation["scenario_id"]: evaluation for evaluation in evaluations}
    stress_plus_5 = scenario_by_id["stress_plus_5pips"]
    stress_plus_10 = scenario_by_id["stress_plus_10pips"]
    double_drag = scenario_by_id["double_base_to_stress_drag"]
    flags: list[str] = []
    if not stress_plus_5["both_lcb_positive"]:
        flags.append("fails_extra_5_pips_lcb_gate")
    if not double_drag["both_lcb_positive"]:
        flags.append("fails_double_base_to_stress_drag_lcb_gate")
    if not stress_plus_10["both_lcb_positive"]:
        flags.append("fails_extra_10_pips_lcb_gate")
    if flags:
        return (
            "FAST154_FAST151_COST_MARGIN_POSITIVE_BUT_NOT_UNLIMITED_REQUIRES_PROSPECTIVE_OBSERVATION",
            reasons + flags + ["fresh_preregistered_prospective_observation_still_required"],
        )
    return (
        "FAST154_FAST151_COST_SENSITIVITY_CLEAN_REQUIRES_PROSPECTIVE_OBSERVATION",
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
            for key in ("trade_count", "mean_stress_pips", "lower_stress_pips", "total_stress_pips")
        },
        "baseline_current": {
            key: baseline["current"][key]
            for key in ("trade_count", "mean_stress_pips", "lower_stress_pips", "total_stress_pips")
        },
        "break_even_extra_fixed_pips": report["summary"]["break_even_extra_fixed_pips"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

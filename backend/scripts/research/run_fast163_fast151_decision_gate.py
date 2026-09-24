"""Fast163: decision gate for the Fast151 operating candidate.

Fast151 and the follow-up audits found a historical candidate, then tested
robustness, sizing, swap, basket composition and management variants. This
runner turns those separate reports into one fail-closed decision:

* what evidence supports the candidate;
* what still blocks shadow/Testnet/LIVE;
* what exact next actions unblock the next research stage.

Historical/research coordination only. No orders, no ledger, no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast163" / "fast151-decision-gate"
DEFAULT_REPORT_PATHS = {
    "fast152_readiness": REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast152"
    / "fast151-readiness-monitor"
    / "report.json",
    "fast156_bootstrap_risk": REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast156"
    / "fast151-bootstrap-risk-audit"
    / "report.json",
    "fast157_sizing": REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast157"
    / "fast151-sizing-envelope-audit"
    / "report.json",
    "fast159_exact_swap": REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast159"
    / "fast151-exact-swap-rollover-audit"
    / "report.json",
    "fast161_subbasket": REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast161"
    / "fast151-subbasket-robustness-audit"
    / "report.json",
    "fast162_management": REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast162"
    / "fast151-management-variant-audit"
    / "report.json",
}
BROKER_SWAP_TEMPLATE_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast159"
    / "fast151-exact-swap-rollover-audit"
    / "broker-swap-template.csv"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    loaded_reports = load_required_reports(DEFAULT_REPORT_PATHS)
    gate = evaluate_decision_gate(loaded_reports)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "decision-gate-summary.json"
    write_json(summary_path, gate)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast163_is_research_decision_gate_not_strategy_approval",
            "no_shadow_testnet_or_live_without_fresh_forward_evidence",
            "no_shadow_testnet_or_live_without_broker_swap_table_gate",
            "explicit_user_authorization_required_for_any_execution_stage",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast163_fast151_decision_gate",
        "status": gate["status"],
        "status_reasons": gate["status_reasons"],
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "summary": gate,
        "input_reports": input_report_artifacts(DEFAULT_REPORT_PATHS),
        "artifacts": {
            "decision_gate_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)}
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def load_required_reports(report_paths: dict[str, Path]) -> dict[str, dict[str, Any] | None]:
    loaded: dict[str, dict[str, Any] | None] = {}
    for identifier, path in report_paths.items():
        if not path.exists():
            loaded[identifier] = None
            continue
        loaded[identifier] = json.loads(path.read_text())
    return loaded


def input_report_artifacts(report_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for identifier, path in report_paths.items():
        artifacts[identifier] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return artifacts


def evaluate_decision_gate(reports: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    missing = sorted(identifier for identifier, report in reports.items() if report is None)
    if missing:
        return missing_artifact_gate(missing)

    fast152 = require_report(reports, "fast152_readiness")
    fast156 = require_report(reports, "fast156_bootstrap_risk")
    fast157 = require_report(reports, "fast157_sizing")
    fast159 = require_report(reports, "fast159_exact_swap")
    fast161 = require_report(reports, "fast161_subbasket")
    fast162 = require_report(reports, "fast162_management")

    evidence = {
        "historical_edge": historical_edge_summary(fast156),
        "sizing_envelope": sizing_summary(fast157),
        "swap_gate": swap_summary(fast159),
        "basket_gate": basket_summary(fast161),
        "management_gate": management_summary(fast162),
        "forward_readiness": forward_summary(fast152),
    }
    blockers = hard_blockers(fast152, fast159)
    warnings = risk_warnings(fast156, fast157, fast159)
    next_actions = next_actions_for(blockers)
    status, status_reasons = decision_status(blockers, warnings)
    return {
        "status": status,
        "status_reasons": status_reasons,
        "decision": "RESEARCH_CANDIDATE_NOT_APPROVED",
        "approved_strategy": False,
        "shadow_allowed": False,
        "testnet_allowed": False,
        "live_allowed": False,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "evidence": evidence,
        "hard_blockers": blockers,
        "risk_warnings": warnings,
        "next_actions": next_actions,
        "broker_swap_template_path": str(BROKER_SWAP_TEMPLATE_PATH),
    }


def require_report(reports: dict[str, dict[str, Any] | None], identifier: str) -> dict[str, Any]:
    report = reports[identifier]
    if report is None:
        raise ValueError(f"missing required report: {identifier}")
    return report


def missing_artifact_gate(missing: list[str]) -> dict[str, Any]:
    return {
        "status": "FAST163_MISSING_REQUIRED_REPORTS",
        "status_reasons": ["not_approved_strategy", "research_only", "missing_required_reports"],
        "decision": "RESEARCH_INCOMPLETE",
        "approved_strategy": False,
        "shadow_allowed": False,
        "testnet_allowed": False,
        "live_allowed": False,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "evidence": {},
        "hard_blockers": [{"id": "missing_required_reports", "reports": missing}],
        "risk_warnings": [],
        "next_actions": [{"id": "rerun_missing_reports", "missing_reports": missing}],
        "broker_swap_template_path": str(BROKER_SWAP_TEMPLATE_PATH),
    }


def historical_edge_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "status": report["status"],
        "pre": summary["pre"]["actual"],
        "current": summary["current"]["actual"],
        "current_trade_bootstrap_probability_total_lte_0": summary["current"]["trade_bootstrap"][
            "probability_total_pips_lte_0"
        ],
        "pre_year_block_probability_drawdown_lte_1500": summary["pre"]["year_block_bootstrap"][
            "probability_max_drawdown_lte_1500_pips"
        ],
        "current_year_block_probability_at_least_3_losing_years": summary["current"]["year_block_bootstrap"][
            "probability_at_least_3_losing_years"
        ],
    }


def sizing_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    by_budget = {
        str(row["drawdown_budget_fraction"]): row for row in summary["sizing_envelope"]
    }
    by_risk = {
        str(row["risk_per_trade_fraction"]): row for row in summary["risk_candidate_drawdowns"]
    }
    return {
        "status": report["status"],
        "worst_tail_drawdown_r_p05": summary["worst_tail_drawdown_r_p05"],
        "max_risk_for_10pct_drawdown_budget_percent": by_budget["0.1"]["max_risk_per_trade_percent"],
        "risk_0_25pct_tail_drawdown_percent": by_risk["0.0025"]["tail_drawdown_percent"],
        "risk_0_50pct_tail_drawdown_percent": by_risk["0.005"]["tail_drawdown_percent"],
        "risk_1_00pct_tail_drawdown_percent": by_risk["0.01"]["tail_drawdown_percent"],
    }


def swap_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "status": report["status"],
        "broker_swap_path": report.get("broker_swap_path"),
        "broker_swap_template_path": str(BROKER_SWAP_TEMPLATE_PATH),
        "current_break_even_pips_per_rollover_day": summary["break_even"][
            "current_lcb_zero_pips_per_rollover_day"
        ],
        "pre_break_even_pips_per_rollover_day": summary["break_even"][
            "pre_lcb_zero_pips_per_rollover_day"
        ],
        "current_mean_rollover_days": summary["rollover_day_diagnostics"]["current"]["mean_rollover_days"],
        "current_total_rollover_days": summary["rollover_day_diagnostics"]["current"]["total_rollover_days"],
    }


def basket_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["summary"]["strict_selected_basket"]
    return {
        "status": report["status"],
        "selected_basket_id": selected["basket_id"],
        "pre_lcb": selected["pre"]["summary"]["lower_stress_pips"],
        "current_lcb": selected["current"]["summary"]["lower_stress_pips"],
        "current_trade_count": selected["current"]["summary"]["trade_count"],
    }


def management_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["summary"]["strict_selected_variant"]
    trailing = report["summary"]["best_trailing_variant"]
    return {
        "status": report["status"],
        "selected_risk_policy_id": selected["risk_policy_id"],
        "selected_current_lcb": selected["current"]["summary"]["lower_stress_pips"],
        "selected_current_swap_break_even": selected["current"][
            "uniform_swap_break_even_lcb_zero_pips_per_rollover_day"
        ],
        "best_trailing_risk_policy_id": trailing["risk_policy_id"] if trailing else None,
        "best_trailing_current_lcb": trailing["current"]["summary"]["lower_stress_pips"] if trailing else None,
    }


def forward_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "has_post_preregistration_m1_data": report["has_post_preregistration_m1_data"],
        "eligible_observation_count": report["eligible_observation_count"],
        "base_candidate_count": report["base_candidate_count"],
        "fast151_observation_status": report["fast151_observation_status"],
        "raw_m1_min_pair_end": report.get("raw_m1_min_pair_end"),
        "next_action": report.get("next_action"),
    }


def hard_blockers(fast152: dict[str, Any], fast159: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not fast152["has_post_preregistration_m1_data"]:
        blockers.append(
            {
                "id": "post_preregistration_m1_data_missing",
                "detail": "local and refreshed HistData M1 do not yet include bars after Fast151 preregistration",
                "raw_m1_min_pair_end": fast152.get("raw_m1_min_pair_end"),
                "required_next_action": "refresh_or_import_m1_after_fast151_preregistration_then_rerun_fast151_and_fast152",
            }
        )
    if fast152["eligible_observation_count"] <= 0:
        blockers.append(
            {
                "id": "no_prospective_operational_observation_yet",
                "eligible_observation_count": fast152["eligible_observation_count"],
                "fast151_observation_status": fast152["fast151_observation_status"],
            }
        )
    if fast159.get("broker_swap_path") is None:
        blockers.append(
            {
                "id": "broker_swap_table_missing",
                "template_path": str(BROKER_SWAP_TEMPLATE_PATH),
                "required_next_action": "fill broker short/long swap pips per rollover day and rerun Fast159",
            }
        )
    blockers.append(
        {
            "id": "explicit_execution_authorization_missing",
            "required_next_action": "only after research gates pass, require explicit user authorization for shadow/Testnet",
        }
    )
    return blockers


def risk_warnings(fast156: dict[str, Any], fast157: dict[str, Any], fast159: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "bootstrap_drawdown_tail",
            "pre_year_block_probability_drawdown_lte_1000_pips": fast156["summary"]["pre"]["year_block_bootstrap"][
                "probability_max_drawdown_lte_1000_pips"
            ],
            "current_trade_probability_drawdown_lte_1000_pips": fast156["summary"]["current"]["trade_bootstrap"][
                "probability_max_drawdown_lte_1000_pips"
            ],
        },
        {
            "id": "conservative_sizing_required",
            "worst_tail_drawdown_r_p05": fast157["summary"]["worst_tail_drawdown_r_p05"],
            "risk_0_25pct_tail_drawdown_percent": next(
                row["tail_drawdown_percent"]
                for row in fast157["summary"]["risk_candidate_drawdowns"]
                if row["risk_per_trade_fraction"] == 0.0025
            ),
            "risk_1pct_tail_drawdown_percent": next(
                row["tail_drawdown_percent"]
                for row in fast157["summary"]["risk_candidate_drawdowns"]
                if row["risk_per_trade_fraction"] == 0.01
            ),
        },
        {
            "id": "tight_swap_margin",
            "current_break_even_pips_per_rollover_day": fast159["summary"]["break_even"][
                "current_lcb_zero_pips_per_rollover_day"
            ],
        },
    ]


def next_actions_for(blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    blocker_ids = {blocker["id"] for blocker in blockers}
    if "post_preregistration_m1_data_missing" in blocker_ids:
        actions.append(
            {
                "id": "refresh_forward_m1",
                "command": "backend/.venv/bin/python backend/scripts/research/run_fast152_fast151_readiness_monitor.py",
                "done_when": "status is no longer FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA",
            }
        )
    if "broker_swap_table_missing" in blocker_ids:
        actions.append(
            {
                "id": "fill_broker_swap_table",
                "template_path": str(BROKER_SWAP_TEMPLATE_PATH),
                "rerun_command": "backend/.venv/bin/python backend/scripts/research/run_fast159_fast151_exact_swap_rollover_audit.py --broker-swap-path <filled.csv>",
                "done_when": "Fast159 status is FAST159_FAST151_BROKER_SWAP_TABLE_PASSES_REQUIRES_PROSPECTIVE_OBSERVATION",
            }
        )
    actions.append(
        {
            "id": "keep_research_fail_closed",
            "done_when": "shadow/Testnet/LIVE remain false until all gates pass and explicit authorization is given",
        }
    )
    return actions


def decision_status(
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],  # noqa: ARG001 - kept for future gate branching
) -> tuple[str, list[str]]:
    blocker_ids = {blocker["id"] for blocker in blockers}
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "fast151_decision_gate"]
    if {
        "post_preregistration_m1_data_missing",
        "broker_swap_table_missing",
    }.issubset(blocker_ids):
        return (
            "FAST163_FAST151_CANDIDATE_BLOCKED_BY_FORWARD_DATA_AND_BROKER_SWAP",
            reasons + sorted(blocker_ids),
        )
    if "post_preregistration_m1_data_missing" in blocker_ids:
        return "FAST163_FAST151_CANDIDATE_BLOCKED_BY_FORWARD_DATA", reasons + sorted(blocker_ids)
    if "broker_swap_table_missing" in blocker_ids:
        return "FAST163_FAST151_CANDIDATE_BLOCKED_BY_BROKER_SWAP", reasons + sorted(blocker_ids)
    return (
        "FAST163_FAST151_RESEARCH_REVIEW_READY_REQUIRES_EXPLICIT_AUTHORIZATION",
        reasons + sorted(blocker_ids),
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "decision": summary["decision"],
        "shadow_allowed": summary["shadow_allowed"],
        "testnet_allowed": summary["testnet_allowed"],
        "live_allowed": summary["live_allowed"],
        "hard_blockers": summary["hard_blockers"],
        "next_actions": summary["next_actions"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

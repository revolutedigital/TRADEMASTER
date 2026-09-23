"""Fast63 economic feasibility audit for the closest short-horizon edge."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_FAST40_REPORT = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "book-payoff-model-scout-fast40-30s-short-horizon-100ms"
    / "report.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast63-closest-edge-feasibility-audit"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast40-report", type=Path, default=DEFAULT_FAST40_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-window-cost-buffer-bps", type=float, default=0.25)
    arguments = parser.parse_args()

    report = run_fast63_closest_edge_feasibility_audit(
        fast40_report_path=arguments.fast40_report,
        output_root=arguments.output_root,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_window_cost_buffer_bps=arguments.minimum_window_cost_buffer_bps,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast63_closest_edge_feasibility_audit(
    *,
    fast40_report_path: Path,
    output_root: Path,
    stress_cost_bps: float = 2.0,
    minimum_window_cost_buffer_bps: float = 0.25,
) -> dict[str, Any]:
    if stress_cost_bps < 0:
        raise ValueError("stress_cost_bps cannot be negative")
    if minimum_window_cost_buffer_bps < 0:
        raise ValueError("minimum_window_cost_buffer_bps cannot be negative")

    fast40_report = json.loads(fast40_report_path.read_text(encoding="utf-8"))
    _validate_fast40_boundary(fast40_report)
    feasibility = closest_edge_feasibility(
        fast40_report,
        stress_cost_bps=stress_cost_bps,
        minimum_window_cost_buffer_bps=minimum_window_cost_buffer_bps,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast63_audits_existing_failed_fast40_candidate",
            "closest_edge_is_too_thin_in_worst_window",
            "stress_cost_result_is_negative",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast63_closest_edge_feasibility_audit",
        "status": feasibility["status"],
        "status_reasons": feasibility["status_reasons"],
        "fast40_report_path": str(fast40_report_path),
        "fast40_report_sha256": _sha256_file(fast40_report_path),
        **feasibility,
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def closest_edge_feasibility(
    fast40_report: dict[str, Any],
    *,
    stress_cost_bps: float,
    minimum_window_cost_buffer_bps: float,
) -> dict[str, Any]:
    best = fast40_report["best_candidate"]
    one_position = best["aggregate_by_mode"]["one_position"]
    net_mean_by_cost = {float(key): float(value) for key, value in one_position["net_mean_by_cost_bps"].items()}
    net_total_by_cost = {
        float(key): float(value) for key, value in one_position["net_total_by_cost_bps"].items()
    }
    positive_windows_by_cost = {
        float(key): int(value)
        for key, value in one_position["positive_window_count_by_cost_bps"].items()
    }
    holdouts = [
        {
            "label": str(holdout["label"]),
            "one_position_executable_count": int(
                holdout["one_position_summary"]["executable_count"]
            ),
            "break_even_explicit_cost_bps": float(
                holdout["one_position_summary"]["break_even_explicit_cost_bps"]
            ),
            "net_mean_at_0bps": _cost_sensitivity_mean(holdout, 0.0),
            "net_mean_at_1bps": _cost_sensitivity_mean(holdout, 1.0),
            "net_mean_at_2bps": _cost_sensitivity_mean(holdout, 2.0),
        }
        for holdout in best["holdouts"]
    ]
    worst_window = min(holdouts, key=lambda row: row["break_even_explicit_cost_bps"])
    holdout_count = int(best["holdout_count"])
    aggregate_break_even_cost_bps = float(one_position["gross_mean_bps"])
    stress_net_mean_bps = net_mean_by_cost.get(stress_cost_bps)
    stress_net_total_bps = net_total_by_cost.get(stress_cost_bps)
    stress_positive_windows = positive_windows_by_cost.get(stress_cost_bps, 0)
    if stress_net_mean_bps is None or stress_net_total_bps is None:
        raise ValueError(f"Fast40 report has no cost sensitivity for {stress_cost_bps} bps")

    status_reasons = [
        "not_approved_strategy",
        "diagnostic_only",
        "fast40_already_failed_replication",
    ]
    if stress_net_total_bps <= 0:
        status_reasons.append(f"stress_total_non_positive_at_{stress_cost_bps:g}bps")
    if stress_positive_windows < holdout_count:
        status_reasons.append(
            f"stress_positive_windows_below_holdout_count:{stress_positive_windows}<{holdout_count}"
        )
    if worst_window["break_even_explicit_cost_bps"] <= minimum_window_cost_buffer_bps:
        status_reasons.append(
            "worst_window_break_even_cost_below_buffer:"
            f"{worst_window['break_even_explicit_cost_bps']}<={minimum_window_cost_buffer_bps}"
        )

    status = (
        "FAST63_CLOSEST_EDGE_TOO_THIN_AND_REGIME_UNSTABLE"
        if len(status_reasons) > 3
        else "FAST63_CLOSEST_EDGE_REQUIRES_MORE_AUDIT"
    )
    return {
        "status": status,
        "status_reasons": status_reasons,
        "source_status": fast40_report["status"],
        "candidate": {
            "model": best["model"],
            "target_bps": float(best["target_bps"]),
            "trailing_distance_bps": float(best["trailing_distance_bps"]),
            "top_fraction": float(best["top_fraction"]),
            "selected_signal_count": int(best["selected_signal_count"]),
            "one_position_executable_count": int(one_position["executable_count"]),
        },
        "aggregate_break_even_explicit_cost_bps": aggregate_break_even_cost_bps,
        "aggregate_stress_net_mean_bps": stress_net_mean_bps,
        "aggregate_stress_net_total_bps": stress_net_total_bps,
        "stress_positive_window_count": stress_positive_windows,
        "holdout_count": holdout_count,
        "worst_window": worst_window,
        "holdouts": holdouts,
        "minimum_window_cost_buffer_bps": minimum_window_cost_buffer_bps,
    }


def _cost_sensitivity_mean(holdout: dict[str, Any], cost_bps: float) -> float:
    for row in holdout["one_position_summary"]["cost_sensitivity"]:
        if float(row["explicit_cost_bps"]) == cost_bps:
            return float(row["net_executable_mean_bps"])
    raise ValueError(f"holdout {holdout['label']} has no {cost_bps} bps row")


def _validate_fast40_boundary(report: dict[str, Any]) -> None:
    if report.get("research_only") is not True:
        raise ValueError("Fast40 report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError("Fast40 report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError("Fast40 report must have execution_authorization=none")
    if report.get("approved_strategy") is not False:
        raise ValueError("Fast40 report must not be an approved strategy")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate": report["candidate"],
        "aggregate_break_even_explicit_cost_bps": report["aggregate_break_even_explicit_cost_bps"],
        "aggregate_stress_net_total_bps": report["aggregate_stress_net_total_bps"],
        "stress_positive_window_count": report["stress_positive_window_count"],
        "holdout_count": report["holdout_count"],
        "worst_window": report["worst_window"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

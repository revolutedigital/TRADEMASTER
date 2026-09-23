"""Summarize regime-gate candidates across walk-forward report folds.

This script does not fit a new trading rule. It audits candidate stability from
previously generated regime-gate reports, grouping the same feature/direction/
fraction across folds and requiring confirmation performance in every fold
before naming a candidate for future holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-paths", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("backend/data/microstructure_v1/reports/book-regime-gate-walkforward"),
    )
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--minimum-confirmation-selected-count", type=int, default=10)
    parser.add_argument("--max-report-candidates", type=int, default=50)
    arguments = parser.parse_args()
    report = summarize_walkforward_reports(
        report_paths=_parse_paths(arguments.report_paths),
        output_root=arguments.output_root,
        profit_cost_bps=arguments.profit_cost_bps,
        minimum_confirmation_selected_count=(
            arguments.minimum_confirmation_selected_count
        ),
        max_report_candidates=arguments.max_report_candidates,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def summarize_walkforward_reports(
    *,
    report_paths: tuple[Path, ...],
    output_root: Path,
    profit_cost_bps: float = 1.0,
    minimum_confirmation_selected_count: int = 10,
    max_report_candidates: int = 50,
) -> dict[str, object]:
    if not report_paths:
        raise ValueError("report_paths cannot be empty")
    if profit_cost_bps < 0:
        raise ValueError("profit_cost_bps must be non-negative")
    if minimum_confirmation_selected_count < 1:
        raise ValueError("minimum_confirmation_selected_count must be positive")
    if max_report_candidates < 1:
        raise ValueError("max_report_candidates must be positive")

    folds: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = {}
    complete_candidate_exports = True
    for fold_index, report_path in enumerate(report_paths):
        source = json.loads(report_path.read_text(encoding="utf-8"))
        candidates = source.get("top_candidates_by_gate_development", [])
        if not isinstance(candidates, list):
            candidates = []
        candidates_evaluated = int(source.get("candidates_evaluated", 0) or 0)
        complete_export = len(candidates) >= candidates_evaluated
        complete_candidate_exports = complete_candidate_exports and complete_export
        folds.append(
            {
                "fold_index": fold_index,
                "report_path": str(report_path),
                "report_sha256": _sha256(report_path),
                "source_status": source.get("status"),
                "candidates_evaluated": candidates_evaluated,
                "candidates_exported": len(candidates),
                "complete_candidate_export": complete_export,
                "selected_gate": (
                    source.get("selected_gate", {}).get("gate")
                    if isinstance(source.get("selected_gate"), dict)
                    else None
                ),
            }
        )
        for rank, candidate in enumerate(candidates, start=1):
            gate = candidate["gate"]
            key = (str(gate["feature"]), str(gate["direction"]), float(gate["fraction"]))
            grouped.setdefault(key, []).append(
                _candidate_fold_summary(
                    fold_index=fold_index,
                    rank=rank,
                    gate=gate,
                    candidate=candidate,
                    profit_cost_bps=profit_cost_bps,
                    minimum_confirmation_selected_count=(
                        minimum_confirmation_selected_count
                    ),
                )
            )

    required_fold_count = len(report_paths)
    candidate_groups = [
        _candidate_group_summary(
            key=key,
            fold_summaries=fold_summaries,
            required_fold_count=required_fold_count,
            profit_cost_bps=profit_cost_bps,
        )
        for key, fold_summaries in grouped.items()
    ]
    ranked_groups = sorted(
        candidate_groups,
        key=lambda item: (
            item["positive_profit_cost_fold_count"],
            item["positive_half_bps_fold_count"],
            item["valid_confirmation_fold_count"],
            item["minimum_confirmation_net_at_profit_cost_bps"],
            item["average_confirmation_net_at_profit_cost_bps"],
            -item["average_rank"],
        ),
        reverse=True,
    )
    selected = next(
        (
            item
            for item in ranked_groups
            if item["fold_count"] == required_fold_count
            and item["valid_confirmation_fold_count"] == required_fold_count
            and item["positive_profit_cost_fold_count"] == required_fold_count
        ),
        None,
    )
    status_reasons: list[str] = []
    if not complete_candidate_exports:
        status = "INCOMPLETE_CANDIDATE_EXPORT"
        status_reasons.append("source_reports_do_not_export_all_candidates")
        selected = None
    elif selected:
        status = "ROBUST_GATE_CANDIDATE_FOR_FUTURE_HOLDOUT"
        status_reasons.append("future_holdout_required_before_strategy_approval")
    else:
        status = "NO_ROBUST_GATE_CANDIDATE"
        status_reasons.append("no_candidate_positive_in_every_confirmation_fold")

    return _write_report(
        output_root,
        {
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
            "approved_strategy": False,
            "approval_reasons": [
                "diagnostic_only",
                "candidate_selected_from_prior_walkforward_folds",
                "future_holdout_required",
                "no_statistical_gate",
                "no_order_execution",
            ],
            "experiment_stage": "book_regime_gate_walkforward_matrix",
            "created_at": datetime.now(UTC).isoformat(),
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": bool(selected),
            "profit_cost_bps": profit_cost_bps,
            "minimum_confirmation_selected_count": (
                minimum_confirmation_selected_count
            ),
            "required_fold_count": required_fold_count,
            "folds": folds,
            "complete_candidate_exports": complete_candidate_exports,
            "groups_evaluated": len(candidate_groups),
            "selected_robust_candidate": selected,
            "top_candidate_groups": ranked_groups[:max_report_candidates],
        },
    )


def _candidate_fold_summary(
    *,
    fold_index: int,
    rank: int,
    gate: dict[str, object],
    candidate: dict[str, object],
    profit_cost_bps: float,
    minimum_confirmation_selected_count: int,
) -> dict[str, object]:
    development = candidate["gate_development_metrics"]
    confirmation = candidate["confirmation_metrics"]
    confirmation_count = int(confirmation["selected_count"])
    confirmation_net_at_profit_cost = _net_at_cost(confirmation, profit_cost_bps)
    confirmation_net_at_half_bps = _net_at_cost(confirmation, 0.5)
    return {
        "fold_index": fold_index,
        "rank": rank,
        "threshold": float(gate["threshold"]),
        "development_selected_count": int(development["selected_count"]),
        "development_gross_path_payoff_mean_bps": float(
            development["selected_gross_path_payoff_mean_bps"]
        ),
        "confirmation_selected_count": confirmation_count,
        "confirmation_gross_path_payoff_mean_bps": float(
            confirmation["selected_gross_path_payoff_mean_bps"]
        ),
        "confirmation_net_at_profit_cost_bps": confirmation_net_at_profit_cost,
        "confirmation_net_at_half_bps": confirmation_net_at_half_bps,
        "valid_confirmation_count": (
            confirmation_count >= minimum_confirmation_selected_count
        ),
        "positive_at_profit_cost": confirmation_net_at_profit_cost > 0,
        "positive_at_half_bps": confirmation_net_at_half_bps > 0,
    }


def _candidate_group_summary(
    *,
    key: tuple[str, str, float],
    fold_summaries: list[dict[str, object]],
    required_fold_count: int,
    profit_cost_bps: float,
) -> dict[str, object]:
    feature, direction, fraction = key
    confirmation_counts = [
        int(item["confirmation_selected_count"]) for item in fold_summaries
    ]
    gross_values = [
        float(item["confirmation_gross_path_payoff_mean_bps"])
        for item in fold_summaries
    ]
    profit_cost_values = [
        float(item["confirmation_net_at_profit_cost_bps"])
        for item in fold_summaries
    ]
    ranks = [int(item["rank"]) for item in fold_summaries]
    valid_count = sum(bool(item["valid_confirmation_count"]) for item in fold_summaries)
    positive_profit_count = sum(
        bool(item["positive_at_profit_cost"]) for item in fold_summaries
    )
    positive_half_count = sum(
        bool(item["positive_at_half_bps"]) for item in fold_summaries
    )
    return {
        "feature": feature,
        "direction": direction,
        "fraction": fraction,
        "fold_count": len(fold_summaries),
        "required_fold_count": required_fold_count,
        "valid_confirmation_fold_count": valid_count,
        "positive_profit_cost_fold_count": positive_profit_count,
        "positive_half_bps_fold_count": positive_half_count,
        "profit_cost_bps": profit_cost_bps,
        "minimum_confirmation_selected_count": min(confirmation_counts),
        "average_confirmation_selected_count": float(
            sum(confirmation_counts) / len(confirmation_counts)
        ),
        "minimum_confirmation_gross_path_payoff_mean_bps": min(gross_values),
        "average_confirmation_gross_path_payoff_mean_bps": float(
            sum(gross_values) / len(gross_values)
        ),
        "minimum_confirmation_net_at_profit_cost_bps": min(profit_cost_values),
        "average_confirmation_net_at_profit_cost_bps": float(
            sum(profit_cost_values) / len(profit_cost_values)
        ),
        "average_rank": float(sum(ranks) / len(ranks)),
        "folds_missing_candidate": required_fold_count - len(fold_summaries),
        "folds": sorted(fold_summaries, key=lambda item: int(item["fold_index"])),
    }


def _net_at_cost(metrics: dict[str, Any], cost_bps: float) -> float:
    cost_sensitivity = metrics.get("cost_sensitivity", [])
    for item in cost_sensitivity:
        if float(item["total_cost_bps"]) == cost_bps:
            return float(item["selected_net_mean_bps"])
    selected_gross = float(metrics["selected_gross_path_payoff_mean_bps"])
    return selected_gross - cost_bps


def _parse_paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(part.strip()) for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("path list cannot be empty")
    return paths


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "approved_strategy": report["approved_strategy"],
        "selected_robust_candidate": report.get("selected_robust_candidate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

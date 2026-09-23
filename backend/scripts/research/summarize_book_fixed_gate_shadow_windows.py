"""Summarize fixed-gate shadow settlements across prospective windows.

This is an evidence aggregation report, not a promotion step. It accepts
research-only settlement reports emitted by settle_book_fixed_gate_shadow_decisions.py
or by run_book_fixed_gate_prospective_window.py children, validates their safety
boundary, and computes window-level and weighted execution metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_book_rule_confirmation import _parse_float_list  # noqa: E402


@dataclass(frozen=True)
class SettlementReportRef:
    label: str
    path: Path


@dataclass(frozen=True)
class WindowModeSummary:
    label: str
    latency_ms: int
    mode: str
    selected_signal_count: int
    executable_count: int
    skipped_count: int
    gross_mean_bps: float
    gross_total_bps: float
    maximum_drawdown_bps: float
    exit_reasons: dict[str, int]
    net_by_cost_bps: dict[str, float]
    net_total_by_cost_bps: dict[str, float]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settlement-reports",
        required=True,
        help="Comma-separated LABEL=path report list.",
    )
    parser.add_argument("--minimum-windows", type=int, default=3)
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-fixed-gate-shadow-window-summary",
    )
    arguments = parser.parse_args()
    report = summarize_book_fixed_gate_shadow_windows(
        settlement_report_refs=_parse_report_refs(arguments.settlement_reports),
        output_root=arguments.output_root,
        minimum_windows=arguments.minimum_windows,
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def summarize_book_fixed_gate_shadow_windows(
    *,
    settlement_report_refs: tuple[SettlementReportRef, ...],
    output_root: Path,
    minimum_windows: int = 3,
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, object]:
    if not settlement_report_refs:
        raise ValueError("at least one settlement report is required")
    if minimum_windows < 1:
        raise ValueError("minimum_windows must be positive")
    if profit_cost_bps < 0 or stress_cost_bps < 0:
        raise ValueError("cost bps thresholds cannot be negative")
    if profit_cost_bps not in cost_levels_bps:
        raise ValueError("profit_cost_bps must be present in cost_levels_bps")
    if stress_cost_bps not in cost_levels_bps:
        raise ValueError("stress_cost_bps must be present in cost_levels_bps")

    window_summaries: list[WindowModeSummary] = []
    source_reports: list[dict[str, object]] = []
    seen_labels: set[str] = set()
    for ref in settlement_report_refs:
        if ref.label in seen_labels:
            raise ValueError(f"duplicate settlement report label: {ref.label}")
        seen_labels.add(ref.label)
        payload = json.loads(ref.path.read_text(encoding="utf-8"))
        _validate_settlement_boundary(payload, ref)
        source_reports.append(
            {
                "label": ref.label,
                "path": str(ref.path),
                "sha256": _sha256_file(ref.path),
                "status": payload["status"],
                "status_reasons": payload.get("status_reasons", []),
                "shadow_selected_count": payload.get("shadow_selected_count"),
            }
        )
        latency_ms = int(payload["execution_replay"]["latency_ms"])
        window_summaries.append(
            _window_mode_summary(
                label=ref.label,
                latency_ms=latency_ms,
                mode="independent",
                summary=payload["independent_summary"],
                cost_levels_bps=cost_levels_bps,
            )
        )
        window_summaries.append(
            _window_mode_summary(
                label=ref.label,
                latency_ms=latency_ms,
                mode="one_position",
                summary=payload["one_position_summary"],
                cost_levels_bps=cost_levels_bps,
            )
        )

    modes = sorted({summary.mode for summary in window_summaries})
    aggregate_by_mode = {
        mode: _aggregate_mode(
            [summary for summary in window_summaries if summary.mode == mode],
            cost_levels_bps=cost_levels_bps,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
        )
        for mode in modes
    }
    status, reasons = _status(
        window_count=len(settlement_report_refs),
        minimum_windows=minimum_windows,
        aggregate_by_mode=aggregate_by_mode,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    return _write_report(
        output_root,
        {
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
            "committed": False,
            "dry_run": True,
            "approved_strategy": False,
            "experiment_stage": "book_fixed_gate_shadow_window_summary",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "minimum_windows": minimum_windows,
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "cost_levels_bps": list(cost_levels_bps),
            "window_count": len(settlement_report_refs),
            "source_reports": source_reports,
            "windows": [asdict(summary) for summary in window_summaries],
            "aggregate_by_mode": aggregate_by_mode,
            "status": status,
            "status_reasons": reasons,
        },
    )


def _window_mode_summary(
    *,
    label: str,
    latency_ms: int,
    mode: str,
    summary: dict[str, Any],
    cost_levels_bps: tuple[float, ...],
) -> WindowModeSummary:
    net_by_cost: dict[str, float] = {}
    net_total_by_cost: dict[str, float] = {}
    cost_rows = summary["cost_sensitivity"]
    for cost_bps in cost_levels_bps:
        row = _cost_row(cost_rows, cost_bps)
        key = _cost_key(cost_bps)
        net_by_cost[key] = float(row["net_executable_mean_bps"])
        net_total_by_cost[key] = float(row["net_executable_total_bps"])
    return WindowModeSummary(
        label=label,
        latency_ms=latency_ms,
        mode=mode,
        selected_signal_count=int(summary["selected_signal_count"]),
        executable_count=int(summary["executable_count"]),
        skipped_count=int(summary["skipped_count"]),
        gross_mean_bps=float(summary["gross_executable_mean_bps"]),
        gross_total_bps=float(summary["gross_executable_total_bps"]),
        maximum_drawdown_bps=float(summary["maximum_drawdown_bps"]),
        exit_reasons={str(key): int(value) for key, value in summary["exit_reasons"].items()},
        net_by_cost_bps=net_by_cost,
        net_total_by_cost_bps=net_total_by_cost,
    )


def _aggregate_mode(
    summaries: list[WindowModeSummary],
    *,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, object]:
    executable_count = sum(summary.executable_count for summary in summaries)
    if executable_count <= 0:
        raise ValueError(f"mode {summaries[0].mode if summaries else '<empty>'} has no executable trades")
    selected_signal_count = sum(summary.selected_signal_count for summary in summaries)
    net_mean_by_cost = {}
    net_total_by_cost = {}
    positive_window_count_by_cost = {}
    min_window_net_by_cost = {}
    for cost_bps in cost_levels_bps:
        key = _cost_key(cost_bps)
        net_total = sum(summary.net_total_by_cost_bps[key] for summary in summaries)
        net_total_by_cost[key] = net_total
        net_mean_by_cost[key] = net_total / executable_count
        positive_window_count_by_cost[key] = sum(
            1 for summary in summaries if summary.net_by_cost_bps[key] > 0
        )
        min_window_net_by_cost[key] = min(summary.net_by_cost_bps[key] for summary in summaries)
    gross_total = sum(summary.gross_total_bps for summary in summaries)
    exit_reasons: dict[str, int] = {}
    for summary in summaries:
        for reason, count in summary.exit_reasons.items():
            exit_reasons[reason] = exit_reasons.get(reason, 0) + count
    return {
        "window_count": len(summaries),
        "selected_signal_count": selected_signal_count,
        "executable_count": executable_count,
        "gross_mean_bps": gross_total / executable_count,
        "gross_total_bps": gross_total,
        "net_mean_by_cost_bps": net_mean_by_cost,
        "net_total_by_cost_bps": net_total_by_cost,
        "positive_window_count_by_cost_bps": positive_window_count_by_cost,
        "min_window_net_by_cost_bps": min_window_net_by_cost,
        "exit_reasons": exit_reasons,
        "max_window_drawdown_bps": max(summary.maximum_drawdown_bps for summary in summaries),
        "all_windows_positive_at_profit_cost": (
            positive_window_count_by_cost[_cost_key(profit_cost_bps)] == len(summaries)
        ),
        "all_windows_positive_at_stress_cost": (
            positive_window_count_by_cost[_cost_key(stress_cost_bps)] == len(summaries)
        ),
    }


def _status(
    *,
    window_count: int,
    minimum_windows: int,
    aggregate_by_mode: dict[str, dict[str, object]],
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if window_count < minimum_windows:
        reasons.append(f"window_count_below_minimum:{window_count}<{minimum_windows}")
    for mode, aggregate in aggregate_by_mode.items():
        if not aggregate["all_windows_positive_at_profit_cost"]:
            reasons.append(f"{mode}_has_non_positive_window_at_{profit_cost_bps:g}bps")
        if not aggregate["all_windows_positive_at_stress_cost"]:
            reasons.append(f"{mode}_has_non_positive_window_at_{stress_cost_bps:g}bps")
    if any(reason.endswith(f"at_{profit_cost_bps:g}bps") for reason in reasons):
        return "SHADOW_WINDOW_SUMMARY_FAILED_PROFIT_COST_REPLICATION", reasons
    if window_count < minimum_windows:
        return "SHADOW_WINDOW_SUMMARY_MORE_WINDOWS_REQUIRED", reasons
    if any(reason.endswith(f"at_{stress_cost_bps:g}bps") for reason in reasons):
        return "SHADOW_WINDOW_SUMMARY_REPLICATED_AT_PROFIT_COST_STRESS_COST_FRAGILE", reasons
    return "SHADOW_WINDOW_SUMMARY_REPLICATED_RESEARCH_ONLY", reasons


def _validate_settlement_boundary(
    report: dict[str, Any],
    ref: SettlementReportRef,
) -> None:
    if report.get("research_only") is not True:
        raise ValueError(f"{ref.label} settlement report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError(f"{ref.label} settlement report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError(f"{ref.label} settlement report must have execution_authorization=none")
    if report.get("committed") is not False or report.get("dry_run") is not True:
        raise ValueError(f"{ref.label} settlement report must be dry-run only")
    if report.get("approved_strategy") is not False:
        raise ValueError(f"{ref.label} settlement report must not approve a strategy")
    if "independent_summary" not in report or "one_position_summary" not in report:
        raise ValueError(f"{ref.label} settlement report is missing mode summaries")
    if "execution_replay" not in report or "latency_ms" not in report["execution_replay"]:
        raise ValueError(f"{ref.label} settlement report is missing execution_replay.latency_ms")


def _cost_row(rows: list[dict[str, Any]], cost_bps: float) -> dict[str, Any]:
    for row in rows:
        if float(row["explicit_cost_bps"]) == cost_bps:
            return row
    raise ValueError(f"cost level missing from settlement report: {cost_bps:g} bps")


def _parse_report_refs(value: str) -> tuple[SettlementReportRef, ...]:
    refs = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("settlement report refs must use LABEL=path")
        label, raw_path = part.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("settlement report label cannot be empty")
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"settlement report not found: {path}")
        refs.append(SettlementReportRef(label=label, path=path))
    if not refs:
        raise ValueError("at least one settlement report ref is required")
    return tuple(refs)


def _cost_key(cost_bps: float) -> str:
    return f"{cost_bps:g}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "approved_strategy": report["approved_strategy"],
        "window_count": report["window_count"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "aggregate_by_mode": report["aggregate_by_mode"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

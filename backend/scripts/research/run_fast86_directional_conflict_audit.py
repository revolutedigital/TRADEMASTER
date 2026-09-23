"""Fast86 directional conflict audit for Fast80 historical outcomes.

Fast86 checks whether simple post-filters could rescue the failed Fast85
historical Fast80 matches: side-only filters, conflict veto, and highest
probability per timestamp. This is diagnostic only and cannot approve a
strategy because it is selected after observing historical outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast85-fast80-historical-outcome-audit"
    / "independent_settlements.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast86-directional-conflict-audit"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    report = run_fast86_directional_conflict_audit(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast86_directional_conflict_audit(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)
    conflict_keys = {
        key
        for key, group in group_by_decision_timestamp(rows).items()
        if len({str(row["side"]) for row in group}) > 1
    }
    variants = {
        "all_independent": rows,
        "buy_only": [row for row in rows if row.get("side") == "BUY"],
        "sell_only": [row for row in rows if row.get("side") == "SELL"],
        "conflict_rows": [
            row for row in rows if decision_timestamp_key(row) in conflict_keys
        ],
        "non_conflict_rows": [
            row for row in rows if decision_timestamp_key(row) not in conflict_keys
        ],
        "top_probability_per_timestamp": top_probability_per_timestamp(rows),
        "conflict_veto_top_probability": top_probability_per_timestamp(
            [row for row in rows if decision_timestamp_key(row) not in conflict_keys]
        ),
    }
    variant_summaries = {
        name: summarize_rows(variant_rows)
        for name, variant_rows in variants.items()
    }
    variants_path = output_root / "variant_rows.jsonl"
    write_jsonl(
        variants_path,
        [
            {"variant": name, **row}
            for name, variant_rows in variants.items()
            for row in variant_rows
        ],
    )
    status, status_reasons = fast86_status(variant_summaries)
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
            "fast86_is_post_outcome_diagnostic_only",
            "filters_are_not_preregistered",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast86_directional_conflict_audit",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "row_count": len(rows),
        "decision_timestamp_count": len(group_by_decision_timestamp(rows)),
        "conflict_timestamp_count": len(conflict_keys),
        "variant_summaries": variant_summaries,
        "variant_rows_path": str(variants_path),
        "variant_rows_sha256": _sha256_file(variants_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def decision_timestamp_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["label"]), int(row["decision_time_ms"]))


def group_by_decision_timestamp(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(decision_timestamp_key(row), []).append(row)
    return grouped


def top_probability_per_timestamp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for _key, group in sorted(group_by_decision_timestamp(rows).items()):
        selected.append(
            sorted(
                group,
                key=lambda row: (-float(row["probability"]), str(row["side"])),
            )[0]
        )
    return selected


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_values = [float(row["expected_net_bps"]) for row in rows]
    stress_values = [float(row["stress_net_bps"]) for row in rows]
    return {
        "trade_count": len(rows),
        "expected_total_bps": sum(expected_values),
        "stress_total_bps": sum(stress_values),
        "expected_mean_bps": safe_mean(expected_values),
        "stress_mean_bps": safe_mean(stress_values),
        "expected_win_rate": safe_mean([1.0 if value > 0 else 0.0 for value in expected_values]),
        "stress_win_rate": safe_mean([1.0 if value > 0 else 0.0 for value in stress_values]),
        "initial_stop_count": sum(row.get("exit_reason") == "INITIAL_STOP" for row in rows),
        "breakeven_stop_count": sum(row.get("exit_reason") == "BREAKEVEN_STOP" for row in rows),
        "timeout_count": sum(row.get("exit_reason") == "TIMEOUT" for row in rows),
    }


def fast86_status(variant_summaries: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "post_outcome_filters_not_deployable",
    ]
    positive_variants = [
        name
        for name, summary in variant_summaries.items()
        if int(summary["trade_count"]) > 0 and float(summary["stress_total_bps"]) > 0
    ]
    if not positive_variants:
        return "FAST86_DIRECTIONAL_CONFLICT_FILTERS_FAILED", reasons + [
            "no_variant_has_positive_stress_total"
        ]
    return "FAST86_DIRECTIONAL_CONFLICT_FILTER_HAS_POSITIVE_DIAGNOSTIC_ONLY", reasons + [
        f"positive_variants:{','.join(positive_variants)}",
        "requires_preregistration_before_reuse",
    ]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "row_count": report["row_count"],
        "conflict_timestamp_count": report["conflict_timestamp_count"],
        "variant_summaries": report["variant_summaries"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

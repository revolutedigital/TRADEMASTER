"""Fast85 historical outcome audit for Fast80-like local regime matches.

Fast85 is not a strategy gate. It settles the Fast80 rule matches found by
Fast84 in all local WALs to answer a narrow diagnostic question: when this
rare vol+flow regime appears, did the frozen trailing replay make or lose money?

The input may include records before the Fast80 preregistration timestamp, so
the output is historical diagnostics only and cannot approve Testnet or LIVE.
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

from scripts.research.run_fast60_fast57_shadow_dry_run import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    WalRootRef,
    parse_wal_root_refs,
)
from scripts.research.run_fast61_fast60_shadow_settlement import (  # noqa: E402
    DEFAULT_POLICY_NAME,
    _apply_one_position_constraint,
    _policy_by_name,
    _settle_label_decisions,
    _sha256_file,
    _summary_with_labels,
    _write_jsonl,
)
from scripts.research.run_fast80_rare_vol_flow_observation import (  # noqa: E402
    PREREGISTRATION_TIMESTAMP,
)


DEFAULT_INPUT_PATH = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "fast84-top-p-silence-diagnostic-all-local-wals"
    / "all_scored_decisions.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT / "reports" / "fast85-fast80-historical-outcome-audit"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--wal-roots",
        nargs="+",
        help="Optional LABEL=path WAL roots. Defaults to backend/data/microstructure_v1/<label>.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--expected-product", default="usdm_perpetual")
    arguments = parser.parse_args()
    report = run_fast85_fast80_historical_outcome_audit(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
        data_root=arguments.data_root,
        wal_roots=(
            parse_wal_root_refs(arguments.wal_roots)
            if arguments.wal_roots
            else ()
        ),
        policy_name=arguments.policy_name,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        expected_product=arguments.expected_product,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast85_fast80_historical_outcome_audit(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    wal_roots: tuple[WalRootRef, ...] = (),
    policy_name: str = DEFAULT_POLICY_NAME,
    latency_ms: int = 100,
    expected_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    expected_product: str = "usdm_perpetual",
) -> dict[str, Any]:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if min(expected_cost_bps, stress_cost_bps) < 0:
        raise ValueError("costs cannot be negative")
    if stress_cost_bps < expected_cost_bps:
        raise ValueError("stress cost cannot be below expected cost")
    output_root.mkdir(parents=True, exist_ok=True)
    policy = _policy_by_name(policy_name)
    override_roots = {ref.label: ref.path for ref in wal_roots}
    scored_rows = load_jsonl(input_path)
    candidates = sorted(
        [
            normalize_candidate(row)
            for row in scored_rows
            if is_fast80_top_p_rule_match(row)
        ],
        key=lambda row: (
            int(row["decision_time_ms"]),
            -float(row["probability"]),
            str(row["side"]),
        ),
    )
    candidate_path = output_root / "fast80_top_p_candidates.jsonl"
    _write_jsonl(candidate_path, candidates)

    independent_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    missing_roots: list[dict[str, str]] = []
    for label, rows in group_by_label(candidates).items():
        wal_root = override_roots.get(label, data_root / label)
        if not wal_root.exists():
            missing_roots.append({"label": label, "wal_root": str(wal_root)})
            incomplete_rows.extend(
                {
                    **minimal_incomplete_row(row),
                    "incomplete_reason": "wal_root_not_found",
                    "wal_root": str(wal_root),
                }
                for row in rows
            )
            continue
        independent, incomplete = _settle_label_decisions(
            rows,
            wal_root=wal_root,
            policy=policy,
            latency_ms=latency_ms,
            expected_cost_bps=expected_cost_bps,
            stress_cost_bps=stress_cost_bps,
            expected_product=expected_product,
        )
        independent_rows.extend(enrich_settlement_rows(independent))
        incomplete_rows.extend(enrich_settlement_rows(incomplete))

    independent_rows = sorted(
        independent_rows,
        key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"])),
    )
    one_position_rows = _apply_one_position_constraint(independent_rows)
    independent_path = output_root / "independent_settlements.jsonl"
    one_position_path = output_root / "one_position_settlements.jsonl"
    incomplete_path = output_root / "incomplete_signals.jsonl"
    _write_jsonl(independent_path, independent_rows)
    _write_jsonl(one_position_path, one_position_rows)
    _write_jsonl(incomplete_path, incomplete_rows)

    independent_summary = _summary_with_labels(independent_rows)
    one_position_summary = _summary_with_labels(one_position_rows)
    aggregate = {
        "input_scored_rows": len(scored_rows),
        "candidate_count": len(candidates),
        "candidate_label_count": len({str(row["label"]) for row in candidates}),
        "complete_outcome_count": len(independent_rows),
        "incomplete_outcome_count": len(incomplete_rows),
        "one_position_outcome_count": len(one_position_rows),
        "missing_wal_root_count": len(missing_roots),
        "pre_preregistration_candidate_count": sum(
            int(row["decision_time_ms"]) <= preregistration_timestamp_ms()
            for row in candidates
        ),
        "post_preregistration_candidate_count": sum(
            int(row["decision_time_ms"]) > preregistration_timestamp_ms()
            for row in candidates
        ),
    }
    status, status_reasons = fast85_status(
        aggregate=aggregate,
        independent_summary=independent_summary,
        one_position_summary=one_position_summary,
    )
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
            "fast85_is_historical_diagnostic_only",
            "input_includes_pre_preregistration_rows",
            "no_order_execution",
            "no_ledger_commit",
            "prospective_fast80_gate_remains_authoritative",
        ],
        "experiment_stage": "fast85_fast80_historical_outcome_audit",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256_file(candidate_path),
        "independent_settlements_path": str(independent_path),
        "independent_settlements_sha256": _sha256_file(independent_path),
        "one_position_settlements_path": str(one_position_path),
        "one_position_settlements_sha256": _sha256_file(one_position_path),
        "incomplete_signals_path": str(incomplete_path),
        "incomplete_signals_sha256": _sha256_file(incomplete_path),
        "policy": {
            "name": policy.name,
            "initial_stop_bps": policy.initial_stop_bps,
            "breakeven_activation_bps": policy.breakeven_activation_bps,
            "breakeven_lock_bps": policy.breakeven_lock_bps,
            "trailing_activation_bps": policy.trailing_activation_bps,
            "trailing_distance_bps": policy.trailing_distance_bps,
        },
        "latency_ms": latency_ms,
        "expected_cost_bps": expected_cost_bps,
        "stress_cost_bps": stress_cost_bps,
        "aggregate": aggregate,
        "independent_summary": independent_summary,
        "one_position_summary": one_position_summary,
        "missing_wal_roots": missing_roots,
        "fast80_preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def is_fast80_top_p_rule_match(row: dict[str, Any]) -> bool:
    return bool(row.get("would_enter")) and bool(row.get("fast80_rule_match_diagnostic"))


def normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "fast85_historical_diagnostic_only": True,
        "approved_strategy": False,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def enrich_settlement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "fast85_historical_diagnostic_only": True,
            "approved_strategy": False,
            "order_submission_allowed": False,
            "execution_authorization": "none",
        }
        for row in rows
    ]


def minimal_incomplete_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("label"),
        "decision_time": row.get("decision_time"),
        "decision_time_ms": row.get("decision_time_ms"),
        "side": row.get("side"),
        "horizon_seconds": row.get("horizon_seconds"),
        "probability": row.get("probability"),
        "threshold": row.get("threshold"),
        "model_sha256": row.get("model_sha256"),
        "feature_vector_sha256": row.get("feature_vector_sha256"),
        "settlement_complete": False,
    }


def group_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    return grouped


def fast85_status(
    *,
    aggregate: dict[str, int],
    independent_summary: dict[str, float | int],
    one_position_summary: dict[str, float | int],
) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "historical_pre_preregistration_diagnostic_only",
    ]
    if int(aggregate["candidate_count"]) <= 0:
        return "FAST85_NO_FAST80_TOP_P_CANDIDATES", reasons + ["no_rule_match_candidates"]
    if int(aggregate["complete_outcome_count"]) <= 0:
        return "FAST85_NO_COMPLETE_OUTCOMES", reasons + ["no_candidate_had_complete_replay"]
    independent_stress = float(independent_summary["stress_total_log_return_bps"])
    one_position_stress = float(one_position_summary["stress_total_log_return_bps"])
    if independent_stress <= 0 or one_position_stress <= 0:
        return "FAST85_FAST80_HISTORICAL_OUTCOME_FAILED", reasons + [
            "stress_total_non_positive_after_trailing_replay"
        ]
    return "FAST85_FAST80_HISTORICAL_OUTCOME_POSITIVE_DIAGNOSTIC_ONLY", reasons + [
        "prospective_confirmation_still_required"
    ]


def preregistration_timestamp_ms() -> int:
    return int(PREREGISTRATION_TIMESTAMP.timestamp() * 1000)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "aggregate": report["aggregate"],
        "independent_summary": report["independent_summary"],
        "one_position_summary": report["one_position_summary"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

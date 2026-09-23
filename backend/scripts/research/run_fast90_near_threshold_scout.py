"""Fast90 near-threshold entry scout.

Fast89 showed the frozen top-p threshold can stay silent in a live WAL window.
Fast90 asks the next diagnostic question: would near-threshold decisions have
paid after the same trailing replay, or is the entry generator itself weak?

This is explicitly post-outcome research. It may generate hypotheses for a new
preregistration, but it does not approve Testnet or LIVE trading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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
    _write_jsonl,
)
from scripts.research.run_fast80_rare_vol_flow_observation import (  # noqa: E402
    HORIZON_SECONDS,
    MODEL_SHA256,
)
from scripts.research.run_fast87_directional_filter_scout import (  # noqa: E402
    apply_conflict_policy,
    feature,
    group_by_label,
    summarize_variant_rows,
)


DEFAULT_INPUT_PATH = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "fast84-top-p-silence-diagnostic-all-local-wals"
    / "all_scored_decisions.jsonl"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast90-near-threshold-scout"
DEFAULT_PROBABILITY_MARGINS = (0.02, 0.05)
MINIMUM_DIAGNOSTIC_TRADES = 10
MINIMUM_DIAGNOSTIC_LABELS = 2


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    probability_margin: float
    conflict_policy: str
    predicate_name: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--wal-roots",
        nargs="+",
        help="Optional LABEL=path overrides. Defaults to data-root/<label>.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--expected-product", default="usdm_perpetual")
    parser.add_argument(
        "--probability-margins",
        default=",".join(f"{margin:g}" for margin in DEFAULT_PROBABILITY_MARGINS),
        help="Comma-separated threshold relaxation margins, e.g. 0.02,0.05.",
    )
    arguments = parser.parse_args()
    report = run_fast90_near_threshold_scout(
        input_path=arguments.input_path,
        data_root=arguments.data_root,
        wal_roots=(
            parse_wal_root_refs(arguments.wal_roots)
            if arguments.wal_roots
            else ()
        ),
        output_root=arguments.output_root,
        policy_name=arguments.policy_name,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        expected_product=arguments.expected_product,
        probability_margins=parse_probability_margins(arguments.probability_margins),
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast90_near_threshold_scout(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    wal_roots: tuple[WalRootRef, ...] = (),
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    policy_name: str = DEFAULT_POLICY_NAME,
    latency_ms: int = 100,
    expected_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    expected_product: str = "usdm_perpetual",
    probability_margins: tuple[float, ...] = DEFAULT_PROBABILITY_MARGINS,
) -> dict[str, Any]:
    validate_parameters(
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
        probability_margins=probability_margins,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    policy = _policy_by_name(policy_name)
    scored_rows = load_jsonl(input_path)
    max_margin = max(probability_margins)
    candidate_universe = [
        normalize_candidate(row)
        for row in scored_rows
        if is_model_horizon_row(row) and passes_probability_margin(row, max_margin)
    ]
    override_roots = {ref.label: ref.path for ref in wal_roots}
    independent_rows, incomplete_rows, missing_roots = settle_candidate_universe(
        candidate_universe,
        data_root=data_root,
        override_roots=override_roots,
        policy=policy,
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
        expected_product=expected_product,
    )

    candidate_path = output_root / "candidate_universe.jsonl"
    independent_path = output_root / "independent_settlements.jsonl"
    incomplete_path = output_root / "incomplete_signals.jsonl"
    _write_jsonl(candidate_path, candidate_universe)
    _write_jsonl(independent_path, independent_rows)
    _write_jsonl(incomplete_path, incomplete_rows)

    variants = evaluate_variants(independent_rows, variant_specs(probability_margins))
    variant_rows_path = output_root / "variant_rows.jsonl"
    _write_jsonl(
        variant_rows_path,
        [
            {"variant": variant["name"], **row}
            for variant in variants
            for row in variant["one_position_rows"]
        ],
    )
    status, status_reasons = fast90_status(
        candidate_universe_count=len(candidate_universe),
        complete_outcome_count=len(independent_rows),
        variants=variants,
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
            "fast90_is_post_outcome_near_threshold_scout",
            "threshold_relaxation_requires_new_preregistration",
            "positive_variants_do_not_approve_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast90_near_threshold_scout",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
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
        "probability_margins": list(probability_margins),
        "scored_row_count": len(scored_rows),
        "candidate_universe_count": len(candidate_universe),
        "complete_outcome_count": len(independent_rows),
        "incomplete_outcome_count": len(incomplete_rows),
        "missing_wal_roots": missing_roots,
        "candidate_universe_path": str(candidate_path),
        "candidate_universe_sha256": _sha256_file(candidate_path),
        "independent_settlements_path": str(independent_path),
        "independent_settlements_sha256": _sha256_file(independent_path),
        "incomplete_signals_path": str(incomplete_path),
        "incomplete_signals_sha256": _sha256_file(incomplete_path),
        "variant_rows_path": str(variant_rows_path),
        "variant_rows_sha256": _sha256_file(variant_rows_path),
        "variant_summaries": [
            {key: value for key, value in variant.items() if key != "one_position_rows"}
            for variant in variants
        ],
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_parameters(
    *,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    probability_margins: tuple[float, ...],
) -> None:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if min(expected_cost_bps, stress_cost_bps) < 0:
        raise ValueError("costs cannot be negative")
    if stress_cost_bps < expected_cost_bps:
        raise ValueError("stress cost cannot be below expected cost")
    if not probability_margins:
        raise ValueError("at least one probability margin is required")
    for margin in probability_margins:
        if margin < 0 or margin > 0.20:
            raise ValueError("probability margins must be between 0 and 0.20")


def parse_probability_margins(raw: str) -> tuple[float, ...]:
    margins = tuple(sorted({float(part.strip()) for part in raw.split(",") if part.strip()}))
    validate_parameters(
        latency_ms=0,
        expected_cost_bps=0,
        stress_cost_bps=0,
        probability_margins=margins,
    )
    return margins


def is_model_horizon_row(row: dict[str, Any]) -> bool:
    return (
        row.get("model_sha256") == MODEL_SHA256
        and int(row.get("horizon_seconds", -1)) == HORIZON_SECONDS
    )


def passes_probability_margin(row: dict[str, Any], margin: float) -> bool:
    return float(row["probability"]) >= probability_floor(row, margin)


def probability_floor(row: dict[str, Any], margin: float) -> float:
    return max(0.0, float(row["threshold"]) - margin)


def normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "fast90_near_threshold_scout_only": True,
        "approved_strategy": False,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def settle_candidate_universe(
    candidates: list[dict[str, Any]],
    *,
    data_root: Path,
    override_roots: dict[str, Path],
    policy: Any,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    expected_product: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    independent_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    missing_roots: list[dict[str, str]] = []
    for label, label_rows in group_by_label(candidates).items():
        wal_root = override_roots.get(label, data_root / label)
        if not wal_root.exists():
            missing_roots.append({"label": label, "wal_root": str(wal_root)})
            incomplete_rows.extend(
                {
                    **row,
                    "settlement_complete": False,
                    "incomplete_reason": "wal_root_not_found",
                    "wal_root": str(wal_root),
                }
                for row in label_rows
            )
            continue
        independent, incomplete = _settle_label_decisions(
            label_rows,
            wal_root=wal_root,
            policy=policy,
            latency_ms=latency_ms,
            expected_cost_bps=expected_cost_bps,
            stress_cost_bps=stress_cost_bps,
            expected_product=expected_product,
        )
        independent_rows.extend(enrich_rows(independent))
        incomplete_rows.extend(enrich_rows(incomplete))
    return (
        sorted(independent_rows, key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"]))),
        sorted(incomplete_rows, key=lambda row: int(row.get("decision_time_ms") or 0)),
        missing_roots,
    )


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "fast90_near_threshold_scout_only": True,
            "approved_strategy": False,
            "order_submission_allowed": False,
            "execution_authorization": "none",
        }
        for row in rows
    ]


def variant_specs(probability_margins: tuple[float, ...]) -> tuple[VariantSpec, ...]:
    specs: list[VariantSpec] = []
    predicates = (
        ("all", "No directional filter."),
        ("flow5_positive", "Side-aligned 5s flow imbalance > 0."),
        ("return5_positive", "Side-aligned 5s return > 0."),
        ("return5_and_flow5_positive", "Side-aligned 5s return > 0 and 5s flow > 0."),
        ("vol15_flow5_positive", "realized_vol_30s_bps >= 15 and side-aligned 5s flow > 0."),
    )
    for margin in probability_margins:
        margin_name = probability_margin_name(margin)
        for predicate_name, description in predicates:
            for conflict_policy in ("top_probability", "veto"):
                specs.append(
                    VariantSpec(
                        name=f"near{margin_name}_{predicate_name}_{conflict_policy}",
                        description=(
                            f"probability >= threshold - {margin:g}; "
                            f"{description}; conflict={conflict_policy}."
                        ),
                        probability_margin=margin,
                        conflict_policy=conflict_policy,
                        predicate_name=predicate_name,
                    )
                )
    return tuple(specs)


def probability_margin_name(margin: float) -> str:
    return f"{int(round(margin * 1000)):03d}"


def evaluate_variants(
    rows: list[dict[str, Any]],
    specs: tuple[VariantSpec, ...],
) -> list[dict[str, Any]]:
    variants = []
    for spec in specs:
        filtered = [
            row
            for row in rows
            if passes_probability_margin(row, spec.probability_margin)
            and passes_predicate(row, spec.predicate_name)
        ]
        filtered = apply_conflict_policy(filtered, spec.conflict_policy)
        one_position_rows = _apply_one_position_constraint(filtered)
        variants.append(
            {
                "name": spec.name,
                "description": spec.description,
                "probability_margin": spec.probability_margin,
                "probability_floor_example": (
                    probability_floor(rows[0], spec.probability_margin) if rows else None
                ),
                "predicate_name": spec.predicate_name,
                "conflict_policy": spec.conflict_policy,
                "independent_summary": summarize_variant_rows(filtered),
                "one_position_summary": summarize_variant_rows(one_position_rows),
                "one_position_rows": one_position_rows,
            }
        )
    return sorted(
        variants,
        key=lambda variant: (
            float(variant["one_position_summary"]["stress_total_bps"]),
            int(variant["one_position_summary"]["trade_count"]),
        ),
        reverse=True,
    )


def passes_predicate(row: dict[str, Any], predicate_name: str) -> bool:
    if predicate_name == "all":
        return True
    if predicate_name == "flow5_positive":
        return feature(row, "directed_flow_imbalance_5s") > 0
    if predicate_name == "return5_positive":
        return feature(row, "directed_return_5s_bps") > 0
    if predicate_name == "return5_and_flow5_positive":
        return (
            feature(row, "directed_return_5s_bps") > 0
            and feature(row, "directed_flow_imbalance_5s") > 0
        )
    if predicate_name == "vol15_flow5_positive":
        return (
            feature(row, "realized_vol_30s_bps") >= 15
            and feature(row, "directed_flow_imbalance_5s") > 0
        )
    raise ValueError(f"unknown predicate: {predicate_name}")


def fast90_status(
    *,
    candidate_universe_count: int,
    complete_outcome_count: int,
    variants: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "post_outcome_threshold_scout_not_deployable",
    ]
    if candidate_universe_count == 0:
        return "FAST90_NO_NEAR_THRESHOLD_CANDIDATES", reasons
    if complete_outcome_count == 0:
        return "FAST90_NO_COMPLETE_NEAR_THRESHOLD_OUTCOMES", reasons
    diagnostic_candidates = [
        variant
        for variant in variants
        if int(variant["one_position_summary"]["trade_count"]) >= MINIMUM_DIAGNOSTIC_TRADES
        and int(variant["one_position_summary"]["label_count"]) >= MINIMUM_DIAGNOSTIC_LABELS
        and float(variant["one_position_summary"]["stress_total_bps"]) > 0
        and float(variant["one_position_summary"]["stress_mean_bps"]) > 0
        and int(variant["one_position_summary"]["negative_label_count"]) == 0
    ]
    if not diagnostic_candidates:
        return "FAST90_NO_NEAR_THRESHOLD_EDGE_FOUND", reasons + [
            "no_variant_positive_with_minimum_sample_and_no_negative_label"
        ]
    return "FAST90_NEAR_THRESHOLD_CANDIDATE_DIAGNOSTIC_ONLY", reasons + [
        f"candidate_count:{len(diagnostic_candidates)}",
        "requires_new_preregistration_before_any_reuse",
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        "scored_row_count": report["scored_row_count"],
        "candidate_universe_count": report["candidate_universe_count"],
        "complete_outcome_count": report["complete_outcome_count"],
        "top_variants": report["variant_summaries"][:10],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

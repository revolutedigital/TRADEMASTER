"""Fast87 directional-filter scout for the next strategy family.

Fast87 takes the lesson from Fast80/Fast86: high-volatility regimes can produce
strong top-p scores, but direction conflicts killed the outcome. This scout
settles all local frozen top-p entries and evaluates predeclared directional
filters (side-aligned return/flow, conflict veto, and top-probability choice).

It is explicitly exploratory and post-outcome. Positive variants only become
ideas for a new preregistration; they do not approve Testnet or LIVE trading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
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
    _summary_with_labels,
    _write_jsonl,
)


DEFAULT_INPUT_PATH = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "fast84-top-p-silence-diagnostic-all-local-wals"
    / "all_scored_decisions.jsonl"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast87-directional-filter-scout"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    predicate: Callable[[dict[str, Any]], bool]
    conflict_policy: str = "allow"


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
    arguments = parser.parse_args()
    report = run_fast87_directional_filter_scout(
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
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast87_directional_filter_scout(
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
) -> dict[str, Any]:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if min(expected_cost_bps, stress_cost_bps) < 0:
        raise ValueError("costs cannot be negative")
    if stress_cost_bps < expected_cost_bps:
        raise ValueError("stress cost cannot be below expected cost")

    output_root.mkdir(parents=True, exist_ok=True)
    policy = _policy_by_name(policy_name)
    scored_rows = load_jsonl(input_path)
    top_p_rows = [
        normalize_candidate(row)
        for row in scored_rows
        if bool(row.get("would_enter"))
    ]
    override_roots = {ref.label: ref.path for ref in wal_roots}
    independent_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    missing_roots: list[dict[str, str]] = []
    for label, label_rows in group_by_label(top_p_rows).items():
        wal_root = override_roots.get(label, data_root / label)
        if not wal_root.exists():
            missing_roots.append({"label": label, "wal_root": str(wal_root)})
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

    independent_rows = sorted(
        independent_rows,
        key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"])),
    )
    top_p_path = output_root / "all_top_p_candidates.jsonl"
    independent_path = output_root / "all_top_p_independent_settlements.jsonl"
    incomplete_path = output_root / "all_top_p_incomplete_signals.jsonl"
    _write_jsonl(top_p_path, top_p_rows)
    _write_jsonl(independent_path, independent_rows)
    _write_jsonl(incomplete_path, incomplete_rows)

    variants = evaluate_variants(independent_rows, variant_specs())
    variant_rows_path = output_root / "variant_rows.jsonl"
    _write_jsonl(
        variant_rows_path,
        [
            {"variant": variant["name"], **row}
            for variant in variants
            for row in variant["one_position_rows"]
        ],
    )
    status, status_reasons = fast87_status(variants)
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
            "fast87_is_post_outcome_directional_scout",
            "positive_variants_require_new_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast87_directional_filter_scout",
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
        "scored_row_count": len(scored_rows),
        "top_p_candidate_count": len(top_p_rows),
        "complete_top_p_outcome_count": len(independent_rows),
        "incomplete_top_p_outcome_count": len(incomplete_rows),
        "missing_wal_roots": missing_roots,
        "all_top_p_candidates_path": str(top_p_path),
        "all_top_p_candidates_sha256": _sha256_file(top_p_path),
        "all_top_p_independent_settlements_path": str(independent_path),
        "all_top_p_independent_settlements_sha256": _sha256_file(independent_path),
        "all_top_p_incomplete_signals_path": str(incomplete_path),
        "all_top_p_incomplete_signals_sha256": _sha256_file(incomplete_path),
        "variant_rows_path": str(variant_rows_path),
        "variant_rows_sha256": _sha256_file(variant_rows_path),
        "variant_summaries": [
            {
                key: value
                for key, value in variant.items()
                if key != "one_position_rows"
            }
            for variant in variants
        ],
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def variant_specs() -> tuple[VariantSpec, ...]:
    specs: list[VariantSpec] = [
        VariantSpec(
            name="baseline_all_top_p",
            description="All settled local top-p entries.",
            predicate=lambda _row: True,
        ),
        VariantSpec(
            name="baseline_top_probability_per_timestamp",
            description="Highest probability side per label/timestamp.",
            predicate=lambda _row: True,
            conflict_policy="top_probability",
        ),
        VariantSpec(
            name="baseline_conflict_veto",
            description="Drop label/timestamps where both BUY and SELL pass.",
            predicate=lambda _row: True,
            conflict_policy="veto",
        ),
    ]
    vol_thresholds = (15.0, 45.0, 90.0)
    directional_rules: tuple[tuple[str, str, Callable[[dict[str, Any]], bool]], ...] = (
        (
            "directed_return_5s_positive",
            "Side-aligned 5s return > 0.",
            lambda row: feature(row, "directed_return_5s_bps") > 0,
        ),
        (
            "directed_return_30s_positive",
            "Side-aligned 30s return > 0.",
            lambda row: feature(row, "directed_return_30s_bps") > 0,
        ),
        (
            "directed_flow_5s_positive",
            "Side-aligned 5s flow imbalance > 0.",
            lambda row: feature(row, "directed_flow_imbalance_5s") > 0,
        ),
        (
            "directed_flow_30s_positive",
            "Side-aligned 30s flow imbalance > 0.",
            lambda row: feature(row, "directed_flow_imbalance_30s") > 0,
        ),
        (
            "return5_and_flow5_positive",
            "Side-aligned 5s return > 0 and 5s flow > 0.",
            lambda row: feature(row, "directed_return_5s_bps") > 0
            and feature(row, "directed_flow_imbalance_5s") > 0,
        ),
        (
            "return30_and_flow30_positive",
            "Side-aligned 30s return > 0 and 30s flow > 0.",
            lambda row: feature(row, "directed_return_30s_bps") > 0
            and feature(row, "directed_flow_imbalance_30s") > 0,
        ),
    )
    for threshold in vol_thresholds:
        specs.append(
            VariantSpec(
                name=f"vol{threshold:g}_all_top_p",
                description=f"realized_vol_30s_bps >= {threshold:g}.",
                predicate=lambda row, threshold=threshold: feature(
                    row,
                    "realized_vol_30s_bps",
                )
                >= threshold,
            )
        )
        for suffix, description, directional_predicate in directional_rules:
            for conflict_policy in ("allow", "top_probability", "veto"):
                policy_suffix = "" if conflict_policy == "allow" else f"_{conflict_policy}"
                specs.append(
                    VariantSpec(
                        name=f"vol{threshold:g}_{suffix}{policy_suffix}",
                        description=f"vol >= {threshold:g}; {description}; conflict={conflict_policy}.",
                        predicate=lambda row, threshold=threshold, directional_predicate=directional_predicate: (
                            feature(row, "realized_vol_30s_bps") >= threshold
                            and directional_predicate(row)
                        ),
                        conflict_policy=conflict_policy,
                    )
                )
    return tuple(specs)


def evaluate_variants(
    rows: list[dict[str, Any]],
    specs: tuple[VariantSpec, ...],
) -> list[dict[str, Any]]:
    variants = []
    for spec in specs:
        filtered = [row for row in rows if spec.predicate(row)]
        filtered = apply_conflict_policy(filtered, spec.conflict_policy)
        one_position_rows = _apply_one_position_constraint(filtered)
        independent_summary = summarize_variant_rows(filtered)
        one_position_summary = summarize_variant_rows(one_position_rows)
        variants.append(
            {
                "name": spec.name,
                "description": spec.description,
                "conflict_policy": spec.conflict_policy,
                "independent_summary": independent_summary,
                "one_position_summary": one_position_summary,
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


def apply_conflict_policy(rows: list[dict[str, Any]], conflict_policy: str) -> list[dict[str, Any]]:
    if conflict_policy == "allow":
        return rows
    grouped = group_by_decision_timestamp(rows)
    selected = []
    for group in grouped.values():
        sides = {str(row["side"]) for row in group}
        if conflict_policy == "veto":
            if len(sides) == 1:
                selected.extend(group)
            continue
        if conflict_policy == "top_probability":
            selected.append(
                sorted(group, key=lambda row: (-float(row["probability"]), str(row["side"])))[0]
            )
            continue
        raise ValueError(f"unknown conflict policy: {conflict_policy}")
    return sorted(selected, key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"])))


def summarize_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = _summary_with_labels(rows)
    label_summaries = label_stress_summaries(rows)
    return {
        "trade_count": len(rows),
        "label_count": len(label_summaries),
        "stress_total_bps": float(base["stress_total_log_return_bps"]),
        "stress_mean_bps": float(base["stress_mean_bps"]),
        "expected_total_bps": float(base["expected_total_log_return_bps"]),
        "expected_mean_bps": float(base["expected_mean_bps"]),
        "expected_win_rate": float(base["expected_win_rate"]),
        "maximum_drawdown_bps": float(base["maximum_drawdown_bps"]),
        "positive_label_count": sum(
            summary["stress_total_bps"] > 0 for summary in label_summaries.values()
        ),
        "negative_label_count": sum(
            summary["stress_total_bps"] <= 0 for summary in label_summaries.values()
        ),
        "worst_label_stress_total_bps": (
            min(summary["stress_total_bps"] for summary in label_summaries.values())
            if label_summaries
            else None
        ),
        "label_summaries": label_summaries,
    }


def label_stress_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped = group_by_label(rows)
    summaries = {}
    for label, label_rows in grouped.items():
        stress_values = [float(row["stress_net_bps"]) for row in label_rows]
        expected_values = [float(row["expected_net_bps"]) for row in label_rows]
        summaries[label] = {
            "trade_count": len(label_rows),
            "stress_total_bps": float(sum(stress_values)),
            "expected_total_bps": float(sum(expected_values)),
        }
    return summaries


def fast87_status(variants: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "post_outcome_scout_not_deployable",
    ]
    positive_variants = [
        variant
        for variant in variants
        if int(variant["one_position_summary"]["trade_count"]) >= 5
        and float(variant["one_position_summary"]["stress_total_bps"]) > 0
        and int(variant["one_position_summary"]["negative_label_count"]) == 0
    ]
    if not positive_variants:
        return "FAST87_NO_ROBUST_DIRECTIONAL_FILTER_FOUND", reasons + [
            "no_variant_has_positive_stress_with_minimum_sample_and_no_negative_label"
        ]
    return "FAST87_DIRECTIONAL_FILTER_CANDIDATE_DIAGNOSTIC_ONLY", reasons + [
        f"candidate_count:{len(positive_variants)}",
        "requires_new_preregistration",
    ]


def normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "fast87_directional_scout_only": True,
        "approved_strategy": False,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "fast87_directional_scout_only": True,
            "approved_strategy": False,
            "order_submission_allowed": False,
            "execution_authorization": "none",
        }
        for row in rows
    ]


def feature(row: dict[str, Any], name: str) -> float:
    if row.get(name) is not None:
        return float(row[name])
    causal = row.get("causal_feature_values", {})
    if isinstance(causal, dict) and causal.get(name) is not None:
        return float(causal[name])
    raise KeyError(name)


def group_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    return grouped


def decision_timestamp_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["label"]), int(row["decision_time_ms"]))


def group_by_decision_timestamp(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(decision_timestamp_key(row), []).append(row)
    return grouped


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
    top_variants = report["variant_summaries"][:10]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "scored_row_count": report["scored_row_count"],
        "top_p_candidate_count": report["top_p_candidate_count"],
        "complete_top_p_outcome_count": report["complete_top_p_outcome_count"],
        "top_variants": top_variants,
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

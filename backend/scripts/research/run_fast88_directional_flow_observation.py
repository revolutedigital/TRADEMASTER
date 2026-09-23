"""Fast88 preregistered directional-flow observation runner.

Fast88 is not a strategy. It is a fail-closed observer for the rule frozen in
docs/forex/fast88-directional-flow-observation-preregistration.md:

top-p entry + realized_vol_30s_bps >= 15 + directed_flow_imbalance_5s > 0,
with same-timestamp BUY/SELL conflict resolved by highest probability.

It settles matches offline with the frozen trailing policy, writes local
research artifacts only, and never sends orders or writes ledger state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

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
    BLOCK_FAIL_MINIMUM_TRADES,
    BLOCK_FAIL_STRESS_TOTAL_BPS,
    EARLY_FAIL_STRESS_TOTAL_BPS,
    EARLY_FAIL_TRADE_COUNT,
    HORIZON_SECONDS,
    MODEL_SHA256,
    SHADOW_MINIMUM_DAYS,
    SHADOW_MINIMUM_EXPECTED_MEAN_BPS,
    SHADOW_MINIMUM_STRESS_MEAN_BPS,
    SHADOW_MINIMUM_TRADES,
    lower_confidence_bound,
)


DEFAULT_INPUT_PATH = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "fast84-top-p-silence-diagnostic-fast82-fast83"
    / "all_scored_decisions.jsonl"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast88-directional-flow-observation"
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast88-directional-flow-observation-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast88-registry.jsonl"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-23T20:18:35Z")
PREREGISTRATION_SHA256 = "be5339212b39646c662dda766bd9e0a8f5eb0bbd7736993831b222ce3f833701"
SOURCE_FAST87_REPORT_SHA256 = "07eae274b31dafd5cb239c0b8ec7f6f2727a56724bbb4a5b79bb3fe339a502f4"
REALIZED_VOL_30S_THRESHOLD = 15.0
DIRECTED_FLOW_IMBALANCE_5S_THRESHOLD = 0.0
REQUIRED_FEATURES = ("realized_vol_30s_bps", "directed_flow_imbalance_5s")
DEFAULT_ALPHA = 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--wal-roots",
        nargs="+",
        help="Optional LABEL=path WAL root overrides. Defaults to data-root/<label>.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--expected-product", default="usdm_perpetual")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args()
    report = run_fast88_directional_flow_observation(
        input_path=arguments.input_path,
        data_root=arguments.data_root,
        wal_roots=(
            parse_wal_root_refs(arguments.wal_roots)
            if arguments.wal_roots
            else ()
        ),
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        policy_name=arguments.policy_name,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        expected_product=arguments.expected_product,
        alpha=arguments.alpha,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast88_directional_flow_observation(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    wal_roots: tuple[WalRootRef, ...] = (),
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    policy_name: str = DEFAULT_POLICY_NAME,
    latency_ms: int = 100,
    expected_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    expected_product: str = "usdm_perpetual",
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_parameters(
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
        alpha=alpha,
    )
    preregistration = validate_preregistration(
        preregistration_path=preregistration_path,
        registry_path=registry_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    policy = _policy_by_name(policy_name)
    override_roots = {ref.label: ref.path for ref in wal_roots}
    input_records = load_jsonl(input_path)
    normalized_records = [normalize_record(record) for record in input_records]
    feature_audit = audit_required_features(normalized_records)

    post_preregistration: list[dict[str, Any]] = []
    candidate_universe: list[dict[str, Any]] = []
    rule_matches: list[dict[str, Any]] = []
    conflict_resolved_candidates: list[dict[str, Any]] = []
    independent_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    one_position_rows: list[dict[str, Any]] = []
    missing_roots: list[dict[str, str]] = []
    if not feature_audit["missing_required_features"]:
        post_preregistration = [
            record
            for record in normalized_records
            if record.get("decision_time_ms") is not None
            and int(record["decision_time_ms"]) > preregistration_timestamp_ms()
        ]
        candidate_universe = [
            record
            for record in post_preregistration
            if record.get("model_sha256") == MODEL_SHA256
            and int(record.get("horizon_seconds", -1)) == HORIZON_SECONDS
            and bool(record.get("would_enter"))
        ]
        rule_matches = [record for record in candidate_universe if passes_fast88_rule(record)]
        conflict_resolved_candidates = resolve_probability_conflicts(rule_matches)
        independent_rows, incomplete_rows, missing_roots = settle_candidates(
            conflict_resolved_candidates,
            data_root=data_root,
            override_roots=override_roots,
            policy=policy,
            latency_ms=latency_ms,
            expected_cost_bps=expected_cost_bps,
            stress_cost_bps=stress_cost_bps,
            expected_product=expected_product,
        )
        one_position_rows = _apply_one_position_constraint(independent_rows)

    candidates_path = output_root / "rule_matches.jsonl"
    conflict_resolved_path = output_root / "conflict_resolved_candidates.jsonl"
    independent_path = output_root / "independent_settlements.jsonl"
    one_position_path = output_root / "one_position_settlements.jsonl"
    incomplete_path = output_root / "incomplete_signals.jsonl"
    observations_path = output_root / "observations.jsonl"
    _write_jsonl(candidates_path, rule_matches)
    _write_jsonl(conflict_resolved_path, conflict_resolved_candidates)
    _write_jsonl(independent_path, independent_rows)
    _write_jsonl(one_position_path, one_position_rows)
    _write_jsonl(incomplete_path, incomplete_rows)
    observations = [observation_record(row) for row in one_position_rows]
    _write_jsonl(observations_path, observations)

    summary = summarize_observations(observations, alpha=alpha)
    gates = evaluate_gates(summary)
    status, status_reasons = fast88_status(
        input_record_count=len(normalized_records),
        feature_audit=feature_audit,
        post_preregistration_record_count=len(post_preregistration),
        candidate_universe_count=len(candidate_universe),
        rule_match_count=len(rule_matches),
        complete_outcome_count=len(independent_rows),
        summary=summary,
        gates=gates,
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
            "fast88_is_preregistered_observation_only",
            "fail_closed_when_required_features_missing",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_preregistration_and_explicit_authorization",
        ],
        "experiment_stage": "fast88_directional_flow_observation",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path) if input_path.exists() else None,
        "input_record_count": len(normalized_records),
        "post_preregistration_record_count": len(post_preregistration),
        "candidate_universe_count": len(candidate_universe),
        "rule_match_count": len(rule_matches),
        "conflict_resolved_candidate_count": len(conflict_resolved_candidates),
        "complete_outcome_count": len(independent_rows),
        "incomplete_outcome_count": len(incomplete_rows),
        "eligible_observation_count": len(observations),
        "observation_rule": {
            "model_sha256": MODEL_SHA256,
            "horizon_seconds": HORIZON_SECONDS,
            "realized_vol_30s_bps_min": REALIZED_VOL_30S_THRESHOLD,
            "directed_flow_imbalance_5s_min_exclusive": DIRECTED_FLOW_IMBALANCE_5S_THRESHOLD,
            "conflict_policy": "highest_probability_per_label_decision_time",
        },
        "required_features": list(REQUIRED_FEATURES),
        "feature_audit": feature_audit,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": _sha256_file(registry_path),
        "source_fast87_report_sha256": SOURCE_FAST87_REPORT_SHA256,
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
        "rule_matches_path": str(candidates_path),
        "rule_matches_sha256": _sha256_file(candidates_path),
        "conflict_resolved_candidates_path": str(conflict_resolved_path),
        "conflict_resolved_candidates_sha256": _sha256_file(conflict_resolved_path),
        "independent_settlements_path": str(independent_path),
        "independent_settlements_sha256": _sha256_file(independent_path),
        "one_position_settlements_path": str(one_position_path),
        "one_position_settlements_sha256": _sha256_file(one_position_path),
        "incomplete_signals_path": str(incomplete_path),
        "incomplete_signals_sha256": _sha256_file(incomplete_path),
        "observations_path": str(observations_path),
        "observations_sha256": _sha256_file(observations_path),
        "missing_wal_roots": missing_roots,
        "summary": summary,
        "gates": gates,
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
    alpha: float,
) -> None:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if min(expected_cost_bps, stress_cost_bps) < 0:
        raise ValueError("costs cannot be negative")
    if stress_cost_bps < expected_cost_bps:
        raise ValueError("stress cost cannot be below expected cost")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> dict[str, str]:
    if not preregistration_path.exists():
        raise FileNotFoundError(preregistration_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    preregistration_sha256 = _sha256_file(preregistration_path)
    if preregistration_sha256 != PREREGISTRATION_SHA256:
        raise ValueError("Fast88 preregistration hash changed")
    registry_events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in registry_events
        if event.get("event") == "directional_flow_observation_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast88 registry must contain exactly one preregistration event")
    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fast87_source_report_sha256": SOURCE_FAST87_REPORT_SHA256,
        "volatility_rule": "realized_vol_30s_bps>=15.0",
        "directional_feature": "directed_flow_imbalance_5s>0.0",
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast88 registry mismatch: {mismatches}")
    return {"preregistration_sha256": preregistration_sha256}


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if normalized.get("decision_time_ms") is None and normalized.get("decision_time") is not None:
        normalized["decision_time_ms"] = int(
            pd.Timestamp(str(normalized["decision_time"])).timestamp() * 1000
        )
    for feature_name in REQUIRED_FEATURES:
        feature_value = feature_value_from_record(record, feature_name)
        if feature_value is not None:
            normalized[feature_name] = feature_value
    return normalized


def feature_value_from_record(record: dict[str, Any], feature_name: str) -> float | None:
    candidate_sources = (
        record,
        record.get("causal_feature_values", {}),
        record.get("feature_values", {}),
        record.get("causal_features", {}),
        record.get("directional_gate_values", {}),
        record.get("gate_values", {}),
    )
    for source in candidate_sources:
        if isinstance(source, dict) and source.get(feature_name) is not None:
            return float(source[feature_name])
    return None


def audit_required_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    missing_counts = {
        feature_name: sum(record.get(feature_name) is None for record in records)
        for feature_name in REQUIRED_FEATURES
    }
    missing_required_features = [
        feature_name
        for feature_name, missing_count in missing_counts.items()
        if missing_count > 0
    ]
    return {
        "record_count": len(records),
        "missing_counts": missing_counts,
        "missing_required_features": missing_required_features,
        "scoreable": not missing_required_features,
    }


def passes_fast88_rule(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("would_enter"))
        and float(record["realized_vol_30s_bps"]) >= REALIZED_VOL_30S_THRESHOLD
        and float(record["directed_flow_imbalance_5s"]) > DIRECTED_FLOW_IMBALANCE_5S_THRESHOLD
    )


def resolve_probability_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (str(record["label"]), int(record["decision_time_ms"])),
            [],
        ).append(record)
    selected = []
    for group in grouped.values():
        selected.append(
            sorted(
                group,
                key=lambda row: (-float(row["probability"]), str(row["side"])),
            )[0]
        )
    return sorted(selected, key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"])))


def settle_candidates(
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
        independent_rows.extend(enrich_rows(independent))
        incomplete_rows.extend(enrich_rows(incomplete))
    return (
        sorted(independent_rows, key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"]))),
        sorted(incomplete_rows, key=lambda row: int(row.get("decision_time_ms") or 0)),
        missing_roots,
    )


def minimal_incomplete_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": record.get("label"),
        "decision_time": record.get("decision_time"),
        "decision_time_ms": record.get("decision_time_ms"),
        "side": record.get("side"),
        "horizon_seconds": record.get("horizon_seconds"),
        "probability": record.get("probability"),
        "threshold": record.get("threshold"),
        "model_sha256": record.get("model_sha256"),
        "feature_vector_sha256": record.get("feature_vector_sha256"),
        "settlement_complete": False,
    }


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "fast88_observation_only": True,
            "approved_strategy": False,
            "order_submission_allowed": False,
            "execution_authorization": "none",
        }
        for row in rows
    ]


def observation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "decision_time_ms": int(record["decision_time_ms"]),
        "decision_time": pd.to_datetime(int(record["decision_time_ms"]), unit="ms", utc=True).isoformat(),
        "entry_time_ms": int(record["entry_time_ms"]),
        "exit_time_ms": int(record["exit_time_ms"]),
        "label": str(record.get("label", "")),
        "side": str(record.get("side", "")),
        "model_sha256": str(record.get("model_sha256", "")),
        "horizon_seconds": int(record.get("horizon_seconds", HORIZON_SECONDS)),
        "probability": float(record.get("probability", 0.0)),
        "realized_vol_30s_bps": float(feature_value_from_record(record, "realized_vol_30s_bps") or 0.0),
        "directed_flow_imbalance_5s": float(feature_value_from_record(record, "directed_flow_imbalance_5s") or 0.0),
        "expected_net_bps": float(record["expected_net_bps"]),
        "stress_net_bps": float(record["stress_net_bps"]),
        "exit_reason": str(record.get("exit_reason", "")),
    }


def summarize_observations(observations: list[dict[str, Any]], *, alpha: float) -> dict[str, Any]:
    if not observations:
        return empty_summary()
    expected = pd.Series([record["expected_net_bps"] for record in observations], dtype="float64")
    stress = pd.Series([record["stress_net_bps"] for record in observations], dtype="float64")
    decision_times = pd.Series([record["decision_time_ms"] for record in observations], dtype="int64")
    block_summaries = block_summaries_from_observations(observations, alpha=alpha)
    sufficient_blocks = [
        block for block in block_summaries if block["trade_count"] >= BLOCK_FAIL_MINIMUM_TRADES
    ]
    negative_sufficient_blocks = [
        block for block in sufficient_blocks if block["stress_total_bps"] <= BLOCK_FAIL_STRESS_TOTAL_BPS
    ]
    return {
        "trades": len(observations),
        "distinct_days": int(pd.to_datetime(decision_times, unit="ms", utc=True).dt.date.nunique()),
        "labels": sorted({str(record.get("label", "")) for record in observations}),
        "expected_mean_bps": float(expected.mean()),
        "stress_mean_bps": float(stress.mean()),
        "expected_total_bps": float(expected.sum()),
        "stress_total_bps": float(stress.sum()),
        "expected_win_rate": float((expected > 0).mean()),
        "lower_stress_pips_uncorrected": lower_confidence_bound(stress, alpha=alpha),
        "first_20_stress_total_bps": first_n_total(stress, EARLY_FAIL_TRADE_COUNT),
        "block_summaries": block_summaries,
        "positive_block_count": sum(block["stress_total_bps"] > 0 for block in block_summaries),
        "sufficient_block_count": len(sufficient_blocks),
        "negative_sufficient_block_count": len(negative_sufficient_blocks),
    }


def empty_summary() -> dict[str, Any]:
    return {
        "trades": 0,
        "distinct_days": 0,
        "labels": [],
        "expected_mean_bps": None,
        "stress_mean_bps": None,
        "expected_total_bps": 0.0,
        "stress_total_bps": 0.0,
        "expected_win_rate": None,
        "lower_stress_pips_uncorrected": None,
        "first_20_stress_total_bps": None,
        "block_summaries": [],
        "positive_block_count": 0,
        "sufficient_block_count": 0,
        "negative_sufficient_block_count": 0,
    }


def block_summaries_from_observations(
    observations: list[dict[str, Any]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in observations:
        label = str(record.get("label") or record["decision_time"][:10])
        groups.setdefault(label, []).append(record)
    summaries = []
    for label, records in sorted(groups.items()):
        stress = pd.Series([record["stress_net_bps"] for record in records], dtype="float64")
        expected = pd.Series([record["expected_net_bps"] for record in records], dtype="float64")
        summaries.append(
            {
                "label": label,
                "trade_count": len(records),
                "expected_total_bps": float(expected.sum()),
                "stress_total_bps": float(stress.sum()),
                "expected_mean_bps": float(expected.mean()),
                "stress_mean_bps": float(stress.mean()),
                "lower_stress_pips_uncorrected": lower_confidence_bound(stress, alpha=alpha),
            }
        )
    return summaries


def evaluate_gates(summary: dict[str, Any]) -> dict[str, Any]:
    early_fail_total = summary["first_20_stress_total_bps"]
    early_fail = (
        summary["trades"] >= EARLY_FAIL_TRADE_COUNT
        and early_fail_total is not None
        and early_fail_total <= EARLY_FAIL_STRESS_TOTAL_BPS
    )
    block_fail = any(
        block["trade_count"] >= BLOCK_FAIL_MINIMUM_TRADES
        and block["stress_total_bps"] <= BLOCK_FAIL_STRESS_TOTAL_BPS
        for block in summary["block_summaries"]
    )
    lower_stress = summary["lower_stress_pips_uncorrected"]
    minimum_shadow_candidate = (
        summary["trades"] >= SHADOW_MINIMUM_TRADES
        and summary["distinct_days"] >= SHADOW_MINIMUM_DAYS
        and summary["stress_mean_bps"] is not None
        and summary["stress_mean_bps"] > SHADOW_MINIMUM_STRESS_MEAN_BPS
        and summary["expected_mean_bps"] is not None
        and summary["expected_mean_bps"] > SHADOW_MINIMUM_EXPECTED_MEAN_BPS
        and lower_stress is not None
        and lower_stress > 0
        and not early_fail
        and not block_fail
    )
    return {
        "early_fail_trade_count": EARLY_FAIL_TRADE_COUNT,
        "early_fail_stress_total_bps": EARLY_FAIL_STRESS_TOTAL_BPS,
        "early_fail_triggered": early_fail,
        "block_fail_minimum_trades": BLOCK_FAIL_MINIMUM_TRADES,
        "block_fail_stress_total_bps": BLOCK_FAIL_STRESS_TOTAL_BPS,
        "block_fail_triggered": block_fail,
        "minimum_shadow_candidate_trade_count": SHADOW_MINIMUM_TRADES,
        "minimum_shadow_candidate_days": SHADOW_MINIMUM_DAYS,
        "minimum_shadow_candidate_stress_mean_bps": SHADOW_MINIMUM_STRESS_MEAN_BPS,
        "minimum_shadow_candidate_expected_mean_bps": SHADOW_MINIMUM_EXPECTED_MEAN_BPS,
        "minimum_shadow_candidate_gate_passed": minimum_shadow_candidate,
        "testnet_allowed": False,
        "live_allowed": False,
    }


def fast88_status(
    *,
    input_record_count: int,
    feature_audit: dict[str, Any],
    post_preregistration_record_count: int,
    candidate_universe_count: int,
    rule_match_count: int,
    complete_outcome_count: int,
    summary: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if input_record_count == 0:
        return "FAST88_OBSERVATION_NO_INPUT_RECORDS", reasons + ["input_jsonl_empty"]
    if feature_audit["missing_required_features"]:
        return "FAST88_OBSERVATION_NOT_SCOREABLE_MISSING_FEATURES", reasons + [
            f"missing_required_features:{','.join(feature_audit['missing_required_features'])}",
            "fail_closed_no_feature_substitution",
        ]
    if post_preregistration_record_count == 0:
        return "FAST88_OBSERVATION_NO_PROSPECTIVE_RECORDS_YET", reasons + [
            "waiting_for_post_preregistration_records",
        ]
    if candidate_universe_count == 0:
        return "FAST88_OBSERVATION_NO_TOP_P_ENTRIES_YET", reasons + [
            "no_post_preregistration_top_p_entries",
        ]
    if rule_match_count == 0:
        return "FAST88_OBSERVATION_NO_RULE_MATCHES_YET", reasons + [
            "no_post_preregistration_entries_passed_directional_flow_rule",
        ]
    if complete_outcome_count == 0:
        return "FAST88_OBSERVATION_NO_COMPLETE_OUTCOMES_YET", reasons + [
            "rule_matches_lack_complete_replay_horizon",
        ]
    if gates["early_fail_triggered"] or gates["block_fail_triggered"]:
        return "FAST88_OBSERVATION_FAILED_PREREGISTERED_GATE", reasons + [
            "preregistered_failure_gate_triggered",
        ]
    if gates["minimum_shadow_candidate_gate_passed"]:
        return "FAST88_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW", reasons + [
            "minimum_shadow_candidate_gate_passed",
            "testnet_still_requires_new_preregistration",
        ]
    return "FAST88_OBSERVATION_CONTINUES_COLLECTING", reasons + [
        "minimum_shadow_candidate_gate_not_passed",
    ]


def first_n_total(series: pd.Series, n: int) -> float | None:
    if len(series) < n:
        return None
    return float(series.iloc[:n].astype(float).sum())


def group_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)
    return grouped


def preregistration_timestamp_ms() -> int:
    return int(PREREGISTRATION_TIMESTAMP.timestamp() * 1000)


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
        "input_record_count": report["input_record_count"],
        "post_preregistration_record_count": report["post_preregistration_record_count"],
        "candidate_universe_count": report["candidate_universe_count"],
        "rule_match_count": report["rule_match_count"],
        "eligible_observation_count": report["eligible_observation_count"],
        "summary": report["summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

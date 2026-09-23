"""Fast80 preregistered rare vol+flow observation runner.

Fast80 is not a strategy. It is a fail-closed observer for the single setup
frozen in docs/forex/fast80-rare-vol-flow-observation-preregistration.md.

The runner refuses to score records that do not carry the required causal
features. It never sends orders and never writes ledger state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast61-fast60-shadow-settlement"
    / "one_position_settlements.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast80-rare-vol-flow-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast80-rare-vol-flow-observation-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast80-registry.jsonl"
PREREGISTRATION_COMMIT = "a81096175d60cc41c1ad21885c0727b2d225013c"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-23T18:40:07Z")
PREREGISTRATION_SHA256 = "8c98efee222e794ae8c485f2c8f6e61ce9d2ce0ba59700efcf4c9d6685470a2a"
SOURCE_FAST79_REPORT_SHA256 = "bc9b8707e5bb8980b96aac7cb9db306fab4fb98bab83dfff16130a60cfaea7ba"
MODEL_SHA256 = "9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68"
HORIZON_SECONDS = 300
REALIZED_VOL_30S_THRESHOLD = 45.38807193200911
FLOW_IMBALANCE_5S_THRESHOLD = -0.2193072079643347
REQUIRED_FEATURES = ("realized_vol_30s_bps", "flow_imbalance_5s")
DEFAULT_ALPHA = 0.05
EARLY_FAIL_TRADE_COUNT = 20
EARLY_FAIL_STRESS_TOTAL_BPS = -20.0
BLOCK_FAIL_MINIMUM_TRADES = 10
BLOCK_FAIL_STRESS_TOTAL_BPS = -10.0
SHADOW_MINIMUM_TRADES = 100
SHADOW_MINIMUM_DAYS = 10
SHADOW_MINIMUM_STRESS_MEAN_BPS = 1.5
SHADOW_MINIMUM_EXPECTED_MEAN_BPS = 2.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args()
    report = run_fast80_rare_vol_flow_observation(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast80_rare_vol_flow_observation(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha)
    preregistration = validate_preregistration(
        preregistration_path=preregistration_path,
        registry_path=registry_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    input_records = load_jsonl(input_path)
    normalized_records = [normalize_record(record) for record in input_records]
    feature_audit = audit_required_features(normalized_records)

    observations: list[dict[str, Any]] = []
    summary = empty_summary()
    gates = evaluate_gates(summary)
    status, status_reasons = fast80_status(
        input_record_count=len(normalized_records),
        feature_audit=feature_audit,
        post_preregistration_record_count=0,
        summary=summary,
        gates=gates,
    )
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
        ]
        rule_matches = [record for record in candidate_universe if passes_fast80_rule(record)]
        observations = one_position_observations(rule_matches)
        summary = summarize_observations(observations, alpha=alpha)
        gates = evaluate_gates(summary)
        status, status_reasons = fast80_status(
            input_record_count=len(normalized_records),
            feature_audit=feature_audit,
            post_preregistration_record_count=len(post_preregistration),
            summary=summary,
            gates=gates,
        )
    else:
        post_preregistration = []
        candidate_universe = []
        rule_matches = []

    observations_path = output_root / "observations.jsonl"
    write_jsonl(observations_path, observations)
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast80_is_preregistered_observation_only",
            "fail_closed_when_required_features_missing",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_preregistration_and_explicit_authorization",
        ],
        "experiment_stage": "fast80_rare_vol_flow_observation",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path) if input_path.exists() else None,
        "input_record_count": len(normalized_records),
        "post_preregistration_record_count": len(post_preregistration),
        "candidate_universe_count": len(candidate_universe),
        "rule_match_count": len(rule_matches),
        "eligible_observation_count": len(observations),
        "observation_rule": {
            "model_sha256": MODEL_SHA256,
            "horizon_seconds": HORIZON_SECONDS,
            "realized_vol_30s_bps_min": REALIZED_VOL_30S_THRESHOLD,
            "flow_imbalance_5s_max": FLOW_IMBALANCE_5S_THRESHOLD,
        },
        "required_features": list(REQUIRED_FEATURES),
        "feature_audit": feature_audit,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": _sha256_file(registry_path),
        "source_fast79_report_sha256": SOURCE_FAST79_REPORT_SHA256,
        "observations_path": str(observations_path),
        "observations_sha256": _sha256_file(observations_path),
        "summary": summary,
        "gates": gates,
    }
    return _write_report(output_root, report)


def validate_parameters(*, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> dict[str, str]:
    if not preregistration_path.exists():
        raise FileNotFoundError(preregistration_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    preregistration_sha256 = _sha256_file(preregistration_path)
    if preregistration_sha256 != PREREGISTRATION_SHA256:
        raise ValueError("Fast80 preregistration hash changed")
    registry_events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in registry_events
        if event.get("event") == "rare_vol_flow_observation_preregistration_created"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast80 registry must contain exactly one preregistration event")
    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_fast79_report_sha256": SOURCE_FAST79_REPORT_SHA256,
        "feature_required_1": "realized_vol_30s_bps",
        "feature_required_2": "flow_imbalance_5s",
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast80 registry mismatch: {mismatches}")
    return {"preregistration_sha256": preregistration_sha256}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if normalized.get("decision_time_ms") is None and normalized.get("decision_time") is not None:
        normalized["decision_time_ms"] = int(
            pd.Timestamp(str(normalized["decision_time"])).timestamp() * 1000
        )
    for feature in REQUIRED_FEATURES:
        feature_value = feature_value_from_record(record, feature)
        if feature_value is not None:
            normalized[feature] = feature_value
    return normalized


def feature_value_from_record(record: dict[str, Any], feature: str) -> float | None:
    candidate_sources = (
        record,
        record.get("feature_values", {}),
        record.get("causal_features", {}),
        record.get("directional_gate_values", {}),
        record.get("gate_values", {}),
    )
    for source in candidate_sources:
        if isinstance(source, dict) and source.get(feature) is not None:
            return float(source[feature])
    return None


def audit_required_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    missing_counts = {
        feature: sum(record.get(feature) is None for record in records)
        for feature in REQUIRED_FEATURES
    }
    missing_required_features = [
        feature for feature, missing_count in missing_counts.items() if missing_count > 0
    ]
    return {
        "record_count": len(records),
        "missing_counts": missing_counts,
        "missing_required_features": missing_required_features,
        "scoreable": not missing_required_features,
    }


def preregistration_timestamp_ms() -> int:
    return int(PREREGISTRATION_TIMESTAMP.timestamp() * 1000)


def passes_fast80_rule(record: dict[str, Any]) -> bool:
    return (
        float(record["realized_vol_30s_bps"]) >= REALIZED_VOL_30S_THRESHOLD
        and float(record["flow_imbalance_5s"]) <= FLOW_IMBALANCE_5S_THRESHOLD
    )


def one_position_observations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda record: (
            int(record["decision_time_ms"]),
            -float(record.get("probability", 0.0)),
        ),
    )
    observations = []
    position_exit_ms = -1
    for record in ordered:
        decision_time_ms = int(record["decision_time_ms"])
        if decision_time_ms <= position_exit_ms:
            continue
        position_exit_ms = int(record["exit_time_ms"])
        observations.append(observation_record(record))
    return observations


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
        "realized_vol_30s_bps": float(record["realized_vol_30s_bps"]),
        "flow_imbalance_5s": float(record["flow_imbalance_5s"]),
        "expected_net_bps": float(record["expected_net_bps"]),
        "stress_net_bps": float(record["stress_net_bps"]),
        "exit_reason": str(record.get("exit_reason", "")),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


def summarize_observations(observations: list[dict[str, Any]], *, alpha: float) -> dict[str, Any]:
    if not observations:
        return empty_summary()
    expected = pd.Series([record["expected_net_bps"] for record in observations], dtype="float64")
    stress = pd.Series([record["stress_net_bps"] for record in observations], dtype="float64")
    decision_times = pd.Series([record["decision_time_ms"] for record in observations], dtype="int64")
    labels = sorted({str(record.get("label", "")) for record in observations})
    block_summaries = block_summaries_from_observations(observations, alpha=alpha)
    positive_blocks = sum(block["stress_total_bps"] > 0 for block in block_summaries)
    sufficient_blocks = [
        block for block in block_summaries if block["trade_count"] >= BLOCK_FAIL_MINIMUM_TRADES
    ]
    negative_sufficient_blocks = [
        block for block in sufficient_blocks if block["stress_total_bps"] <= BLOCK_FAIL_STRESS_TOTAL_BPS
    ]
    return {
        "trades": len(observations),
        "distinct_days": int(pd.to_datetime(decision_times, unit="ms", utc=True).dt.date.nunique()),
        "labels": labels,
        "expected_mean_bps": float(expected.mean()),
        "stress_mean_bps": float(stress.mean()),
        "expected_total_bps": float(expected.sum()),
        "stress_total_bps": float(stress.sum()),
        "expected_win_rate": float((expected > 0).mean()),
        "lower_stress_pips_uncorrected": lower_confidence_bound(stress, alpha=alpha),
        "first_20_stress_total_bps": first_n_total(stress, EARLY_FAIL_TRADE_COUNT),
        "block_summaries": block_summaries,
        "positive_block_count": positive_blocks,
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


def fast80_status(
    *,
    input_record_count: int,
    feature_audit: dict[str, Any],
    post_preregistration_record_count: int,
    summary: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if input_record_count == 0:
        return "FAST80_OBSERVATION_NO_INPUT_RECORDS", reasons + ["input_jsonl_empty"]
    if feature_audit["missing_required_features"]:
        return "FAST80_OBSERVATION_NOT_SCOREABLE_MISSING_FEATURES", reasons + [
            f"missing_required_features:{','.join(feature_audit['missing_required_features'])}",
            "fail_closed_no_feature_substitution",
        ]
    if post_preregistration_record_count == 0:
        return "FAST80_OBSERVATION_NO_PROSPECTIVE_RECORDS_YET", reasons + [
            "waiting_for_post_preregistration_records",
        ]
    if summary["trades"] == 0:
        return "FAST80_OBSERVATION_NO_RULE_MATCHES_YET", reasons + [
            "no_post_preregistration_records_passed_frozen_rule",
        ]
    if gates["early_fail_triggered"] or gates["block_fail_triggered"]:
        return "FAST80_OBSERVATION_FAILED_PREREGISTERED_GATE", reasons + [
            "preregistered_failure_gate_triggered",
        ]
    if gates["minimum_shadow_candidate_gate_passed"]:
        return "FAST80_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW", reasons + [
            "minimum_shadow_candidate_gate_passed",
            "testnet_still_requires_new_preregistration",
        ]
    return "FAST80_OBSERVATION_CONTINUES_COLLECTING", reasons + [
        "minimum_shadow_candidate_gate_not_passed",
    ]


def first_n_total(series: pd.Series, n: int) -> float | None:
    if len(series) < n:
        return None
    return float(series.iloc[:n].astype(float).sum())


def lower_confidence_bound(values: list[float] | pd.Series | np.ndarray, *, alpha: float) -> float | None:
    series = pd.Series(values, dtype="float64").dropna()
    sample_size = len(series)
    if sample_size < 2:
        return None
    standard_error = float(series.std(ddof=1) / np.sqrt(sample_size))
    return float(series.mean() - t.ppf(1.0 - alpha, sample_size - 1) * standard_error)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


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
        "eligible_observation_count": report["eligible_observation_count"],
        "feature_audit": report["feature_audit"],
        "summary": report["summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

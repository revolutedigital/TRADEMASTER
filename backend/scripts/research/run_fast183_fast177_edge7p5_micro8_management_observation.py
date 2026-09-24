"""Fast183 future-only observer for Fast177 edge-7.5 plus micro_8_3 management."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.trailing_portfolio import HistoricalTrailingSimulator, LOW_COST_TRAILING_POLICIES  # noqa: E402
from scripts.research.run_fast179_fast177_path_management_audit import (  # noqa: E402
    DEFAULT_FAST177_ROWS,
    DEFAULT_TRADE_PARTITION,
    load_source_rows,
    load_trade_partition,
    parse_decision_time_ms,
    score_decision_path,
    sha256_file,
    write_jsonl,
)
from scripts.research.run_fast180_fast178_edge2_micro3_management_observation import (  # noqa: E402
    DEFAULT_ALPHA,
    EARLY_FAIL_DAY_BPS,
    EARLY_FAIL_TOTAL_BPS,
    HORIZON_SECONDS,
    LATENCY_MS,
    MIN_EARLY_FAIL_DAY_DECISIONS,
    MIN_EARLY_FAIL_DECISIONS,
    PROFIT_COST_BPS,
    SHADOW_MIN_DAYS,
    SHADOW_MIN_DECISIONS,
    SHADOW_MIN_MEAN_BPS,
    SHADOW_MIN_POSITIVE_DAY_SHARE,
    STRESS_COST_BPS,
    evaluate_gates,
    lower_confidence_bound,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast183-fast177-edge7p5-micro8-management-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast183-fast177-edge7p5-micro8-management-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast183-registry.jsonl"

PREREGISTRATION_TIMESTAMP = datetime(2026, 9, 24, 14, 33, 5, 785000, tzinfo=UTC)
PREREGISTRATION_SHA256 = "745e324548e825c0cff3737f410606e0a51df415bc34e2f42578773eb566d998"
TARGET_SYMBOL = "ETHUSDT"
SOURCE_EXPERIMENT = "fast177_fast176_relaxed_edge_observation"
MANAGEMENT_POLICY_NAME = "micro_8_3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast177-rows", type=Path, default=DEFAULT_FAST177_ROWS)
    parser.add_argument("--trade-partition", type=Path, default=DEFAULT_TRADE_PARTITION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args(argv)
    report = run_fast183_observation(
        fast177_rows_path=args.fast177_rows,
        trade_partition_path=args.trade_partition,
        output_root=args.output_root,
        preregistration_path=args.preregistration_path,
        registry_path=args.registry_path,
        alpha=args.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast183_observation(
    *,
    fast177_rows_path: Path = DEFAULT_FAST177_ROWS,
    trade_partition_path: Path = DEFAULT_TRADE_PARTITION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_preregistration(preregistration_path=preregistration_path, registry_path=registry_path)
    policy = management_policy()
    source_rows = filter_post_preregistration_rows(
        load_source_rows(fast177_rows_path, source_experiment="fast177")
    )
    event_times_ms, prices = load_trade_partition(trade_partition_path)
    simulator = HistoricalTrailingSimulator(
        event_times_ms,
        prices,
        latency_ms=LATENCY_MS,
        expected_round_trip_bps=PROFIT_COST_BPS,
        stress_round_trip_bps=STRESS_COST_BPS,
    )
    scored_rows = [
        score_decision_path(
            row,
            event_times_ms=event_times_ms,
            prices=prices,
            simulator=simulator,
            policies=(policy,),
            latency_ms=LATENCY_MS,
            horizon_seconds=HORIZON_SECONDS,
            stress_cost_bps=STRESS_COST_BPS,
        )
        for row in source_rows
    ]
    observation_rows = [observation_record(row) for row in scored_rows if bool(row["eligible"])]
    observation_summary = summarize_observations(observation_rows, alpha=alpha)
    gates = evaluate_gates(observation_summary)
    status, status_reasons = fast183_status(
        prospective_row_count=len(scored_rows),
        observation_count=len(observation_rows),
        gates=gates,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    prospective_path = output_root / "prospective-diagnostics.jsonl"
    observations_path = output_root / "observations.jsonl"
    summary_path = output_root / "summary.json"
    write_jsonl(prospective_path, scored_rows)
    write_jsonl(observations_path, observation_rows)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "target_symbol": TARGET_SYMBOL,
        "source_experiment": SOURCE_EXPERIMENT,
        "management_policy": MANAGEMENT_POLICY_NAME,
        "horizon_seconds": HORIZON_SECONDS,
        "latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "prospective_row_count": len(scored_rows),
        "eligible_observation_count": len(observation_rows),
        "observation_summary": observation_summary,
        "gates": gates,
    }
    write_json(summary_path, summary)
    report_path = output_root / "report.json"
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast183_is_preregistered_future_only_observation",
            "source_fast177_rows_must_be_generated_before_fast183_replay",
            "micro8_management_is_frozen_after_fast182_historical_grid",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast183_fast177_edge7p5_micro8_management_observation",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "source": {
            "fast177_rows_path": str(fast177_rows_path),
            "fast177_rows_sha256": sha256_file(fast177_rows_path),
            "trade_partition_path": str(trade_partition_path),
            "trade_partition_sha256": sha256_file(trade_partition_path),
        },
        "artifacts": {
            "prospective_diagnostics": {"path": str(prospective_path), "sha256": sha256_file(prospective_path)},
            "observations": {"path": str(observations_path), "sha256": sha256_file(observations_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> None:
    actual_sha = sha256_file(preregistration_path)
    if actual_sha != PREREGISTRATION_SHA256:
        raise ValueError(
            f"Fast183 preregistration hash mismatch: expected {PREREGISTRATION_SHA256}, got {actual_sha}"
        )
    events = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preregistered = [
        event
        for event in events
        if event.get("event") == "fast183_fast177_edge7p5_micro8_management_preregistered"
    ]
    if not preregistered:
        raise ValueError("Fast183 preregistration event missing from registry")
    if preregistered[-1].get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("Fast183 registry preregistration hash mismatch")


def management_policy():
    for policy in LOW_COST_TRAILING_POLICIES:
        if policy.name == MANAGEMENT_POLICY_NAME:
            return policy
    raise ValueError(f"missing management policy: {MANAGEMENT_POLICY_NAME}")


def filter_post_preregistration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_ms = int(PREREGISTRATION_TIMESTAMP.timestamp() * 1_000)
    return [row for row in rows if parse_decision_time_ms(str(row["decision_time"])) > anchor_ms]


def observation_record(row: dict[str, Any]) -> dict[str, Any]:
    policy_result = row["policy_results"][0]
    return {
        "decision_time": row["decision_time"],
        "date": row["decision_time"][:10],
        "target_symbol": TARGET_SYMBOL,
        "side": row["side"],
        "prediction": row["prediction"],
        "edge": row["edge"],
        "actual_bps": row["actual_bps"],
        "terminal_stress_net_bps": row["terminal_stress_net_bps"],
        "maximum_favorable_bps": row["maximum_favorable_bps"],
        "maximum_adverse_bps": row["maximum_adverse_bps"],
        "favorable_before_adverse": row["favorable_before_adverse"],
        "management_policy": MANAGEMENT_POLICY_NAME,
        "exit_reason": policy_result["exit_reason"],
        "gross_bps": policy_result["gross_bps"],
        "expected_net_bps": policy_result["expected_net_bps"],
        "stress_net_bps": policy_result["stress_net_bps"],
        "regime_gate_value": row["regime_gate_value"],
        "regime_gate_threshold": row["regime_gate_threshold"],
    }


def summarize_observations(observations: list[dict[str, Any]], *, alpha: float) -> dict[str, Any]:
    if not observations:
        return {
            "decision_count": 0,
            "day_count": 0,
            "stress_net_total_bps": 0.0,
            "stress_net_mean_bps": 0.0,
            "stress_net_lcb_bps": 0.0,
            "positive_day_share": 0.0,
            "sufficient_day_count": 0,
            "worst_sufficient_day_net_bps": 0.0,
            "exit_reason_counts": {},
        }
    import pandas as pd

    frame = pd.DataFrame(observations)
    stress = frame["stress_net_bps"].astype(float)
    day_summary = frame.groupby("date", sort=True).agg(
        trade_count=("stress_net_bps", "size"),
        stress_net_total_bps=("stress_net_bps", "sum"),
    )
    sufficient_days = day_summary.loc[day_summary["trade_count"] >= MIN_EARLY_FAIL_DAY_DECISIONS]
    return {
        "decision_count": int(len(frame)),
        "day_count": int(frame["date"].nunique()),
        "stress_net_total_bps": float(stress.sum()),
        "stress_net_mean_bps": float(stress.mean()),
        "stress_net_lcb_bps": lower_confidence_bound(stress, alpha=alpha),
        "positive_day_share": (
            0.0 if sufficient_days.empty else float((sufficient_days["stress_net_total_bps"] > 0).mean())
        ),
        "sufficient_day_count": int(len(sufficient_days)),
        "worst_sufficient_day_net_bps": (
            0.0 if sufficient_days.empty else float(sufficient_days["stress_net_total_bps"].min())
        ),
        "exit_reason_counts": {
            str(reason): int(count)
            for reason, count in frame["exit_reason"].value_counts(sort=False).sort_index().items()
        },
    }


def fast183_status(*, prospective_row_count: int, observation_count: int, gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "prospective_observation"]
    if prospective_row_count <= 0:
        return "FAST183_WAITING_FOR_POST_PREREGISTRATION_DATA", reasons + ["no_post_preregistration_decision_rows"]
    if observation_count <= 0:
        return "FAST183_DATA_FRESH_NO_ELIGIBLE_DECISIONS_YET", reasons + ["fast177_edge_or_regime_gate_not_met"]
    if gates["early_fail"]:
        return "FAST183_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST183_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed", "still_not_approved_strategy"],
        )
    return "FAST183_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE", reasons + gates["shadow_reasons"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "approved_strategy": report["approved_strategy"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "prospective_row_count": report["summary"]["prospective_row_count"],
        "eligible_observation_count": report["summary"]["eligible_observation_count"],
        "observation_summary": report["summary"]["observation_summary"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

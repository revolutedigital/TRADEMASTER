"""Fast186 edge-calibration diagnostic for the Fast185 candidate.

Fast185 has observed fresh post-preregistration rows, but none crossed the
historical edge>=7.5 gate. Fast186 checks whether simply lowering that edge
gate would be justified by historical post-training robustness and by the
current Fast185 rows.

This is diagnostic research only. It does not change Fast185, does not
preregister a new candidate, and never authorizes orders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import (  # noqa: E402
    DEFAULT_BTC_TRADE_ROOT,
    DEFAULT_ETH_TRADE_ROOT,
    DEFAULT_SOL_TRADE_ROOT,
    TRAIN_BLOCKS,
    build_all_feature_rows,
    leadlag_model,
)
from scripts.research.run_fast178_fast177_edge2_observation import (  # noqa: E402
    build_training_label_rows,
    feature_columns_from_rows,
)
from scripts.research.run_fast179_fast177_path_management_audit import sha256_file, write_jsonl  # noqa: E402
from scripts.research.run_fast181_edge2_micro3_historical_management_audit import (  # noqa: E402
    HORIZON_SECONDS,
    LATENCY_MS,
    PROFIT_COST_BPS,
    REGIME_GATE_FEATURE,
    REGIME_GATE_THRESHOLD,
    STRESS_COST_BPS,
    simulator_for_date,
)
from scripts.research.run_fast184_robust_session_filter_historical_audit import (  # noqa: E402
    POST_TRAINING_DATES,
    build_post_training_rows,
    score_post_training_rows,
    select_filter_rows,
    settle_rows,
    summarize_filter_records,
)
from scripts.research.run_fast185_fast177_edge7p5_micro5_session_observation import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as FAST185_OUTPUT_ROOT,
    SESSION_END_HOUR_UTC,
    SESSION_START_HOUR_UTC,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast186-fast185-edge-calibration-diagnostic"
)
DEFAULT_FAST185_PROSPECTIVE_ROWS = FAST185_OUTPUT_ROOT / "prospective-diagnostics.jsonl"

TARGET_SYMBOL = "ETHUSDT"
MANAGEMENT_POLICY = "micro_5_2"
EDGE_THRESHOLDS_BPS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5)
SELECTED_EDGE_THRESHOLD_BPS = 7.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--fast185-prospective-rows", type=Path, default=DEFAULT_FAST185_PROSPECTIVE_ROWS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    report = run_fast186_edge_calibration_diagnostic(
        trade_roots={
            "BTCUSDT": args.btc_trade_root,
            "ETHUSDT": args.eth_trade_root,
            "SOLUSDT": args.sol_trade_root,
        },
        fast185_prospective_rows_path=args.fast185_prospective_rows,
        output_root=args.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast186_edge_calibration_diagnostic(
    *,
    trade_roots: dict[str, Path],
    fast185_prospective_rows_path: Path = DEFAULT_FAST185_PROSPECTIVE_ROWS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    training_universe = build_all_feature_rows(trade_roots)
    feature_columns = feature_columns_from_rows(training_universe)
    training_rows = build_training_label_rows(training_universe)
    model = leadlag_model()
    model.fit(training_rows[feature_columns], training_rows["actual_bps"])

    post_training_rows = build_post_training_rows(trade_roots=trade_roots)
    scored_rows = score_post_training_rows(post_training_rows, model, feature_columns)
    simulators = {
        date: simulator_for_date(trade_roots[TARGET_SYMBOL], date)
        for date in sorted(scored_rows["date"].astype(str).unique())
    }
    historical_settlements = []
    for threshold in EDGE_THRESHOLDS_BPS:
        filter_spec = threshold_filter_spec(threshold)
        selected = select_filter_rows(scored_rows, filter_spec)
        historical_settlements.extend(settle_rows(selected, filter_spec=filter_spec, simulators=simulators))
    historical_summaries = summarize_filter_records(historical_settlements)
    live_rows = load_fast185_rows(fast185_prospective_rows_path)
    live_threshold_summaries = summarize_live_thresholds(live_rows)
    status, status_reasons = fast186_status(
        historical_summaries=historical_summaries,
        live_threshold_summaries=live_threshold_summaries,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    historical_settlements_path = output_root / "historical-threshold-settlements.jsonl"
    live_rows_path = output_root / "fast185-live-row-calibration.jsonl"
    summary_path = output_root / "summary.json"
    write_jsonl(historical_settlements_path, historical_settlements)
    write_jsonl(live_rows_path, live_rows)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "target_symbol": TARGET_SYMBOL,
        "train_blocks": list(TRAIN_BLOCKS),
        "post_training_dates": list(POST_TRAINING_DATES),
        "horizon_seconds": HORIZON_SECONDS,
        "latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "session_start_hour_utc": SESSION_START_HOUR_UTC,
        "session_end_hour_utc": SESSION_END_HOUR_UTC,
        "management_policy": MANAGEMENT_POLICY,
        "edge_thresholds_bps": list(EDGE_THRESHOLDS_BPS),
        "selected_edge_threshold_bps": SELECTED_EDGE_THRESHOLD_BPS,
        "historical_threshold_summaries": historical_summaries,
        "live_threshold_summaries": live_threshold_summaries,
        "live_row_count": len(live_rows),
        "live_max_edge_bps": max((float(row["edge"]) for row in live_rows), default=0.0),
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
            "fast186_is_edge_calibration_diagnostic_only",
            "lower_edge_thresholds_are_historical_post_hoc_variants",
            "fast185_rule_remains_unchanged",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast186_fast185_edge_calibration_diagnostic",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "source": {
            "fast185_prospective_rows_path": str(fast185_prospective_rows_path),
            "fast185_prospective_rows_sha256": sha256_file(fast185_prospective_rows_path),
        },
        "artifacts": {
            "historical_threshold_settlements": {
                "path": str(historical_settlements_path),
                "sha256": sha256_file(historical_settlements_path),
            },
            "fast185_live_row_calibration": {
                "path": str(live_rows_path),
                "sha256": sha256_file(live_rows_path),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def threshold_filter_spec(threshold: float) -> dict[str, Any]:
    return {
        "candidate_id": f"edge{threshold:g}_micro5_all_utc05_16",
        "role": "edge_calibration_threshold",
        "edge_threshold_bps": threshold,
        "management_policy": MANAGEMENT_POLICY,
        "side_filter": "ALL",
        "start_hour_utc": SESSION_START_HOUR_UTC,
        "end_hour_utc": SESSION_END_HOUR_UTC,
    }


def load_fast185_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_live_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for threshold in EDGE_THRESHOLDS_BPS:
        selected = [row for row in rows if live_row_passes_threshold(row, threshold)]
        stresses = [float(row["policy_results"][0]["stress_net_bps"]) for row in selected]
        summaries.append(
            {
                "edge_threshold_bps": float(threshold),
                "row_count": len(selected),
                "stress_total_bps": float(sum(stresses)),
                "stress_mean_bps": float(sum(stresses) / len(stresses)) if stresses else 0.0,
                "latest_decision_time": None if not selected else str(selected[-1]["decision_time"]),
                "max_edge_bps": max((float(row["edge"]) for row in selected), default=0.0),
            }
        )
    return summaries


def live_row_passes_threshold(row: dict[str, Any], threshold: float) -> bool:
    return (
        float(row["edge"]) >= threshold
        and bool(row["fast185_regime_gate_passed"])
        and bool(row["fast185_session_gate_passed"])
        and bool(row["fast185_side_gate_passed"])
    )


def fast186_status(
    *,
    historical_summaries: list[dict[str, Any]],
    live_threshold_summaries: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "edge_calibration_diagnostic"]
    selected = summary_for_threshold(historical_summaries, SELECTED_EDGE_THRESHOLD_BPS)
    lower = [
        item
        for item in historical_summaries
        if float(item["edge_threshold_bps"]) < SELECTED_EDGE_THRESHOLD_BPS
    ]
    robust_lower = [
        item
        for item in lower
        if int(item["positive_day_count"]) == len(POST_TRAINING_DATES)
        and float(item["worst_day_stress_total_bps"]) > 0
    ]
    live_lower = [
        item
        for item in live_threshold_summaries
        if 0.0 < float(item["edge_threshold_bps"]) < SELECTED_EDGE_THRESHOLD_BPS
        and int(item["row_count"]) > 0
    ]
    if selected is None:
        return "FAST186_SELECTED_THRESHOLD_MISSING", reasons + ["selected_edge_threshold_not_evaluated"]
    if robust_lower:
        return (
            "FAST186_LOWER_EDGE_HAS_HISTORICAL_ROBUST_SURVIVOR_REQUIRES_NEW_PREREGISTRATION",
            reasons + ["lower_edge_historical_survivor_found", "fresh_future_preregistration_required"],
        )
    return (
        "FAST186_REJECTS_SIMPLE_EDGE_LOWERING",
        reasons
        + [
            "no_lower_edge_threshold_retains_9_of_9_positive_days",
            f"selected_threshold_positive_days:{selected['positive_day_count']}",
            f"selected_threshold_worst_day:{selected['worst_day_stress_total_bps']}",
            f"live_lower_thresholds_with_rows:{len(live_lower)}",
            "keep_fast185_gate_until_new_filter_is_found",
        ],
    )


def summary_for_threshold(summaries: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    for item in summaries:
        if float(item["edge_threshold_bps"]) == threshold:
            return item
    return None


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
        "live_row_count": report["summary"]["live_row_count"],
        "live_max_edge_bps": report["summary"]["live_max_edge_bps"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

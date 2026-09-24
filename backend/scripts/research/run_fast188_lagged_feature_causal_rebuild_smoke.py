"""Fast188 lagged-feature causal rebuild smoke.

Fast187 showed the Fast185 candidate fails when entry is delayed until a
one-minute feature bar can be observed. Fast188 asks a narrower follow-up:

If we force every model feature to be one 5-minute decision step old, does a
simple edge-threshold version of the same cross-asset idea survive?

This is a conservative causal smoke only. It is research-only and never
authorizes orders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    select_filter_rows,
    settle_rows,
    summarize_filter_records,
)
from scripts.research.run_fast185_fast177_edge7p5_micro5_session_observation import (  # noqa: E402
    SESSION_END_HOUR_UTC,
    SESSION_START_HOUR_UTC,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast188-lagged-feature-causal-rebuild-smoke"
)
TARGET_SYMBOL = "ETHUSDT"
EDGE_THRESHOLDS_BPS = (1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)
POLICIES = ("micro_3_1", "micro_5_2", "micro_8_3")
DECISION_LAG_STEPS = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    report = run_fast188_lagged_feature_causal_rebuild_smoke(
        trade_roots={
            "BTCUSDT": args.btc_trade_root,
            "ETHUSDT": args.eth_trade_root,
            "SOLUSDT": args.sol_trade_root,
        },
        output_root=args.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast188_lagged_feature_causal_rebuild_smoke(
    *,
    trade_roots: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    training_universe = build_all_feature_rows(trade_roots)
    feature_columns = feature_columns_from_rows(training_universe)
    lagged_training_universe = lag_feature_columns(training_universe, feature_columns)
    training_rows = build_training_label_rows(lagged_training_universe).dropna(
        subset=[*feature_columns, "actual_bps"]
    )
    model = leadlag_model()
    model.fit(training_rows[feature_columns], training_rows["actual_bps"])

    post_training_rows = build_post_training_rows(trade_roots=trade_roots)
    scored_rows = score_lagged_post_training_rows(
        post_training_rows=post_training_rows,
        model=model,
        feature_columns=feature_columns,
    )
    simulators = {
        date: simulator_for_date(trade_roots[TARGET_SYMBOL], date)
        for date in sorted(scored_rows["date"].astype(str).unique())
    }
    settlements = []
    for policy_name in POLICIES:
        for threshold in EDGE_THRESHOLDS_BPS:
            filter_spec = threshold_filter_spec(threshold=threshold, policy_name=policy_name)
            selected_rows = select_filter_rows(scored_rows, filter_spec)
            settlements.extend(settle_rows(selected_rows, filter_spec=filter_spec, simulators=simulators))

    candidate_summaries = summarize_filter_records(settlements)
    status, status_reasons = fast188_status(candidate_summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    settlements_path = output_root / "lagged-feature-settlements.jsonl"
    summary_path = output_root / "summary.json"
    write_jsonl(settlements_path, settlements)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "target_symbol": TARGET_SYMBOL,
        "train_blocks": list(TRAIN_BLOCKS),
        "post_training_dates": list(POST_TRAINING_DATES),
        "decision_lag_steps": DECISION_LAG_STEPS,
        "horizon_seconds": HORIZON_SECONDS,
        "latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "session_start_hour_utc": SESSION_START_HOUR_UTC,
        "session_end_hour_utc": SESSION_END_HOUR_UTC,
        "edge_thresholds_bps": list(EDGE_THRESHOLDS_BPS),
        "policies": list(POLICIES),
        "feature_count": len(feature_columns),
        "training_row_count": len(training_rows),
        "scored_row_count": len(scored_rows),
        "candidate_summaries": candidate_summaries,
        "top_candidates": candidate_summaries[:10],
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
            "fast188_is_lagged_feature_causal_rebuild_smoke_only",
            "no_simple_lagged_threshold_policy_survivor_found",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast188_lagged_feature_causal_rebuild_smoke",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "lagged_feature_settlements": {
                "path": str(settlements_path),
                "sha256": sha256_file(settlements_path),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def lag_feature_columns(rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    lagged = rows.sort_values(["date", "decision_time"]).copy()
    for column in feature_columns:
        lagged[column] = lagged.groupby("date", sort=False)[column].shift(DECISION_LAG_STEPS)
    return lagged


def score_lagged_post_training_rows(
    *,
    post_training_rows: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> pd.DataFrame:
    scored = lag_feature_columns(post_training_rows, feature_columns)
    shift_rows = int(HORIZON_SECONDS / 60 / 5)
    scored["future_price"] = scored.groupby("date")[f"{TARGET_SYMBOL}_price"].shift(-shift_rows)
    scored["actual_bps"] = (
        scored["future_price"] / scored[f"{TARGET_SYMBOL}_price"] - 1.0
    ) * 10000.0
    scored = scored.dropna(subset=[*feature_columns, "actual_bps"]).copy()
    scored["prediction"] = model.predict(scored[feature_columns])
    scored["edge"] = scored["prediction"].abs()
    scored["side_label"] = np.where(scored["prediction"] > 0, "LONG", "SHORT")
    scored["side_mult"] = np.sign(scored["prediction"])
    scored["terminal_net_bps"] = scored["actual_bps"] * scored["side_mult"] - STRESS_COST_BPS
    scored["decision_hour_utc"] = scored["decision_time"].dt.tz_convert("UTC").dt.hour
    return scored


def threshold_filter_spec(*, threshold: float, policy_name: str) -> dict[str, Any]:
    return {
        "candidate_id": f"lag1_edge{threshold:g}_{policy_name}_all_utc05_16",
        "role": "lagged_feature_threshold_policy",
        "edge_threshold_bps": threshold,
        "management_policy": policy_name,
        "side_filter": "ALL",
        "start_hour_utc": SESSION_START_HOUR_UTC,
        "end_hour_utc": SESSION_END_HOUR_UTC,
    }


def fast188_status(candidate_summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "lagged_feature_causal_smoke"]
    robust = [
        item
        for item in candidate_summaries
        if int(item["positive_day_count"]) == len(POST_TRAINING_DATES)
        and float(item["worst_day_stress_total_bps"]) > 0
    ]
    if robust:
        best = robust[0]
        return (
            "FAST188_LAGGED_FEATURE_CAUSAL_SURVIVOR_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"survivors:{len(robust)}",
                f"best:{best['candidate_id']}",
                "fresh_future_preregistration_required",
            ],
        )
    return (
        "FAST188_LAGGED_FEATURE_CAUSAL_REBUILD_NO_SIMPLE_SURVIVOR",
        reasons + ["no_threshold_policy_candidate_positive_on_9_of_9_days"],
    )


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
        "top_candidates": [
            {
                "candidate_id": item["candidate_id"],
                "trade_count": item["trade_count"],
                "stress_total_bps": item["stress_total_bps"],
                "positive_day_count": item["positive_day_count"],
                "worst_day_stress_total_bps": item["worst_day_stress_total_bps"],
            }
            for item in report["summary"]["top_candidates"][:5]
        ],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

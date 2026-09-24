"""Fast187 timestamp-causality audit for the Fast185 candidate.

Fast187 tests whether the Fast184/Fast185 edge survives a causal close-delay
stress. The existing feature rows are labelled at one-minute timestamps. If a
feature uses the just-closed minute, an execution system cannot enter 100 ms
after the minute's opening timestamp; it must wait for the minute to be closed
and observable.

This diagnostic keeps the Fast185 rule fixed and only delays the simulated
entry timestamp. It is research-only and never authorizes orders.
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

from app.services.backtest.event_replay import OrderSide  # noqa: E402
from app.services.backtest.trailing_portfolio import LOW_COST_TRAILING_POLICIES  # noqa: E402
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
    summarize_rows,
)
from scripts.research.run_fast184_robust_session_filter_historical_audit import (  # noqa: E402
    POST_TRAINING_DATES,
    build_post_training_rows,
    score_post_training_rows,
)
from scripts.research.run_fast185_fast177_edge7p5_micro5_session_observation import (  # noqa: E402
    EDGE_THRESHOLD_BPS,
    MANAGEMENT_POLICY_NAME,
    SESSION_END_HOUR_UTC,
    SESSION_START_HOUR_UTC,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast187-fast185-timestamp-causality-audit"
)
TARGET_SYMBOL = "ETHUSDT"
ENTRY_DELAY_SECONDS = (0, 60, 120, 300)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    report = run_fast187_timestamp_causality_audit(
        trade_roots={
            "BTCUSDT": args.btc_trade_root,
            "ETHUSDT": args.eth_trade_root,
            "SOLUSDT": args.sol_trade_root,
        },
        output_root=args.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast187_timestamp_causality_audit(
    *,
    trade_roots: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    training_universe = build_all_feature_rows(trade_roots)
    feature_columns = feature_columns_from_rows(training_universe)
    training_rows = build_training_label_rows(training_universe)
    model = leadlag_model()
    model.fit(training_rows[feature_columns], training_rows["actual_bps"])

    post_training_rows = build_post_training_rows(trade_roots=trade_roots)
    scored_rows = score_post_training_rows(post_training_rows, model, feature_columns)
    selected_rows = select_fast185_rows(scored_rows)
    simulators = {
        date: simulator_for_date(trade_roots[TARGET_SYMBOL], date)
        for date in sorted(selected_rows["date"].astype(str).unique())
    }
    settlements = []
    policy = management_policy()
    for delay_seconds in ENTRY_DELAY_SECONDS:
        settlements.extend(
            settle_with_entry_delay(
                selected_rows=selected_rows,
                simulators=simulators,
                policy=policy,
                delay_seconds=delay_seconds,
            )
        )
    delay_summaries = summarize_delays(settlements)
    status, status_reasons = fast187_status(delay_summaries)

    output_root.mkdir(parents=True, exist_ok=True)
    settlements_path = output_root / "fast185-delay-settlements.jsonl"
    summary_path = output_root / "summary.json"
    write_jsonl(settlements_path, settlements)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "target_symbol": TARGET_SYMBOL,
        "train_blocks": list(TRAIN_BLOCKS),
        "post_training_dates": list(POST_TRAINING_DATES),
        "horizon_seconds": HORIZON_SECONDS,
        "base_latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "session_start_hour_utc": SESSION_START_HOUR_UTC,
        "session_end_hour_utc": SESSION_END_HOUR_UTC,
        "edge_threshold_bps": EDGE_THRESHOLD_BPS,
        "management_policy": MANAGEMENT_POLICY_NAME,
        "entry_delay_seconds": list(ENTRY_DELAY_SECONDS),
        "feature_count": len(feature_columns),
        "training_row_count": len(training_rows),
        "selected_row_count": len(selected_rows),
        "delay_summaries": delay_summaries,
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
            "fast187_is_timestamp_causality_diagnostic_only",
            "fast185_must_survive_causal_close_delay_before_strategy_consideration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast187_fast185_timestamp_causality_audit",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "fast185_delay_settlements": {
                "path": str(settlements_path),
                "sha256": sha256_file(settlements_path),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def select_fast185_rows(scored_rows: pd.DataFrame) -> pd.DataFrame:
    return scored_rows.loc[
        scored_rows["edge"].ge(EDGE_THRESHOLD_BPS)
        & scored_rows[REGIME_GATE_FEATURE].ge(REGIME_GATE_THRESHOLD)
        & scored_rows["side_mult"].ne(0)
        & scored_rows["decision_hour_utc"].between(SESSION_START_HOUR_UTC, SESSION_END_HOUR_UTC)
    ].copy()


def management_policy():
    for policy in LOW_COST_TRAILING_POLICIES:
        if policy.name == MANAGEMENT_POLICY_NAME:
            return policy
    raise ValueError(f"missing management policy: {MANAGEMENT_POLICY_NAME}")


def settle_with_entry_delay(
    *,
    selected_rows: pd.DataFrame,
    simulators: dict[str, Any],
    policy,
    delay_seconds: int,
) -> list[dict[str, Any]]:
    records = []
    for row in selected_rows.sort_values("decision_time").itertuples(index=False):
        decision_time = pd.Timestamp(row.decision_time).tz_convert("UTC")
        side = OrderSide.BUY if row.side_label == "LONG" else OrderSide.SELL
        trade = simulators[str(row.date)].simulate(
            decision_time_ms=int(decision_time.timestamp() * 1_000) + delay_seconds * 1_000,
            side=side,
            horizon_seconds=HORIZON_SECONDS,
            probability=float(row.edge),
            policy=policy,
        )
        records.append(
            {
                "entry_delay_seconds": delay_seconds,
                "date": str(row.date),
                "decision_time": decision_time.isoformat(),
                "side": str(row.side_label),
                "edge": float(row.edge),
                "prediction": float(row.prediction),
                "regime_gate_value": float(getattr(row, REGIME_GATE_FEATURE)),
                "actual_bps": float(row.actual_bps),
                "terminal_net_bps": float(row.terminal_net_bps),
                "exit_reason": trade.exit_reason.value,
                "gross_bps": trade.gross_bps,
                "expected_net_bps": trade.expected_net_bps,
                "stress_net_bps": trade.stress_net_bps,
                "maximum_favorable_bps_until_exit": trade.maximum_favorable_bps,
                "maximum_adverse_bps_until_exit": trade.maximum_adverse_bps,
            }
        )
    return records


def summarize_delays(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    summaries = []
    for delay_seconds, rows in frame.groupby("entry_delay_seconds", sort=True):
        daily = summarize_daily(rows)
        summaries.append(
            {
                "entry_delay_seconds": int(delay_seconds),
                **summarize_rows(rows),
                "positive_day_count": int(sum(day["stress_total_bps"] > 0 for day in daily)),
                "day_count": int(len(daily)),
                "min_daily_trade_count": min((day["trade_count"] for day in daily), default=0),
                "worst_day_stress_total_bps": min(
                    (day["stress_total_bps"] for day in daily),
                    default=0.0,
                ),
                "daily": daily,
            }
        )
    return summaries


def summarize_daily(rows: pd.DataFrame) -> list[dict[str, Any]]:
    daily = []
    for date, date_rows in rows.groupby("date", sort=True):
        summary = summarize_rows(date_rows)
        daily.append(
            {
                "date": str(date),
                "trade_count": int(summary["trade_count"]),
                "stress_total_bps": float(summary["stress_total_bps"]),
            }
        )
    return daily


def fast187_status(delay_summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "timestamp_causality_audit"]
    zero_delay = summary_for_delay(delay_summaries, 0)
    close_delay = summary_for_delay(delay_summaries, 60)
    if zero_delay is None or close_delay is None:
        return "FAST187_MISSING_DELAY_SUMMARY", reasons + ["zero_or_close_delay_missing"]
    if (
        float(zero_delay["stress_total_bps"]) > 0
        and int(zero_delay["positive_day_count"]) == len(POST_TRAINING_DATES)
        and float(close_delay["stress_total_bps"]) <= 0
    ):
        return (
            "FAST187_REJECTS_FAST185_UNDER_CAUSAL_CLOSE_DELAY",
            reasons
            + [
                "zero_delay_survives",
                "sixty_second_close_delay_fails",
                "feature_timestamp_or_entry_alignment_must_be_fixed_before_strategy_approval",
            ],
        )
    if float(close_delay["stress_total_bps"]) > 0 and int(close_delay["positive_day_count"]) == len(POST_TRAINING_DATES):
        return (
            "FAST187_FAST185_SURVIVES_CAUSAL_CLOSE_DELAY_REQUIRES_FURTHER_REVIEW",
            reasons + ["sixty_second_close_delay_survives", "still_not_approved_strategy"],
        )
    return "FAST187_FAST185_CAUSAL_DELAY_INCONCLUSIVE_OR_FAILED", reasons + ["causal_delay_not_robust"]


def summary_for_delay(summaries: list[dict[str, Any]], delay_seconds: int) -> dict[str, Any] | None:
    for summary in summaries:
        if int(summary["entry_delay_seconds"]) == delay_seconds:
            return summary
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
        "delay_summaries": [
            {
                "entry_delay_seconds": item["entry_delay_seconds"],
                "trade_count": item["trade_count"],
                "stress_total_bps": item["stress_total_bps"],
                "positive_day_count": item["positive_day_count"],
                "worst_day_stress_total_bps": item["worst_day_stress_total_bps"],
            }
            for item in report["summary"]["delay_summaries"]
        ],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

"""Fast184 robust session-filter historical audit.

Fast182 found that wider edge gates plus low-cost trailing could be profitable
on the original validation/extension blocks. A post-training check then showed
the broad micro_8_3 candidate still had bad daily clusters. Fast184 freezes the
diagnostic question before any new prospective evidence is counted:

Can a simple UTC session filter make the Fast177-style ETHUSDT edge signal
daily-robust across every fully historical post-training day available locally?

This script is historical research only. Any survivor requires a fresh
future-only preregistration before it can count as evidence toward a strategy.
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
    DECISION_STRIDE_MINUTES,
    DEFAULT_BTC_TRADE_ROOT,
    DEFAULT_ETH_TRADE_ROOT,
    DEFAULT_SOL_TRADE_ROOT,
    TRAIN_BLOCKS,
    build_all_feature_rows,
    build_date_features,
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


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast184-robust-session-filter-historical-audit"
)
TARGET_SYMBOL = "ETHUSDT"
POST_TRAINING_DATES = (
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-18",
    "2026-09-19",
    "2026-09-20",
    "2026-09-21",
    "2026-09-22",
    "2026-09-23",
)
EDGE_THRESHOLDS_BPS = (5.0, 7.5, 10.0, 12.5, 15.0)
SIDE_FILTERS = ("ALL", "LONG", "SHORT")
MIN_DAILY_TRADES_FOR_ROBUST_SURVIVOR = 2
MIN_POSITIVE_DAYS_FOR_ROBUST_SURVIVOR = 8
HOUR_MIN = 0
HOUR_MAX = 23

SELECTED_FILTERS: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "edge7p5_micro5_all_utc05_16",
        "role": "selected_survivor",
        "edge_threshold_bps": 7.5,
        "management_policy": "micro_5_2",
        "side_filter": "ALL",
        "start_hour_utc": 5,
        "end_hour_utc": 16,
    },
    {
        "candidate_id": "edge10_micro5_all_utc01_11",
        "role": "secondary_perfect_day_survivor",
        "edge_threshold_bps": 10.0,
        "management_policy": "micro_5_2",
        "side_filter": "ALL",
        "start_hour_utc": 1,
        "end_hour_utc": 11,
    },
    {
        "candidate_id": "edge7p5_micro8_all_utc00_23",
        "role": "broad_fast183_benchmark",
        "edge_threshold_bps": 7.5,
        "management_policy": "micro_8_3",
        "side_filter": "ALL",
        "start_hour_utc": 0,
        "end_hour_utc": 23,
    },
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    report = run_fast184_robust_session_filter_historical_audit(
        trade_roots={
            "BTCUSDT": args.btc_trade_root,
            "ETHUSDT": args.eth_trade_root,
            "SOLUSDT": args.sol_trade_root,
        },
        output_root=args.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast184_robust_session_filter_historical_audit(
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
    simulators = {
        date: simulator_for_date(trade_roots[TARGET_SYMBOL], date)
        for date in sorted(scored_rows["date"].astype(str).unique())
    }
    candidate_settlements = settle_selected_filters(
        scored_rows=scored_rows,
        simulators=simulators,
        selected_filters=SELECTED_FILTERS,
    )
    candidate_summaries = summarize_filter_records(candidate_settlements)

    search_settlements = settle_search_grid(scored_rows=scored_rows, simulators=simulators)
    robust_survivors = summarize_robust_search(search_settlements)
    status, status_reasons = fast184_status(candidate_summaries, robust_survivors)

    output_root.mkdir(parents=True, exist_ok=True)
    candidate_settlements_path = output_root / "selected-filter-settlements.jsonl"
    search_survivors_path = output_root / "robust-filter-survivors.json"
    scored_rows_path = output_root / "scored-post-training-rows.parquet"
    summary_path = output_root / "summary.json"
    write_jsonl(candidate_settlements_path, candidate_settlements)
    write_json(search_survivors_path, robust_survivors)
    scored_rows.to_parquet(scored_rows_path, index=False)

    perfect_day_survivors = [
        item for item in robust_survivors if int(item["positive_day_count"]) == len(POST_TRAINING_DATES)
    ]
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "target_symbol": TARGET_SYMBOL,
        "train_blocks": list(TRAIN_BLOCKS),
        "post_training_dates": list(POST_TRAINING_DATES),
        "horizon_seconds": HORIZON_SECONDS,
        "decision_stride_minutes": DECISION_STRIDE_MINUTES,
        "latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "edge_thresholds_bps": list(EDGE_THRESHOLDS_BPS),
        "side_filters": list(SIDE_FILTERS),
        "hour_window_min_utc": HOUR_MIN,
        "hour_window_max_utc": HOUR_MAX,
        "min_daily_trades_for_robust_survivor": MIN_DAILY_TRADES_FOR_ROBUST_SURVIVOR,
        "min_positive_days_for_robust_survivor": MIN_POSITIVE_DAYS_FOR_ROBUST_SURVIVOR,
        "feature_count": len(feature_columns),
        "training_row_count": int(len(training_rows)),
        "post_training_row_count": int(len(post_training_rows)),
        "scored_row_count": int(len(scored_rows)),
        "selected_filter_settlement_count": len(candidate_settlements),
        "selected_filter_summaries": candidate_summaries,
        "robust_survivor_count": len(robust_survivors),
        "perfect_day_survivor_count": len(perfect_day_survivors),
        "top_robust_survivors": robust_survivors[:10],
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
            "fast184_is_historical_filter_audit_not_strategy_approval",
            "session_filter_was_selected_on_historical_post_training_days",
            "survivor_requires_fresh_future_only_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast184_robust_session_filter_historical_audit",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "selected_filter_settlements": {
                "path": str(candidate_settlements_path),
                "sha256": sha256_file(candidate_settlements_path),
            },
            "robust_filter_survivors": {
                "path": str(search_survivors_path),
                "sha256": sha256_file(search_survivors_path),
            },
            "scored_post_training_rows": {
                "path": str(scored_rows_path),
                "sha256": sha256_file(scored_rows_path),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
        "trade_roots": {symbol: str(path) for symbol, path in sorted(trade_roots.items())},
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def build_post_training_rows(*, trade_roots: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for utc_date in POST_TRAINING_DATES:
        frame = build_date_features(trade_roots, utc_date)
        frame["block"] = "post_training_full_day"
        frame["date"] = utc_date
        frames.append(frame.loc[frame["minute_sequence"].mod(DECISION_STRIDE_MINUTES).eq(0)])
    return pd.concat(frames, ignore_index=True).sort_values("decision_time").reset_index(drop=True)


def score_post_training_rows(
    post_training_rows: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> pd.DataFrame:
    shift_rows = int(HORIZON_SECONDS / 60 / DECISION_STRIDE_MINUTES)
    scored = post_training_rows.copy()
    scored["future_price"] = scored.groupby("date")[f"{TARGET_SYMBOL}_price"].shift(-shift_rows)
    scored["actual_bps"] = (
        scored["future_price"] / scored[f"{TARGET_SYMBOL}_price"] - 1.0
    ) * 10000.0
    scored = scored.dropna(subset=["actual_bps"]).copy()
    scored["prediction"] = model.predict(scored[feature_columns])
    scored["edge"] = scored["prediction"].abs()
    scored["side_label"] = np.where(scored["prediction"] > 0, "LONG", "SHORT")
    scored["side_mult"] = np.sign(scored["prediction"])
    scored["terminal_net_bps"] = scored["actual_bps"] * scored["side_mult"] - STRESS_COST_BPS
    scored["decision_hour_utc"] = scored["decision_time"].dt.tz_convert("UTC").dt.hour
    return scored


def settle_selected_filters(
    *,
    scored_rows: pd.DataFrame,
    simulators: dict[str, Any],
    selected_filters: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    records = []
    for filter_spec in selected_filters:
        selected = select_filter_rows(scored_rows, filter_spec)
        records.extend(settle_rows(selected, filter_spec=filter_spec, simulators=simulators))
    return records


def settle_search_grid(*, scored_rows: pd.DataFrame, simulators: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    policy_names = tuple(policy.name for policy in LOW_COST_TRAILING_POLICIES)
    for edge_threshold in EDGE_THRESHOLDS_BPS:
        for management_policy in policy_names:
            filter_spec = {
                "candidate_id": f"edge{edge_threshold:g}_{management_policy}_all_hours",
                "role": "search_grid",
                "edge_threshold_bps": edge_threshold,
                "management_policy": management_policy,
                "side_filter": "ALL",
                "start_hour_utc": HOUR_MIN,
                "end_hour_utc": HOUR_MAX,
            }
            selected = select_filter_rows(scored_rows, filter_spec)
            records.extend(settle_rows(selected, filter_spec=filter_spec, simulators=simulators))
    return records


def select_filter_rows(scored_rows: pd.DataFrame, filter_spec: dict[str, Any]) -> pd.DataFrame:
    selected = scored_rows.loc[
        scored_rows["edge"].ge(float(filter_spec["edge_threshold_bps"]))
        & scored_rows[REGIME_GATE_FEATURE].ge(REGIME_GATE_THRESHOLD)
        & scored_rows["side_mult"].ne(0)
    ].copy()
    side_filter = str(filter_spec["side_filter"])
    if side_filter != "ALL":
        selected = selected.loc[selected["side_label"].eq(side_filter)].copy()
    return selected.loc[
        selected["decision_hour_utc"].between(
            int(filter_spec["start_hour_utc"]),
            int(filter_spec["end_hour_utc"]),
        )
    ].copy()


def settle_rows(
    selected: pd.DataFrame,
    *,
    filter_spec: dict[str, Any],
    simulators: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    policy = management_policy(str(filter_spec["management_policy"]))
    for row in selected.sort_values("decision_time").itertuples(index=False):
        decision_time = pd.Timestamp(row.decision_time).tz_convert("UTC")
        side = OrderSide.BUY if row.side_label == "LONG" else OrderSide.SELL
        trade = simulators[str(row.date)].simulate(
            decision_time_ms=int(decision_time.timestamp() * 1_000),
            side=side,
            horizon_seconds=HORIZON_SECONDS,
            probability=float(row.edge),
            policy=policy,
        )
        records.append(
            {
                "candidate_id": str(filter_spec["candidate_id"]),
                "role": str(filter_spec["role"]),
                "edge_threshold_bps": float(filter_spec["edge_threshold_bps"]),
                "management_policy": str(filter_spec["management_policy"]),
                "side_filter": str(filter_spec["side_filter"]),
                "start_hour_utc": int(filter_spec["start_hour_utc"]),
                "end_hour_utc": int(filter_spec["end_hour_utc"]),
                "date": str(row.date),
                "decision_time": decision_time.isoformat(),
                "decision_hour_utc": int(decision_time.hour),
                "side": str(row.side_label),
                "prediction": float(row.prediction),
                "edge": float(row.edge),
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


def management_policy(policy_name: str):
    for policy in LOW_COST_TRAILING_POLICIES:
        if policy.name == policy_name:
            return policy
    raise ValueError(f"missing management policy: {policy_name}")


def summarize_filter_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    summaries = []
    for candidate_id, candidate_rows in frame.groupby("candidate_id", sort=False):
        daily = summarize_daily(candidate_rows)
        overall = summarize_rows(candidate_rows)
        summaries.append(
            {
                "candidate_id": str(candidate_id),
                "role": str(candidate_rows["role"].iloc[0]),
                "edge_threshold_bps": float(candidate_rows["edge_threshold_bps"].iloc[0]),
                "management_policy": str(candidate_rows["management_policy"].iloc[0]),
                "side_filter": str(candidate_rows["side_filter"].iloc[0]),
                "start_hour_utc": int(candidate_rows["start_hour_utc"].iloc[0]),
                "end_hour_utc": int(candidate_rows["end_hour_utc"].iloc[0]),
                **overall,
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
    summaries = []
    for date, date_rows in rows.groupby("date", sort=True):
        payload = summarize_rows(date_rows)
        summaries.append(
            {
                "date": str(date),
                "trade_count": int(payload["trade_count"]),
                "stress_total_bps": float(payload["stress_total_bps"]),
                "stress_mean_bps": float(payload["stress_mean_bps"]),
                "stress_win_rate": float(payload["stress_win_rate"]),
            }
        )
    return summaries


def summarize_robust_search(search_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(search_records)
    if frame.empty:
        return []
    survivors = []
    for (edge_threshold, policy), policy_rows in frame.groupby(
        ["edge_threshold_bps", "management_policy"],
        sort=True,
    ):
        for side_filter in SIDE_FILTERS:
            side_rows = policy_rows if side_filter == "ALL" else policy_rows.loc[policy_rows["side"].eq(side_filter)]
            if side_rows.empty:
                continue
            for start_hour in range(HOUR_MIN, HOUR_MAX + 1):
                for end_hour in range(start_hour, HOUR_MAX + 1):
                    window_rows = side_rows.loc[
                        side_rows["decision_hour_utc"].between(start_hour, end_hour)
                    ]
                    if window_rows.empty:
                        continue
                    daily = summarize_daily(window_rows)
                    min_daily_trade_count = min((day["trade_count"] for day in daily), default=0)
                    positive_day_count = int(sum(day["stress_total_bps"] > 0 for day in daily))
                    if len(daily) != len(POST_TRAINING_DATES):
                        continue
                    if min_daily_trade_count < MIN_DAILY_TRADES_FOR_ROBUST_SURVIVOR:
                        continue
                    if positive_day_count < MIN_POSITIVE_DAYS_FOR_ROBUST_SURVIVOR:
                        continue
                    overall = summarize_rows(window_rows)
                    survivors.append(
                        {
                            "edge_threshold_bps": float(edge_threshold),
                            "management_policy": str(policy),
                            "side_filter": side_filter,
                            "start_hour_utc": int(start_hour),
                            "end_hour_utc": int(end_hour),
                            **overall,
                            "positive_day_count": positive_day_count,
                            "day_count": int(len(daily)),
                            "min_daily_trade_count": int(min_daily_trade_count),
                            "worst_day_stress_total_bps": float(
                                min(day["stress_total_bps"] for day in daily)
                            ),
                            "daily": daily,
                        }
                    )
    return sorted(
        survivors,
        key=lambda item: (
            int(item["positive_day_count"]),
            float(item["worst_day_stress_total_bps"]),
            float(item["stress_total_bps"]),
            int(item["trade_count"]),
        ),
        reverse=True,
    )


def fast184_status(
    candidate_summaries: list[dict[str, Any]],
    robust_survivors: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "historical_filter_audit"]
    selected = next(
        (
            item
            for item in candidate_summaries
            if item["candidate_id"] == "edge7p5_micro5_all_utc05_16"
        ),
        None,
    )
    perfect_day_survivors = [
        item for item in robust_survivors if int(item["positive_day_count"]) == len(POST_TRAINING_DATES)
    ]
    if selected is None:
        return "FAST184_SELECTED_FILTER_MISSING", reasons + ["selected_survivor_not_settled"]
    if (
        int(selected["positive_day_count"]) == len(POST_TRAINING_DATES)
        and float(selected["worst_day_stress_total_bps"]) > 0
        and perfect_day_survivors
    ):
        return (
            "FAST184_ROBUST_SESSION_FILTER_HISTORICAL_SURVIVOR_REQUIRES_PREREGISTRATION",
            reasons
            + [
                "selected_filter_positive_on_9_of_9_post_training_days",
                f"perfect_day_survivors:{len(perfect_day_survivors)}",
                "fresh_future_preregistration_required",
            ],
        )
    return (
        "FAST184_ROBUST_SESSION_FILTER_NO_APPROVABLE_HISTORICAL_SURVIVOR",
        reasons + ["selected_filter_not_robust_enough"],
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = next(
        item
        for item in report["summary"]["selected_filter_summaries"]
        if item["candidate_id"] == "edge7p5_micro5_all_utc05_16"
    )
    return {
        "research_only": report["research_only"],
        "approved_strategy": report["approved_strategy"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "selected_filter": {
            "candidate_id": selected["candidate_id"],
            "trade_count": selected["trade_count"],
            "stress_total_bps": selected["stress_total_bps"],
            "positive_day_count": selected["positive_day_count"],
            "day_count": selected["day_count"],
            "worst_day_stress_total_bps": selected["worst_day_stress_total_bps"],
        },
        "perfect_day_survivor_count": report["summary"]["perfect_day_survivor_count"],
        "robust_survivor_count": report["summary"]["robust_survivor_count"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

"""Fast189 causal close-time feature rebuild smoke.

Fast187 showed the prior Fast185 candidate fails when we wait for the feature
minute to be observable. Fast188 showed that a crude 5-minute feature lag kills
the simple edge family. Fast189 rebuilds the target with a stricter but less
overconservative causal contract:

* one-minute feature bins are considered observable only at the next minute;
* the model label is measured from that executable decision time;
* trade simulation also starts from that executable decision time.

This is research-only. It does not approve a strategy or send orders.
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
    build_date_features,
    leadlag_model,
)
from scripts.research.run_fast178_fast177_edge2_observation import feature_columns_from_rows  # noqa: E402
from scripts.research.run_fast179_fast177_path_management_audit import (  # noqa: E402
    datetime_series_to_epoch_ms,
    sha256_file,
    write_jsonl,
)
from scripts.research.run_fast181_edge2_micro3_historical_management_audit import (  # noqa: E402
    HORIZON_SECONDS,
    LATENCY_MS,
    PROFIT_COST_BPS,
    REGIME_GATE_FEATURE,
    REGIME_GATE_THRESHOLD,
    STRESS_COST_BPS,
    summarize_rows,
)
from scripts.research.run_fast184_robust_session_filter_historical_audit import (  # noqa: E402
    POST_TRAINING_DATES,
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
    / "fast189-causal-close-feature-rebuild-smoke"
)
TARGET_SYMBOL = "ETHUSDT"
FEATURE_CLOSE_DELAY_SECONDS = 60
EDGE_THRESHOLDS_BPS = (1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0)
SIDE_FILTERS = ("ALL", "LONG", "SHORT")
MIN_DAILY_TRADES_FOR_SURVIVOR = 2
MIN_POSITIVE_DAYS_FOR_SURVIVOR = 8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    report = run_fast189_causal_close_feature_rebuild_smoke(
        trade_roots={
            "BTCUSDT": args.btc_trade_root,
            "ETHUSDT": args.eth_trade_root,
            "SOLUSDT": args.sol_trade_root,
        },
        output_root=args.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast189_causal_close_feature_rebuild_smoke(
    *,
    trade_roots: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    training_universe = build_all_feature_rows(trade_roots)
    feature_columns = feature_columns_from_rows(training_universe)
    training_rows = build_causal_label_rows(
        rows=training_universe.loc[training_universe["block"].isin(TRAIN_BLOCKS)].copy(),
        eth_trade_root=trade_roots[TARGET_SYMBOL],
        feature_columns=feature_columns,
    )
    model = leadlag_model()
    model.fit(training_rows[feature_columns], training_rows["causal_actual_bps"])

    post_training_rows = build_post_training_feature_rows(trade_roots)
    scored_rows = score_causal_rows(
        rows=post_training_rows,
        model=model,
        feature_columns=feature_columns,
        eth_trade_root=trade_roots[TARGET_SYMBOL],
    )
    settlements = settle_search_universe(scored_rows=scored_rows, eth_trade_root=trade_roots[TARGET_SYMBOL])
    robust_survivors = summarize_robust_search(settlements)
    status, status_reasons = fast189_status(robust_survivors)

    output_root.mkdir(parents=True, exist_ok=True)
    scored_path = output_root / "causal-scored-rows.parquet"
    settlements_path = output_root / "causal-settlements.jsonl"
    robust_survivors_path = output_root / "robust-survivors.json"
    summary_path = output_root / "summary.json"
    scored_rows.to_parquet(scored_path, index=False)
    write_jsonl(settlements_path, settlements)
    write_json(robust_survivors_path, robust_survivors)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "target_symbol": TARGET_SYMBOL,
        "train_blocks": list(TRAIN_BLOCKS),
        "post_training_dates": list(POST_TRAINING_DATES),
        "feature_close_delay_seconds": FEATURE_CLOSE_DELAY_SECONDS,
        "horizon_seconds": HORIZON_SECONDS,
        "latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "session_start_hour_utc": SESSION_START_HOUR_UTC,
        "session_end_hour_utc": SESSION_END_HOUR_UTC,
        "edge_thresholds_bps": list(EDGE_THRESHOLDS_BPS),
        "side_filters": list(SIDE_FILTERS),
        "feature_count": len(feature_columns),
        "training_row_count": len(training_rows),
        "scored_row_count": len(scored_rows),
        "settlement_count": len(settlements),
        "robust_survivor_count": len(robust_survivors),
        "perfect_day_survivor_count": sum(
            int(item["positive_day_count"]) == len(POST_TRAINING_DATES)
            and float(item["worst_day_stress_total_bps"]) > 0
            for item in robust_survivors
        ),
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
            "fast189_is_causal_close_feature_rebuild_smoke_only",
            "survivors_require_fresh_future_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast189_causal_close_feature_rebuild_smoke",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "causal_scored_rows": {"path": str(scored_path), "sha256": sha256_file(scored_path)},
            "causal_settlements": {"path": str(settlements_path), "sha256": sha256_file(settlements_path)},
            "robust_survivors": {"path": str(robust_survivors_path), "sha256": sha256_file(robust_survivors_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def build_post_training_feature_rows(trade_roots: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for utc_date in POST_TRAINING_DATES:
        frame = build_date_features(trade_roots, utc_date)
        frame["date"] = utc_date
        frame["block"] = "post_training_causal"
        frames.append(frame.loc[frame["minute_sequence"].mod(5).eq(0)])
    return pd.concat(frames, ignore_index=True).sort_values("decision_time").reset_index(drop=True)


def build_causal_label_rows(
    *,
    rows: pd.DataFrame,
    eth_trade_root: Path,
    feature_columns: list[str],
) -> pd.DataFrame:
    labelled = rows.sort_values(["date", "decision_time"]).copy()
    labelled["feature_time"] = labelled["decision_time"]
    labelled["decision_time"] = labelled["feature_time"] + pd.Timedelta(seconds=FEATURE_CLOSE_DELAY_SECONDS)
    labelled["date"] = labelled["decision_time"].dt.strftime("%Y-%m-%d")
    labelled["causal_actual_bps"] = causal_signed_returns(labelled, eth_trade_root=eth_trade_root)
    return labelled.dropna(subset=[*feature_columns, "causal_actual_bps"]).reset_index(drop=True)


def score_causal_rows(
    *,
    rows: pd.DataFrame,
    model,
    feature_columns: list[str],
    eth_trade_root: Path,
) -> pd.DataFrame:
    scored = build_causal_label_rows(rows=rows, eth_trade_root=eth_trade_root, feature_columns=feature_columns)
    scored["prediction"] = model.predict(scored[feature_columns])
    scored["edge"] = scored["prediction"].abs()
    scored["side_label"] = np.where(scored["prediction"] > 0, "LONG", "SHORT")
    scored["side_mult"] = np.sign(scored["prediction"])
    scored["terminal_net_bps"] = scored["causal_actual_bps"] * scored["side_mult"] - STRESS_COST_BPS
    scored["decision_hour_utc"] = scored["decision_time"].dt.tz_convert("UTC").dt.hour
    return scored


def causal_signed_returns(rows: pd.DataFrame, *, eth_trade_root: Path) -> pd.Series:
    values = pd.Series(np.nan, index=rows.index, dtype=float)
    for utc_date, date_rows in rows.groupby("date", sort=True):
        partition = eth_trade_root / f"date={utc_date}" / "events.parquet"
        if not partition.exists():
            continue
        events = pd.read_parquet(partition, columns=["event_time", "price"]).sort_values("event_time")
        event_times_ms = datetime_series_to_epoch_ms(events["event_time"])
        prices = events["price"].to_numpy(dtype=np.float64)
        for index, row in date_rows.iterrows():
            decision_ms = int(pd.Timestamp(row["decision_time"]).timestamp() * 1_000)
            entry_index = int(np.searchsorted(event_times_ms, decision_ms + LATENCY_MS, side="left"))
            end_index = int(np.searchsorted(event_times_ms, decision_ms + HORIZON_SECONDS * 1_000, side="right"))
            if entry_index >= len(event_times_ms) or entry_index >= end_index:
                continue
            entry_price = float(prices[entry_index])
            terminal_price = float(prices[end_index - 1])
            values.loc[index] = float(np.log(terminal_price / entry_price) * 10_000)
    return values


def settle_search_universe(*, scored_rows: pd.DataFrame, eth_trade_root: Path) -> list[dict[str, Any]]:
    simulators = {date: simulator_for_date(eth_trade_root, date) for date in sorted(scored_rows["date"].unique())}
    records = []
    for policy in LOW_COST_TRAILING_POLICIES:
        selected = scored_rows.loc[
            scored_rows[REGIME_GATE_FEATURE].ge(REGIME_GATE_THRESHOLD) & scored_rows["side_mult"].ne(0)
        ].copy()
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
                    "management_policy": policy.name,
                    "date": str(row.date),
                    "feature_time": pd.Timestamp(row.feature_time).tz_convert("UTC").isoformat(),
                    "decision_time": decision_time.isoformat(),
                    "decision_hour_utc": int(decision_time.hour),
                    "side": str(row.side_label),
                    "edge": float(row.edge),
                    "prediction": float(row.prediction),
                    "regime_gate_value": float(getattr(row, REGIME_GATE_FEATURE)),
                    "causal_actual_bps": float(row.causal_actual_bps),
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


def summarize_robust_search(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    survivors = []
    for (policy, threshold), threshold_rows in threshold_groups(frame):
        for side_filter in SIDE_FILTERS:
            side_rows = threshold_rows if side_filter == "ALL" else threshold_rows.loc[threshold_rows["side"].eq(side_filter)]
            if side_rows.empty:
                continue
            for start_hour in range(SESSION_START_HOUR_UTC, SESSION_END_HOUR_UTC + 1):
                for end_hour in range(start_hour, SESSION_END_HOUR_UTC + 1):
                    rows = side_rows.loc[side_rows["decision_hour_utc"].between(start_hour, end_hour)]
                    if rows.empty:
                        continue
                    daily = summarize_daily(rows)
                    if len(daily) != len(POST_TRAINING_DATES):
                        continue
                    min_daily_trade_count = min((day["trade_count"] for day in daily), default=0)
                    positive_day_count = int(sum(day["stress_total_bps"] > 0 for day in daily))
                    if min_daily_trade_count < MIN_DAILY_TRADES_FOR_SURVIVOR:
                        continue
                    if positive_day_count < MIN_POSITIVE_DAYS_FOR_SURVIVOR:
                        continue
                    overall = summarize_rows(rows)
                    survivors.append(
                        {
                            "edge_threshold_bps": float(threshold),
                            "management_policy": str(policy),
                            "side_filter": side_filter,
                            "start_hour_utc": int(start_hour),
                            "end_hour_utc": int(end_hour),
                            **overall,
                            "positive_day_count": positive_day_count,
                            "day_count": int(len(daily)),
                            "min_daily_trade_count": int(min_daily_trade_count),
                            "worst_day_stress_total_bps": float(min(day["stress_total_bps"] for day in daily)),
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


def threshold_groups(frame: pd.DataFrame):
    for policy, policy_rows in frame.groupby("management_policy", sort=True):
        for threshold in EDGE_THRESHOLDS_BPS:
            yield (policy, threshold), policy_rows.loc[policy_rows["edge"].ge(threshold)]


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


def simulator_for_date(eth_trade_root: Path, utc_date: str):
    from scripts.research.run_fast181_edge2_micro3_historical_management_audit import simulator_for_date as build_simulator

    return build_simulator(eth_trade_root, utc_date)


def fast189_status(robust_survivors: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "causal_close_feature_smoke"]
    perfect = [
        item
        for item in robust_survivors
        if int(item["positive_day_count"]) == len(POST_TRAINING_DATES)
        and float(item["worst_day_stress_total_bps"]) > 0
    ]
    if perfect:
        best = perfect[0]
        return (
            "FAST189_CAUSAL_CLOSE_FEATURE_HISTORICAL_SURVIVOR_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"perfect_day_survivors:{len(perfect)}",
                f"best_policy:{best['management_policy']}",
                f"best_edge:{best['edge_threshold_bps']}",
                f"best_hours:{best['start_hour_utc']}-{best['end_hour_utc']}",
                "fresh_future_preregistration_required",
            ],
        )
    if robust_survivors:
        return (
            "FAST189_CAUSAL_CLOSE_FEATURE_HAS_WEAK_HISTORICAL_SURVIVORS",
            reasons + [f"weak_survivors:{len(robust_survivors)}", "not_enough_for_preregistration"],
        )
    return "FAST189_CAUSAL_CLOSE_FEATURE_NO_SIMPLE_SURVIVORS", reasons + ["no_robust_survivors_found"]


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
        "robust_survivor_count": report["summary"]["robust_survivor_count"],
        "perfect_day_survivor_count": report["summary"]["perfect_day_survivor_count"],
        "top_robust_survivors": report["summary"]["top_robust_survivors"][:5],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

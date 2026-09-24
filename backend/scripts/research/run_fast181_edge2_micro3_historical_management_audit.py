"""Fast181 historical audit for Fast178-style edge gates plus micro_3_1.

Fast179/180 explored whether short breakeven/trailing management can extract
the first favorable move from the Fast178 ETHUSDT edge-2 signal. Fast181 checks
that same idea on the already-defined Fast174 validation and extension blocks,
using the same training blocks and no prospective rows from today.

This is a historical diagnostic only. Positive results would require a fresh
future-only preregistration; negative results reject the management shortcut.
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
from app.services.backtest.trailing_portfolio import (  # noqa: E402
    HistoricalTrailingSimulator,
    LOW_COST_TRAILING_POLICIES,
)
from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import (  # noqa: E402
    DECISION_STRIDE_MINUTES,
    DEFAULT_BTC_TRADE_ROOT,
    DEFAULT_ETH_TRADE_ROOT,
    DEFAULT_SOL_TRADE_ROOT,
    EXTENSION_BLOCK,
    TRAIN_BLOCKS,
    VALIDATION_BLOCK,
    build_all_feature_rows,
    leadlag_model,
)
from scripts.research.run_fast178_fast177_edge2_observation import (  # noqa: E402
    build_training_label_rows,
    feature_columns_from_rows,
)
from scripts.research.run_fast179_fast177_path_management_audit import (  # noqa: E402
    datetime_series_to_epoch_ms,
    sha256_file,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast181-edge2-micro3-historical-management-audit"
)
TARGET_SYMBOL = "ETHUSDT"
HORIZON_SECONDS = 900
REGIME_GATE_FEATURE = "ETHUSDT_qv_30m"
REGIME_GATE_THRESHOLD = 100000000.0
EDGE_THRESHOLDS_BPS = (2.0, 7.5, 10.0)
MANAGEMENT_POLICY_NAME = "micro_3_1"
LATENCY_MS = 100
PROFIT_COST_BPS = 1.0
STRESS_COST_BPS = 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run_fast181_historical_management_audit(
        trade_roots={
            "BTCUSDT": arguments.btc_trade_root,
            "ETHUSDT": arguments.eth_trade_root,
            "SOLUSDT": arguments.sol_trade_root,
        },
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast181_historical_management_audit(
    *,
    trade_roots: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    rows = build_all_feature_rows(trade_roots)
    feature_columns = feature_columns_from_rows(rows)
    training_rows = build_training_label_rows(rows)
    model = leadlag_model()
    model.fit(training_rows[feature_columns], training_rows["actual_bps"])

    scored_rows = score_validation_extension_rows(rows, model, feature_columns)
    policy = management_policy()
    settlements = []
    for threshold in EDGE_THRESHOLDS_BPS:
        selected = scored_rows.loc[
            scored_rows["edge"].ge(threshold)
            & scored_rows[REGIME_GATE_FEATURE].ge(REGIME_GATE_THRESHOLD)
            & scored_rows["side_mult"].ne(0)
        ].copy()
        settlements.extend(
            settle_selected_rows(
                selected,
                threshold=threshold,
                eth_trade_root=trade_roots["ETHUSDT"],
                policy=policy,
            )
        )

    threshold_summaries = summarize_by_threshold(settlements)
    status, status_reasons = fast181_status(threshold_summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    settlements_path = output_root / "managed-settlements.jsonl"
    summary_path = output_root / "summary.json"
    write_jsonl(settlements_path, settlements)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "target_symbol": TARGET_SYMBOL,
        "horizon_seconds": HORIZON_SECONDS,
        "train_blocks": list(TRAIN_BLOCKS),
        "validation_block": VALIDATION_BLOCK,
        "extension_block": EXTENSION_BLOCK,
        "edge_thresholds_bps": list(EDGE_THRESHOLDS_BPS),
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "management_policy": MANAGEMENT_POLICY_NAME,
        "latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "feature_count": len(feature_columns),
        "training_row_count": int(len(training_rows)),
        "scored_row_count": int(len(scored_rows)),
        "settlement_count": len(settlements),
        "threshold_summaries": threshold_summaries,
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
            "fast181_is_historical_management_audit_not_strategy_approval",
            "uses_fast174_validation_and_extension_blocks",
            "positive_result_would_require_fresh_future_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast181_edge2_micro3_historical_management_audit",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "managed_settlements": {
                "path": str(settlements_path),
                "sha256": sha256_file(settlements_path),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
        "trade_roots": {symbol: str(path) for symbol, path in sorted(trade_roots.items())},
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def score_validation_extension_rows(
    rows: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> pd.DataFrame:
    shift_rows = int(HORIZON_SECONDS / 60 / DECISION_STRIDE_MINUTES)
    scored = rows.loc[rows["block"].isin([VALIDATION_BLOCK, EXTENSION_BLOCK])].copy()
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
    return scored


def settle_selected_rows(
    selected: pd.DataFrame,
    *,
    threshold: float,
    eth_trade_root: Path,
    policy,
) -> list[dict[str, Any]]:
    records = []
    simulators = {
        date: simulator_for_date(eth_trade_root, date)
        for date in sorted(selected["date"].astype(str).unique())
    }
    for row in selected.sort_values("decision_time").itertuples(index=False):
        decision_time = pd.Timestamp(row.decision_time).tz_convert("UTC")
        decision_time_ms = int(decision_time.timestamp() * 1_000)
        side = OrderSide.BUY if row.side_label == "LONG" else OrderSide.SELL
        trade = simulators[str(row.date)].simulate(
            decision_time_ms=decision_time_ms,
            side=side,
            horizon_seconds=HORIZON_SECONDS,
            probability=float(row.edge),
            policy=policy,
        )
        records.append(
            {
                "edge_threshold_bps": threshold,
                "block": row.block,
                "date": row.date,
                "decision_time": decision_time.isoformat(),
                "side": row.side_label,
                "prediction": float(row.prediction),
                "edge": float(row.edge),
                "regime_gate_value": float(getattr(row, REGIME_GATE_FEATURE)),
                "actual_bps": float(row.actual_bps),
                "terminal_net_bps": float(row.terminal_net_bps),
                "management_policy": MANAGEMENT_POLICY_NAME,
                "exit_reason": trade.exit_reason.value,
                "gross_bps": trade.gross_bps,
                "expected_net_bps": trade.expected_net_bps,
                "stress_net_bps": trade.stress_net_bps,
                "maximum_favorable_bps_until_exit": trade.maximum_favorable_bps,
                "maximum_adverse_bps_until_exit": trade.maximum_adverse_bps,
            }
        )
    return records


def simulator_for_date(eth_trade_root: Path, utc_date: str) -> HistoricalTrailingSimulator:
    frame = pd.read_parquet(
        eth_trade_root / f"date={utc_date}" / "events.parquet",
        columns=["event_time", "price"],
    ).sort_values("event_time")
    return HistoricalTrailingSimulator(
        datetime_series_to_epoch_ms(frame["event_time"]),
        frame["price"].to_numpy(dtype=np.float64),
        latency_ms=LATENCY_MS,
        expected_round_trip_bps=PROFIT_COST_BPS,
        stress_round_trip_bps=STRESS_COST_BPS,
    )


def management_policy():
    for policy in LOW_COST_TRAILING_POLICIES:
        if policy.name == MANAGEMENT_POLICY_NAME:
            return policy
    raise ValueError(f"missing management policy: {MANAGEMENT_POLICY_NAME}")


def summarize_by_threshold(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    summaries = []
    for threshold, threshold_rows in frame.groupby("edge_threshold_bps", sort=True):
        block_summaries = []
        for block, block_rows in threshold_rows.groupby("block", sort=True):
            block_summaries.append(summarize_rows(block_rows, block=str(block)))
        summaries.append(
            {
                "edge_threshold_bps": float(threshold),
                **summarize_rows(threshold_rows),
                "blocks": block_summaries,
            }
        )
    return summaries


def summarize_rows(rows: pd.DataFrame, *, block: str | None = None) -> dict[str, Any]:
    stress = rows["stress_net_bps"].astype(float)
    terminal = rows["terminal_net_bps"].astype(float)
    payload = {
        "trade_count": int(len(rows)),
        "terminal_total_bps": float(terminal.sum()),
        "terminal_mean_bps": float(terminal.mean()) if len(terminal) else 0.0,
        "stress_total_bps": float(stress.sum()),
        "stress_mean_bps": float(stress.mean()) if len(stress) else 0.0,
        "stress_win_rate": float((stress > 0).mean()) if len(stress) else 0.0,
        "exit_reason_counts": {
            str(reason): int(count)
            for reason, count in rows["exit_reason"].value_counts(sort=False).sort_index().items()
        },
    }
    if block is not None:
        payload["block"] = block
    return payload


def fast181_status(threshold_summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "historical_management_audit"]
    if not threshold_summaries:
        return "FAST181_NO_HISTORICAL_SETTLEMENTS", reasons + ["no_rows_selected"]
    strict = []
    validation_only = []
    for summary in threshold_summaries:
        block_totals = {block["block"]: float(block["stress_total_bps"]) for block in summary["blocks"]}
        if block_totals.get(VALIDATION_BLOCK, 0.0) > 0 and block_totals.get(EXTENSION_BLOCK, 0.0) > 0:
            strict.append(summary)
        elif block_totals.get(VALIDATION_BLOCK, 0.0) > 0:
            validation_only.append(summary)
    if strict:
        return (
            "FAST181_EDGE2_MICRO3_HISTORICAL_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons + [f"strict_positive_thresholds:{len(strict)}", "fresh_future_preregistration_required"],
        )
    if validation_only:
        return (
            "FAST181_EDGE2_MICRO3_FAILED_EXTENSION",
            reasons + [f"validation_positive_thresholds:{len(validation_only)}", "extension_not_positive"],
        )
    return (
        "FAST181_EDGE2_MICRO3_FAILED_VALIDATION_AND_EXTENSION",
        reasons + ["no_threshold_positive_in_validation_and_extension"],
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
        "threshold_summaries": report["summary"]["threshold_summaries"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

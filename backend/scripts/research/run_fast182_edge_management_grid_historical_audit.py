"""Fast182 historical grid audit for ETHUSDT edge gates and low-cost management.

Fast181 showed that the first chosen micro_3_1 management was not enough.
Fast182 widens only the management grid already available in the codebase:
micro_2_1, micro_3_1, micro_5_2, and micro_8_3 across fixed edge thresholds.

This is still historical diagnostics only. Any survivor requires future-only
preregistration before it can count.
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

from app.services.backtest.event_replay import OrderSide  # noqa: E402
from app.services.backtest.trailing_portfolio import LOW_COST_TRAILING_POLICIES  # noqa: E402
from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import (  # noqa: E402
    DEFAULT_BTC_TRADE_ROOT,
    DEFAULT_ETH_TRADE_ROOT,
    DEFAULT_SOL_TRADE_ROOT,
    EXTENSION_BLOCK,
    VALIDATION_BLOCK,
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
    REGIME_GATE_FEATURE,
    REGIME_GATE_THRESHOLD,
    score_validation_extension_rows,
    simulator_for_date,
    summarize_rows,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast182-edge-management-grid-historical-audit"
)
EDGE_THRESHOLDS_BPS = (2.0, 5.0, 7.5, 10.0, 12.5, 15.0)
LATENCY_MS = 100
PROFIT_COST_BPS = 1.0
STRESS_COST_BPS = 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    report = run_fast182_edge_management_grid_historical_audit(
        trade_roots={
            "BTCUSDT": args.btc_trade_root,
            "ETHUSDT": args.eth_trade_root,
            "SOLUSDT": args.sol_trade_root,
        },
        output_root=args.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast182_edge_management_grid_historical_audit(
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

    settlements = []
    simulators = {
        date: simulator_for_date(trade_roots["ETHUSDT"], date)
        for date in sorted(scored_rows["date"].astype(str).unique())
    }
    for threshold in EDGE_THRESHOLDS_BPS:
        selected = scored_rows.loc[
            scored_rows["edge"].ge(threshold)
            & scored_rows[REGIME_GATE_FEATURE].ge(REGIME_GATE_THRESHOLD)
            & scored_rows["side_mult"].ne(0)
        ].copy()
        for policy in LOW_COST_TRAILING_POLICIES:
            settlements.extend(settle_policy_rows(selected, threshold=threshold, policy=policy, simulators=simulators))

    candidate_summaries = summarize_candidates(settlements)
    status, status_reasons = fast182_status(candidate_summaries)
    output_root.mkdir(parents=True, exist_ok=True)
    settlements_path = output_root / "managed-settlements.jsonl"
    summary_path = output_root / "summary.json"
    write_jsonl(settlements_path, settlements)
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "edge_thresholds_bps": list(EDGE_THRESHOLDS_BPS),
        "management_policies": [policy.name for policy in LOW_COST_TRAILING_POLICIES],
        "validation_block": VALIDATION_BLOCK,
        "extension_block": EXTENSION_BLOCK,
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "horizon_seconds": HORIZON_SECONDS,
        "latency_ms": LATENCY_MS,
        "profit_cost_bps": PROFIT_COST_BPS,
        "stress_cost_bps": STRESS_COST_BPS,
        "feature_count": len(feature_columns),
        "training_row_count": int(len(training_rows)),
        "scored_row_count": int(len(scored_rows)),
        "settlement_count": len(settlements),
        "strict_survivor_count": sum(bool(item["strict_passed"]) for item in candidate_summaries),
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
            "fast182_is_historical_grid_audit_not_strategy_approval",
            "strict_survivors_require_future_only_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast182_edge_management_grid_historical_audit",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "managed_settlements": {"path": str(settlements_path), "sha256": sha256_file(settlements_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
        "trade_roots": {symbol: str(path) for symbol, path in sorted(trade_roots.items())},
    }
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def settle_policy_rows(selected: pd.DataFrame, *, threshold: float, policy, simulators: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
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
                "edge_threshold_bps": float(threshold),
                "management_policy": policy.name,
                "block": row.block,
                "date": row.date,
                "decision_time": decision_time.isoformat(),
                "side": row.side_label,
                "edge": float(row.edge),
                "actual_bps": float(row.actual_bps),
                "terminal_net_bps": float(row.terminal_net_bps),
                "stress_net_bps": trade.stress_net_bps,
                "exit_reason": trade.exit_reason.value,
            }
        )
    return records


def summarize_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    summaries = []
    for (threshold, policy), rows in frame.groupby(["edge_threshold_bps", "management_policy"], sort=True):
        blocks = [summarize_rows(block_rows, block=str(block)) for block, block_rows in rows.groupby("block", sort=True)]
        block_totals = {block["block"]: float(block["stress_total_bps"]) for block in blocks}
        overall = summarize_rows(rows)
        summaries.append(
            {
                "edge_threshold_bps": float(threshold),
                "management_policy": str(policy),
                **overall,
                "blocks": blocks,
                "strict_passed": block_totals.get(VALIDATION_BLOCK, 0.0) > 0
                and block_totals.get(EXTENSION_BLOCK, 0.0) > 0,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            bool(item["strict_passed"]),
            float(item["stress_total_bps"]),
            float(item["stress_mean_bps"]),
            int(item["trade_count"]),
        ),
        reverse=True,
    )


def fast182_status(candidate_summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "historical_grid_audit"]
    strict = [item for item in candidate_summaries if item["strict_passed"]]
    if strict:
        best = strict[0]
        return (
            "FAST182_EDGE_MANAGEMENT_GRID_STRICT_HISTORICAL_SURVIVORS",
            reasons
            + [
                f"strict_survivors:{len(strict)}",
                f"best:{best['edge_threshold_bps']}:{best['management_policy']}",
                "fresh_future_preregistration_required",
            ],
        )
    return "FAST182_EDGE_MANAGEMENT_GRID_NO_STRICT_SURVIVORS", reasons + ["no_candidate_positive_in_validation_and_extension"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "top_candidates": report["summary"]["top_candidates"][:5],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "summary_sha256": report["artifacts"]["summary"]["sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

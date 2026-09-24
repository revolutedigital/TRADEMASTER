"""Fast179 path-management audit for the Fast177/Fast178 ETHUSDT line.

Fast176-Fast178 showed that the current cross-asset ETHUSDT ridge signal can
identify some large moves, but it is still directionally fragile. Fast179 does
not select a strategy. It asks the narrower diagnostic question Igor raised:

Did the selected entries actually move in our favor before moving against us,
and would a short breakeven/trailing policy have changed the conclusion?

The audit is research-only, local, and never authorizes order submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
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
    TrailingPolicy,
)


DEFAULT_FAST177_ROWS = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast177-fast176-relaxed-edge-observation"
    / "prospective-diagnostic-rows.jsonl"
)
DEFAULT_FAST178_ROWS = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast178-fast177-edge2-observation"
    / "prospective-diagnostic-rows.jsonl"
)
DEFAULT_TRADE_PARTITION = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_ethusdt_v1"
    / "normalized"
    / "aggTrades"
    / "date=2026-09-24"
    / "events.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast179-fast177-path-management-audit"
)

HORIZON_SECONDS = 900
LATENCY_MS = 100
PROFIT_COST_BPS = 1.0
STRESS_COST_BPS = 2.0
FAVORABLE_TOUCH_BPS = 2.0
ADVERSE_TOUCH_BPS = 6.0
MFE_THRESHOLDS_BPS = (2.0, 5.0, 8.0, 10.0, 12.0, 16.0, 20.0)


@dataclass(frozen=True)
class PathMetrics:
    decision_time_ms: int
    entry_time_ms: int
    horizon_end_time_ms: int
    entry_price: float
    terminal_price: float
    terminal_gross_bps: float
    terminal_stress_net_bps: float
    maximum_favorable_bps: float
    maximum_adverse_bps: float
    first_favorable_touch_ms: int | None
    first_adverse_touch_ms: int | None
    favorable_before_adverse: bool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast177-rows", type=Path, default=DEFAULT_FAST177_ROWS)
    parser.add_argument("--fast178-rows", type=Path, default=DEFAULT_FAST178_ROWS)
    parser.add_argument("--trade-partition", type=Path, default=DEFAULT_TRADE_PARTITION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--latency-ms", type=int, default=LATENCY_MS)
    parser.add_argument("--horizon-seconds", type=int, default=HORIZON_SECONDS)
    parser.add_argument("--profit-cost-bps", type=float, default=PROFIT_COST_BPS)
    parser.add_argument("--stress-cost-bps", type=float, default=STRESS_COST_BPS)
    arguments = parser.parse_args(argv)

    report = run_fast179_path_management_audit(
        fast177_rows_path=arguments.fast177_rows,
        fast178_rows_path=arguments.fast178_rows,
        trade_partition_path=arguments.trade_partition,
        output_root=arguments.output_root,
        latency_ms=arguments.latency_ms,
        horizon_seconds=arguments.horizon_seconds,
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast179_path_management_audit(
    *,
    fast177_rows_path: Path = DEFAULT_FAST177_ROWS,
    fast178_rows_path: Path = DEFAULT_FAST178_ROWS,
    trade_partition_path: Path = DEFAULT_TRADE_PARTITION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    latency_ms: int = LATENCY_MS,
    horizon_seconds: int = HORIZON_SECONDS,
    profit_cost_bps: float = PROFIT_COST_BPS,
    stress_cost_bps: float = STRESS_COST_BPS,
) -> dict[str, Any]:
    validate_parameters(
        latency_ms=latency_ms,
        horizon_seconds=horizon_seconds,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    source_rows = [
        *load_source_rows(fast177_rows_path, source_experiment="fast177"),
        *load_source_rows(fast178_rows_path, source_experiment="fast178"),
    ]
    event_times_ms, prices = load_trade_partition(trade_partition_path)
    simulator = HistoricalTrailingSimulator(
        event_times_ms,
        prices,
        latency_ms=latency_ms,
        expected_round_trip_bps=profit_cost_bps,
        stress_round_trip_bps=stress_cost_bps,
    )
    scored_rows = [
        score_decision_path(
            row,
            event_times_ms=event_times_ms,
            prices=prices,
            simulator=simulator,
            policies=LOW_COST_TRAILING_POLICIES,
            latency_ms=latency_ms,
            horizon_seconds=horizon_seconds,
            stress_cost_bps=stress_cost_bps,
        )
        for row in source_rows
    ]
    aggregate = aggregate_scored_rows(scored_rows)
    status, status_reasons = fast179_status(aggregate)

    output_root.mkdir(parents=True, exist_ok=True)
    scored_path = output_root / "scored-decision-paths.jsonl"
    write_jsonl(scored_path, scored_rows)
    summary_path = output_root / "summary.json"
    summary_payload = {
        "status": status,
        "status_reasons": status_reasons,
        "aggregate": aggregate,
        "scored_decision_paths_path": str(scored_path),
        "scored_decision_paths_sha256": sha256_file(scored_path),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
            "fast179_is_path_management_diagnostic_only",
            "uses_same_prospective_fast177_fast178_rows",
            "positive_trailing_result_would_require_new_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast179_fast177_path_management_audit",
        "status": status,
        "status_reasons": status_reasons,
        "config": {
            "target_symbol": "ETHUSDT",
            "horizon_seconds": horizon_seconds,
            "latency_ms": latency_ms,
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "favorable_touch_bps": FAVORABLE_TOUCH_BPS,
            "adverse_touch_bps": ADVERSE_TOUCH_BPS,
            "policies": [
                {
                    "name": policy.name,
                    "initial_stop_bps": policy.initial_stop_bps,
                    "breakeven_activation_bps": policy.breakeven_activation_bps,
                    "breakeven_lock_bps": policy.breakeven_lock_bps,
                    "trailing_activation_bps": policy.trailing_activation_bps,
                    "trailing_distance_bps": policy.trailing_distance_bps,
                }
                for policy in LOW_COST_TRAILING_POLICIES
            ],
        },
        "source": {
            "fast177_rows_path": str(fast177_rows_path),
            "fast177_rows_sha256": sha256_file(fast177_rows_path),
            "fast178_rows_path": str(fast178_rows_path),
            "fast178_rows_sha256": sha256_file(fast178_rows_path),
            "trade_partition_path": str(trade_partition_path),
            "trade_partition_sha256": sha256_file(trade_partition_path),
        },
        "aggregate": aggregate,
        "scored_decision_paths_path": str(scored_path),
        "scored_decision_paths_sha256": sha256_file(scored_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def validate_parameters(
    *,
    latency_ms: int,
    horizon_seconds: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> None:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if min(profit_cost_bps, stress_cost_bps) < 0:
        raise ValueError("costs cannot be negative")
    if stress_cost_bps < profit_cost_bps:
        raise ValueError("stress cost cannot be below profit cost")


def load_source_rows(path: Path, *, source_experiment: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["source_experiment"] = source_experiment
        rows.append(row)
    return rows


def load_trade_partition(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing trade partition: {path}")
    frame = pd.read_parquet(path, columns=["event_time", "price"]).sort_values(
        ["event_time"], kind="stable"
    )
    if frame.empty:
        raise ValueError("trade partition is empty")
    event_times_ms = datetime_series_to_epoch_ms(frame["event_time"])
    prices = frame["price"].to_numpy(dtype=np.float64)
    if (np.diff(event_times_ms) < 0).any():
        raise ValueError("trade partition times regress")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("trade partition prices must be finite and positive")
    return event_times_ms, prices


def datetime_series_to_epoch_ms(series: pd.Series) -> np.ndarray:
    raw_values = pd.to_datetime(series, utc=True).astype("int64").to_numpy(dtype=np.int64)
    if len(raw_values) == 0:
        return raw_values
    magnitude = int(np.nanmax(np.abs(raw_values)))
    if magnitude >= 10**17:
        return raw_values // 1_000_000
    if magnitude >= 10**14:
        return raw_values // 1_000
    if magnitude >= 10**11:
        return raw_values
    if magnitude >= 10**8:
        return raw_values * 1_000
    raise ValueError("event_time values are too small to be epoch timestamps")


def score_decision_path(
    row: dict[str, Any],
    *,
    event_times_ms: np.ndarray,
    prices: np.ndarray,
    simulator: HistoricalTrailingSimulator,
    policies: tuple[TrailingPolicy, ...],
    latency_ms: int,
    horizon_seconds: int,
    stress_cost_bps: float,
) -> dict[str, Any]:
    decision_time_ms = parse_decision_time_ms(str(row["decision_time"]))
    side = side_from_row(str(row["side"]))
    path_metrics = compute_path_metrics(
        event_times_ms=event_times_ms,
        prices=prices,
        decision_time_ms=decision_time_ms,
        side=side,
        horizon_seconds=horizon_seconds,
        latency_ms=latency_ms,
        stress_cost_bps=stress_cost_bps,
    )
    policy_results = []
    for policy in policies:
        trade = simulator.simulate(
            decision_time_ms=decision_time_ms,
            side=side,
            horizon_seconds=horizon_seconds,
            probability=float(row["edge"]),
            policy=policy,
        )
        policy_results.append(
            {
                "policy": policy.name,
                "exit_reason": trade.exit_reason.value,
                "gross_bps": trade.gross_bps,
                "expected_net_bps": trade.expected_net_bps,
                "stress_net_bps": trade.stress_net_bps,
                "maximum_favorable_bps_until_exit": trade.maximum_favorable_bps,
                "maximum_adverse_bps_until_exit": trade.maximum_adverse_bps,
                "entry_time_ms": trade.entry_time_ms,
                "exit_time_ms": trade.exit_time_ms,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
            }
        )
    return {
        "source_experiment": row["source_experiment"],
        "decision_time": row["decision_time"],
        "decision_time_ms": decision_time_ms,
        "target_symbol": row["target_symbol"],
        "side": row["side"],
        "prediction": row["prediction"],
        "edge": row["edge"],
        "edge_threshold": row["edge_threshold"],
        "edge_gate_passed": row["edge_gate_passed"],
        "regime_gate_value": row["regime_gate_value"],
        "regime_gate_threshold": row["regime_gate_threshold"],
        "regime_gate_passed": row["regime_gate_passed"],
        "eligible": row["eligible"],
        "actual_bps": row["actual_bps"],
        "net_bps_if_eligible": row["net_bps_if_eligible"],
        **asdict(path_metrics),
        "mfe_threshold_passed": {
            f"{threshold:g}bps": path_metrics.maximum_favorable_bps >= threshold
            for threshold in MFE_THRESHOLDS_BPS
        },
        "policy_results": policy_results,
    }


def parse_decision_time_ms(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("decision_time must include timezone")
    return int(parsed.astimezone(UTC).timestamp() * 1_000)


def side_from_row(side: str) -> OrderSide:
    if side == "LONG":
        return OrderSide.BUY
    if side == "SHORT":
        return OrderSide.SELL
    raise ValueError(f"unknown side: {side}")


def compute_path_metrics(
    *,
    event_times_ms: np.ndarray,
    prices: np.ndarray,
    decision_time_ms: int,
    side: OrderSide,
    horizon_seconds: int,
    latency_ms: int,
    stress_cost_bps: float,
) -> PathMetrics:
    entry_index = int(np.searchsorted(event_times_ms, decision_time_ms + latency_ms, side="left"))
    end_index = int(
        np.searchsorted(event_times_ms, decision_time_ms + horizon_seconds * 1_000, side="right")
    )
    if entry_index >= len(event_times_ms) or entry_index >= end_index:
        raise ValueError("no executable full path for decision")
    entry_price = float(prices[entry_index])
    path_prices = prices[entry_index:end_index].astype(np.float64, copy=False)
    path_times = event_times_ms[entry_index:end_index].astype(np.int64, copy=False)
    direction = 1.0 if side == OrderSide.BUY else -1.0
    moves_bps = direction * np.log(path_prices / entry_price) * 10_000
    first_favorable_touch_ms = first_touch_time(
        path_times, moves_bps, threshold_bps=FAVORABLE_TOUCH_BPS, direction="gte"
    )
    first_adverse_touch_ms = first_touch_time(
        path_times, moves_bps, threshold_bps=-ADVERSE_TOUCH_BPS, direction="lte"
    )
    favorable_before_adverse = first_favorable_touch_ms is not None and (
        first_adverse_touch_ms is None or first_favorable_touch_ms <= first_adverse_touch_ms
    )
    terminal_gross_bps = float(moves_bps[-1])
    return PathMetrics(
        decision_time_ms=decision_time_ms,
        entry_time_ms=int(path_times[0]),
        horizon_end_time_ms=int(path_times[-1]),
        entry_price=entry_price,
        terminal_price=float(path_prices[-1]),
        terminal_gross_bps=terminal_gross_bps,
        terminal_stress_net_bps=terminal_gross_bps - stress_cost_bps,
        maximum_favorable_bps=float(np.max(moves_bps)),
        maximum_adverse_bps=float(np.min(moves_bps)),
        first_favorable_touch_ms=first_favorable_touch_ms,
        first_adverse_touch_ms=first_adverse_touch_ms,
        favorable_before_adverse=favorable_before_adverse,
    )


def first_touch_time(
    times_ms: np.ndarray,
    moves_bps: np.ndarray,
    *,
    threshold_bps: float,
    direction: str,
) -> int | None:
    if direction == "gte":
        matches = np.flatnonzero(moves_bps >= threshold_bps)
    elif direction == "lte":
        matches = np.flatnonzero(moves_bps <= threshold_bps)
    else:
        raise ValueError(f"unknown touch direction: {direction}")
    if len(matches) == 0:
        return None
    return int(times_ms[int(matches[0])])


def aggregate_scored_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, Any] = {}
    for source in sorted({str(row["source_experiment"]) for row in rows}):
        source_rows = [row for row in rows if row["source_experiment"] == source]
        by_source[source] = {
            "all_rows": summarize_rows(source_rows),
            "eligible_rows": summarize_rows([row for row in source_rows if bool(row["eligible"])]),
            "policy_results_all_rows": summarize_policy_results(source_rows),
            "policy_results_eligible_rows": summarize_policy_results(
                [row for row in source_rows if bool(row["eligible"])]
            ),
        }
    return {
        "row_count": len(rows),
        "source_count": len(by_source),
        "sources": by_source,
        "best_eligible_policy": best_eligible_policy(by_source),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "eligible_count": 0,
            "edge_gate_pass_count": 0,
            "regime_gate_pass_count": 0,
            "terminal_stress_net_total_bps": 0.0,
            "terminal_stress_net_mean_bps": 0.0,
            "terminal_stress_win_rate": 0.0,
            "mean_maximum_favorable_bps": 0.0,
            "mean_maximum_adverse_bps": 0.0,
            "favorable_before_adverse_share": 0.0,
            "mfe_threshold_shares": {f"{threshold:g}bps": 0.0 for threshold in MFE_THRESHOLDS_BPS},
        }
    terminal = np.array([float(row["terminal_stress_net_bps"]) for row in rows], dtype=np.float64)
    return {
        "row_count": len(rows),
        "eligible_count": sum(bool(row["eligible"]) for row in rows),
        "edge_gate_pass_count": sum(bool(row["edge_gate_passed"]) for row in rows),
        "regime_gate_pass_count": sum(bool(row["regime_gate_passed"]) for row in rows),
        "terminal_stress_net_total_bps": float(np.sum(terminal)),
        "terminal_stress_net_mean_bps": float(np.mean(terminal)),
        "terminal_stress_win_rate": float(np.mean(terminal > 0)),
        "mean_maximum_favorable_bps": float(np.mean([row["maximum_favorable_bps"] for row in rows])),
        "mean_maximum_adverse_bps": float(np.mean([row["maximum_adverse_bps"] for row in rows])),
        "favorable_before_adverse_share": float(
            np.mean([bool(row["favorable_before_adverse"]) for row in rows])
        ),
        "mfe_threshold_shares": {
            f"{threshold:g}bps": float(
                np.mean([float(row["maximum_favorable_bps"]) >= threshold for row in rows])
            )
            for threshold in MFE_THRESHOLDS_BPS
        },
    }


def summarize_policy_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for policy_result in row["policy_results"]:
            grouped[str(policy_result["policy"])].append(policy_result)
    summaries: dict[str, Any] = {}
    for policy_name, policy_rows in sorted(grouped.items()):
        stress = np.array([float(row["stress_net_bps"]) for row in policy_rows], dtype=np.float64)
        exit_counts = Counter(str(row["exit_reason"]) for row in policy_rows)
        summaries[policy_name] = {
            "row_count": len(policy_rows),
            "stress_net_total_bps": float(np.sum(stress)),
            "stress_net_mean_bps": float(np.mean(stress)) if len(stress) else 0.0,
            "stress_win_rate": float(np.mean(stress > 0)) if len(stress) else 0.0,
            "exit_reason_counts": dict(sorted(exit_counts.items())),
        }
    return summaries


def best_eligible_policy(by_source: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for source, payload in by_source.items():
        for policy_name, summary in payload["policy_results_eligible_rows"].items():
            if int(summary["row_count"]) <= 0:
                continue
            candidates.append(
                {
                    "source_experiment": source,
                    "policy": policy_name,
                    **summary,
                }
            )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["stress_net_total_bps"]),
            float(item["stress_net_mean_bps"]),
            float(item["stress_win_rate"]),
        ),
    )


def fast179_status(aggregate: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "path_management_audit"]
    best = aggregate.get("best_eligible_policy")
    if best is None:
        return "FAST179_NO_ELIGIBLE_ROWS_TO_AUDIT", reasons + ["no_eligible_fast177_fast178_rows"]
    if int(best["row_count"]) < 10:
        reasons.append(f"sample_too_small:{best['row_count']}<10")
    if float(best["stress_net_total_bps"]) <= 0:
        return "FAST179_PATH_MANAGEMENT_REJECTS_CURRENT_ELIGIBLE_ENTRIES", reasons + [
            "best_eligible_policy_stress_total_non_positive",
            "do_not_promote_to_strategy",
        ]
    return "FAST179_PATH_MANAGEMENT_DIAGNOSTIC_POSITIVE_BUT_UNAPPROVED", reasons + [
        "positive_path_management_result_requires_fresh_preregistration",
        "do_not_promote_to_strategy",
    ]


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "best_eligible_policy": report["aggregate"]["best_eligible_policy"],
        "report_path": report["report_path"],
        "report_sha256": sha256_file(Path(report["report_path"])),
        "summary_sha256": report["summary_sha256"],
        "scored_decision_paths_sha256": report["scored_decision_paths_sha256"],
        "research_only": report["research_only"],
        "approved_strategy": report["approved_strategy"],
        "order_submission_allowed": report["order_submission_allowed"],
        "orders_sent": report["orders_sent"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

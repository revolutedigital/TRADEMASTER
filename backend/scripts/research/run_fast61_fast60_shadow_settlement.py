"""Settle Fast60 live-like shadow signals with exact trailing replay.

Fast61 consumes the gated decisions emitted by Fast60 and replays the following
trade path with the frozen Fast57 trailing policy. It writes only local research
artifacts: no orders, no ledger commits, and no strategy approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.event_replay import OrderSide  # noqa: E402
from app.services.backtest.trailing_portfolio import (  # noqa: E402
    HistoricalTrailingSimulator,
    LOW_COST_TRAILING_POLICIES,
    TrailingPolicy,
    summarize_portfolio,
)
from app.services.research.research_dataset import load_trade_interval  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_FAST60_REPORT = DEFAULT_DATA_ROOT / "reports" / "fast60-fast57-shadow-dry-run" / "report.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast61-fast60-shadow-settlement"
DEFAULT_POLICY_NAME = "micro_8_3"
STRATEGY_MINIMUM_DAYS = 10
STRATEGY_MINIMUM_TRADES = 100
PRESERVED_DECISION_CONTEXT_FIELDS = (
    "causal_feature_values",
    "quote_volume_30s",
    "realized_vol_30s_bps",
    "directed_flow_imbalance_30s",
    "flow_imbalance_5s",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast60-report", type=Path, default=DEFAULT_FAST60_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-complete-outcomes", type=int, default=10)
    parser.add_argument("--expected-product", default="usdm_perpetual")
    arguments = parser.parse_args()

    report = run_fast61_fast60_shadow_settlement(
        fast60_report_path=arguments.fast60_report,
        output_root=arguments.output_root,
        policy_name=arguments.policy_name,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_complete_outcomes=arguments.minimum_complete_outcomes,
        expected_product=arguments.expected_product,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast61_fast60_shadow_settlement(
    *,
    fast60_report_path: Path,
    output_root: Path,
    policy_name: str = DEFAULT_POLICY_NAME,
    latency_ms: int = 100,
    expected_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    minimum_complete_outcomes: int = 10,
    expected_product: str = "usdm_perpetual",
) -> dict[str, Any]:
    _validate_parameters(
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_complete_outcomes=minimum_complete_outcomes,
    )
    policy = _policy_by_name(policy_name)
    fast60_report = json.loads(fast60_report_path.read_text(encoding="utf-8"))
    _validate_fast60_boundary(fast60_report)
    decisions_path = _resolve_decisions_path(fast60_report_path, fast60_report)
    decisions = [
        record
        for record in _load_jsonl(decisions_path)
        if bool(record.get("would_enter", False))
    ]
    output_root.mkdir(parents=True, exist_ok=True)

    run_wal_roots = _run_wal_roots(fast60_report)
    independent_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    for label, label_decisions in _group_by_label(decisions).items():
        wal_root_raw = run_wal_roots.get(label)
        if wal_root_raw is None:
            incomplete_rows.extend(
                _incomplete_row(record, reason="wal_root_not_found_in_fast60_report")
                for record in label_decisions
            )
            continue
        independent, incomplete = _settle_label_decisions(
            label_decisions,
            wal_root=Path(wal_root_raw),
            policy=policy,
            latency_ms=latency_ms,
            expected_cost_bps=expected_cost_bps,
            stress_cost_bps=stress_cost_bps,
            expected_product=expected_product,
        )
        independent_rows.extend(independent)
        incomplete_rows.extend(incomplete)

    independent_rows = sorted(
        independent_rows,
        key=lambda row: (int(row["decision_time_ms"]), float(row["probability"])),
    )
    one_position_rows = _apply_one_position_constraint(independent_rows)
    independent_path = output_root / "independent_settlements.jsonl"
    one_position_path = output_root / "one_position_settlements.jsonl"
    incomplete_path = output_root / "incomplete_signals.jsonl"
    _write_jsonl(independent_path, independent_rows)
    _write_jsonl(one_position_path, one_position_rows)
    _write_jsonl(incomplete_path, incomplete_rows)

    independent_summary = _summary_with_labels(independent_rows)
    one_position_summary = _summary_with_labels(one_position_rows)
    aggregate = {
        "signal_count": len(decisions),
        "complete_outcome_count": len(independent_rows),
        "incomplete_outcome_count": len(incomplete_rows),
        "one_position_outcome_count": len(one_position_rows),
        "decision_day_count": _decision_day_count(independent_rows),
        "label_count": len({str(row["label"]) for row in decisions}),
        "complete_label_count": len({str(row["label"]) for row in independent_rows}),
    }
    status, status_reasons = fast61_status(
        aggregate=aggregate,
        independent_summary=independent_summary,
        one_position_summary=one_position_summary,
        minimum_complete_outcomes=minimum_complete_outcomes,
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
            "fast61_settles_shadow_decisions_without_exchange_execution",
            "no_order_execution",
            "no_ledger_commit",
            "sample_is_below_strategy_minimum_even_if_positive",
        ],
        "experiment_stage": "fast61_fast60_shadow_settlement",
        "status": status,
        "status_reasons": status_reasons,
        "candidate": {
            **dict(fast60_report["candidate"]),
            "policy_name": policy.name,
            "policy": {
                "initial_stop_bps": policy.initial_stop_bps,
                "breakeven_activation_bps": policy.breakeven_activation_bps,
                "breakeven_lock_bps": policy.breakeven_lock_bps,
                "trailing_activation_bps": policy.trailing_activation_bps,
                "trailing_distance_bps": policy.trailing_distance_bps,
            },
            "latency_ms": latency_ms,
            "expected_cost_bps": expected_cost_bps,
            "stress_cost_bps": stress_cost_bps,
        },
        "strategy_minimums": {
            "minimum_days": STRATEGY_MINIMUM_DAYS,
            "minimum_trades": STRATEGY_MINIMUM_TRADES,
            "minimum_complete_shadow_outcomes": minimum_complete_outcomes,
        },
        "fast60_report_path": str(fast60_report_path),
        "fast60_report_sha256": _sha256_file(fast60_report_path),
        "gated_decisions_path": str(decisions_path),
        "gated_decisions_sha256": _sha256_file(decisions_path),
        "aggregate": aggregate,
        "independent_summary": independent_summary,
        "one_position_summary": one_position_summary,
        "independent_settlements_path": str(independent_path),
        "independent_settlements_sha256": _sha256_file(independent_path),
        "one_position_settlements_path": str(one_position_path),
        "one_position_settlements_sha256": _sha256_file(one_position_path),
        "incomplete_signals_path": str(incomplete_path),
        "incomplete_signals_sha256": _sha256_file(incomplete_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _settle_label_decisions(
    decisions: list[dict[str, Any]],
    *,
    wal_root: Path,
    policy: TrailingPolicy,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    expected_product: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not decisions:
        return [], []
    start_ms = min(int(row["decision_time_ms"]) for row in decisions) + latency_ms - 1_000
    end_ms = max(
        int(row["decision_time_ms"]) + int(row["horizon_seconds"]) * 1_000
        for row in decisions
    ) + 1_000
    try:
        trades = load_trade_interval(
            wal_root / "trade",
            datetime.fromtimestamp(start_ms / 1_000, tz=UTC),
            datetime.fromtimestamp(end_ms / 1_000, tz=UTC),
            expected_product=expected_product,
        )
    except (FileNotFoundError, ValueError) as exc:
        return [], [_incomplete_row(row, reason=f"trade_wal_unavailable:{exc}") for row in decisions]
    if trades.empty:
        return [], [_incomplete_row(row, reason="trade_wal_empty") for row in decisions]

    event_times = trades["event_time_ms"].to_numpy(dtype="int64")
    prices = trades["price"].to_numpy(dtype="float64")
    last_event_time_ms = int(event_times[-1])
    first_event_time_ms = int(event_times[0])
    simulator = HistoricalTrailingSimulator(
        event_times,
        prices,
        latency_ms=latency_ms,
        expected_round_trip_bps=expected_cost_bps,
        stress_round_trip_bps=stress_cost_bps,
    )
    complete_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    for decision in sorted(decisions, key=lambda row: int(row["decision_time_ms"])):
        decision_time_ms = int(decision["decision_time_ms"])
        required_end_ms = decision_time_ms + int(decision["horizon_seconds"]) * 1_000
        required_entry_ms = decision_time_ms + latency_ms
        if first_event_time_ms > required_entry_ms:
            incomplete_rows.append(
                _incomplete_row(
                    decision,
                    reason="entry_latency_time_not_covered_by_wal",
                    first_event_time_ms=first_event_time_ms,
                    last_event_time_ms=last_event_time_ms,
                    required_end_time_ms=required_end_ms,
                )
            )
            continue
        if last_event_time_ms < required_end_ms:
            incomplete_rows.append(
                _incomplete_row(
                    decision,
                    reason="horizon_not_fully_covered_by_wal",
                    first_event_time_ms=first_event_time_ms,
                    last_event_time_ms=last_event_time_ms,
                    required_end_time_ms=required_end_ms,
                )
            )
            continue
        try:
            managed_trade = simulator.simulate(
                decision_time_ms=decision_time_ms,
                side=OrderSide(str(decision["side"])),
                horizon_seconds=int(decision["horizon_seconds"]),
                probability=float(decision["probability"]),
                policy=policy,
            )
        except ValueError as exc:
            incomplete_rows.append(
                _incomplete_row(
                    decision,
                    reason=f"not_executable:{exc}",
                    first_event_time_ms=first_event_time_ms,
                    last_event_time_ms=last_event_time_ms,
                    required_end_time_ms=required_end_ms,
                )
            )
            continue
        complete_rows.append(
            {
                **managed_trade.to_dict(),
                "label": decision["label"],
                "decision_time": decision["decision_time"],
                "threshold": float(decision["threshold"]),
                "model_sha256": decision["model_sha256"],
                "feature_vector_sha256": decision["feature_vector_sha256"],
                "recent_signal_count": decision.get("recent_signal_count"),
                "book_available": decision.get("book_available"),
                "book_age_ms": decision.get("book_age_ms"),
                "spread_bps": decision.get("spread_bps"),
                "top_bid_notional_usdt": decision.get("top_bid_notional_usdt"),
                "top_ask_notional_usdt": decision.get("top_ask_notional_usdt"),
                "settlement_complete": True,
                "wal_first_event_time_ms": first_event_time_ms,
                "wal_last_event_time_ms": last_event_time_ms,
                "required_end_time_ms": required_end_ms,
                **_preserved_decision_context(decision),
            }
        )
    return complete_rows, incomplete_rows


def fast61_status(
    *,
    aggregate: dict[str, int],
    independent_summary: dict[str, float | int],
    one_position_summary: dict[str, float | int],
    minimum_complete_outcomes: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "dry_run_no_ledger_write", "no_order_execution"]
    complete_count = int(aggregate["complete_outcome_count"])
    if int(aggregate["signal_count"]) <= 0:
        return "FAST61_NO_FAST60_GATED_SIGNALS", reasons + ["no_gated_shadow_signals"]
    if complete_count <= 0:
        return "FAST61_NO_COMPLETE_SHADOW_OUTCOMES", reasons + [
            "no_signal_has_complete_horizon_coverage"
        ]

    independent_stress_total = float(independent_summary["stress_total_log_return_bps"])
    independent_stress_mean = float(independent_summary["stress_mean_bps"])
    one_position_stress_total = float(one_position_summary["stress_total_log_return_bps"])
    one_position_stress_mean = float(one_position_summary["stress_mean_bps"])
    if min(
        independent_stress_total,
        independent_stress_mean,
        one_position_stress_total,
        one_position_stress_mean,
    ) <= 0:
        return "FAST61_SHADOW_SETTLEMENT_FAILED", reasons + [
            "stress_result_non_positive_after_trailing_replay",
            *(
                [f"complete_outcomes_below_minimum:{complete_count}<{minimum_complete_outcomes}"]
                if complete_count < minimum_complete_outcomes
                else []
            ),
        ]
    if complete_count < minimum_complete_outcomes:
        return "FAST61_SHADOW_SETTLEMENT_INSUFFICIENT_SAMPLE", reasons + [
            f"complete_outcomes_below_minimum:{complete_count}<{minimum_complete_outcomes}"
        ]

    sample_reasons = []
    decision_days = int(aggregate["decision_day_count"])
    if decision_days < STRATEGY_MINIMUM_DAYS:
        sample_reasons.append(f"decision_days_below_strategy_minimum:{decision_days}<10")
    if complete_count < STRATEGY_MINIMUM_TRADES:
        sample_reasons.append(f"complete_trades_below_strategy_minimum:{complete_count}<100")
    return "FAST61_SHADOW_SETTLEMENT_POSITIVE_RESEARCH_ONLY_MORE_SAMPLE_REQUIRED", (
        reasons + sample_reasons
    )


def _apply_one_position_constraint(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    position_exit_ms = -1
    for row in sorted(
        rows,
        key=lambda item: (int(item["decision_time_ms"]), -float(item["probability"])),
    ):
        if int(row["decision_time_ms"]) <= position_exit_ms:
            continue
        selected.append(row)
        position_exit_ms = int(row["exit_time_ms"])
    return selected


def _summary_with_labels(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    frame = pd.DataFrame(rows)
    summary = summarize_portfolio(frame)
    return {
        **summary,
        "label_count": 0 if frame.empty else int(frame["label"].nunique()),
    }


def _incomplete_row(
    record: dict[str, Any],
    *,
    reason: str,
    first_event_time_ms: int | None = None,
    last_event_time_ms: int | None = None,
    required_end_time_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "label": record.get("label"),
        "decision_time": record.get("decision_time"),
        "decision_time_ms": record.get("decision_time_ms"),
        "side": record.get("side"),
        "horizon_seconds": record.get("horizon_seconds"),
        "probability": record.get("probability"),
        "threshold": record.get("threshold"),
        "model_sha256": record.get("model_sha256"),
        "feature_vector_sha256": record.get("feature_vector_sha256"),
        "settlement_complete": False,
        "incomplete_reason": reason,
        "wal_first_event_time_ms": first_event_time_ms,
        "wal_last_event_time_ms": last_event_time_ms,
        "required_end_time_ms": required_end_time_ms,
        **_preserved_decision_context(record),
    }


def _preserved_decision_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        field: record[field]
        for field in PRESERVED_DECISION_CONTEXT_FIELDS
        if field in record
    }


def _validate_parameters(
    *,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    minimum_complete_outcomes: int,
) -> None:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if min(expected_cost_bps, stress_cost_bps) < 0:
        raise ValueError("costs cannot be negative")
    if stress_cost_bps < expected_cost_bps:
        raise ValueError("stress cost cannot be below expected cost")
    if minimum_complete_outcomes < 1:
        raise ValueError("minimum_complete_outcomes must be positive")


def _policy_by_name(policy_name: str) -> TrailingPolicy:
    for policy in LOW_COST_TRAILING_POLICIES:
        if policy.name == policy_name:
            return policy
    raise ValueError(f"unknown low-cost trailing policy: {policy_name}")


def _validate_fast60_boundary(report: dict[str, Any]) -> None:
    if report.get("research_only") is not True:
        raise ValueError("Fast60 report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError("Fast60 report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError("Fast60 report must have execution_authorization=none")
    if report.get("approved_strategy") is not False:
        raise ValueError("Fast60 report must not be an approved strategy")
    if report.get("committed") is not False or report.get("dry_run") is not True:
        raise ValueError("Fast60 settlement expects a dry-run report")


def _resolve_decisions_path(report_path: Path, report: dict[str, Any]) -> Path:
    raw_path = Path(str(report["gated_decisions_path"]))
    if raw_path.exists():
        return raw_path
    candidate = report_path.parent / raw_path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Fast60 gated decisions file not found: {raw_path}")


def _run_wal_roots(report: dict[str, Any]) -> dict[str, str]:
    return {
        str(run["label"]): str(run["wal_root"])
        for run in report.get("runs", [])
        if "label" in run and "wal_root" in run
    }


def _group_by_label(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["label"])].append(record)
    return dict(grouped)


def _decision_day_count(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return len(
        {
            datetime.fromtimestamp(int(row["decision_time_ms"]) / 1_000, tz=UTC).date()
            for row in rows
        }
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "aggregate": report["aggregate"],
        "independent_summary": report["independent_summary"],
        "one_position_summary": report["one_position_summary"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

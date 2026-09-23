"""Fast62 payoff-filter survivorship audit.

This diagnostic asks whether the Fast60 raw top-p stream contains simple
pre-trade filters that improve trailing payoff, and immediately checks the most
tempting side filter against older Fast43-Fast53 holdout blocks. It is
diagnostic-only and cannot approve a strategy.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast61_fast60_shadow_settlement import (  # noqa: E402
    DEFAULT_FAST60_REPORT,
    _apply_one_position_constraint,
    _group_by_label,
    _load_jsonl,
    _policy_by_name,
    _run_wal_roots,
    _settle_label_decisions,
    _sha256_file,
    _summary_with_labels,
    _validate_fast60_boundary,
    _write_jsonl,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast62-payoff-filter-survivorship-audit"
DEFAULT_POLICY_NAME = "micro_8_3"
FAST_NUMBER_PATTERN = re.compile(r"fast(?P<number>\d+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast60-report", type=Path, default=DEFAULT_FAST60_REPORT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--historical-fast-start", type=int, default=43)
    parser.add_argument("--historical-fast-end", type=int, default=53)
    parser.add_argument("--minimum-filter-trades", type=int, default=5)
    parser.add_argument("--expected-product", default="usdm_perpetual")
    arguments = parser.parse_args()

    report = run_fast62_payoff_filter_survivorship_audit(
        fast60_report_path=arguments.fast60_report,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        policy_name=arguments.policy_name,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        minimum_filter_trades=arguments.minimum_filter_trades,
        expected_product=arguments.expected_product,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast62_payoff_filter_survivorship_audit(
    *,
    fast60_report_path: Path,
    data_root: Path,
    output_root: Path,
    policy_name: str = DEFAULT_POLICY_NAME,
    latency_ms: int = 100,
    expected_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    minimum_filter_trades: int = 5,
    expected_product: str = "usdm_perpetual",
) -> dict[str, Any]:
    if minimum_filter_trades < 1:
        raise ValueError("minimum_filter_trades must be positive")
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")

    fast60_report = json.loads(fast60_report_path.read_text(encoding="utf-8"))
    _validate_fast60_boundary(fast60_report)
    policy = _policy_by_name(policy_name)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_decisions_path = _resolve_raw_decisions_path(fast60_report_path, fast60_report)
    raw_decisions = [
        record
        for record in _load_jsonl(raw_decisions_path)
        if bool(record.get("would_enter", False))
    ]
    _add_recent_signal_counts(raw_decisions, lookback_seconds=300)
    complete_rows, incomplete_rows = _settle_fast60_raw_decisions(
        raw_decisions,
        fast60_report=fast60_report,
        policy_name=policy.name,
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
        expected_product=expected_product,
    )

    complete_path = output_root / "fast60_raw_complete_settlements.jsonl"
    incomplete_path = output_root / "fast60_raw_incomplete_signals.jsonl"
    _write_jsonl(complete_path, complete_rows)
    _write_jsonl(incomplete_path, incomplete_rows)
    filter_candidates = fast60_filter_candidates(
        complete_rows,
        minimum_filter_trades=minimum_filter_trades,
    )
    filter_candidates_path = output_root / "fast60_filter_candidates.json"
    filter_candidates_path.write_text(
        json.dumps(filter_candidates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    historical_paths = discover_historical_holdout_trade_paths(
        data_root,
        fast_start=historical_fast_start,
        fast_end=historical_fast_end,
    )
    historical_side_audit = historical_side_survivorship(historical_paths)
    historical_side_path = output_root / "historical_side_survivorship.json"
    historical_side_path.write_text(
        json.dumps(historical_side_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status, status_reasons = fast62_status(
        filter_candidates=filter_candidates,
        historical_side_audit=historical_side_audit,
        complete_count=len(complete_rows),
        historical_path_count=len(historical_paths),
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
            "fast62_is_diagnostic_only",
            "fast60_raw_filter_search_is_one_day_in_sample",
            "historical_side_survivorship_contradicts_simple_side_filter",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast62_payoff_filter_survivorship_audit",
        "status": status,
        "status_reasons": status_reasons,
        "candidate": {
            **dict(fast60_report["candidate"]),
            "policy_name": policy.name,
            "latency_ms": latency_ms,
            "expected_cost_bps": expected_cost_bps,
            "stress_cost_bps": stress_cost_bps,
        },
        "fast60_report_path": str(fast60_report_path),
        "fast60_report_sha256": _sha256_file(fast60_report_path),
        "fast60_raw_decisions_path": str(raw_decisions_path),
        "fast60_raw_decisions_sha256": _sha256_file(raw_decisions_path),
        "fast60_raw_signal_count": len(raw_decisions),
        "fast60_complete_outcome_count": len(complete_rows),
        "fast60_incomplete_outcome_count": len(incomplete_rows),
        "fast60_baseline_one_position_summary": _summary_with_labels(
            _apply_one_position_constraint(complete_rows)
        ),
        "top_fast60_filter_candidates": filter_candidates[:10],
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "historical_holdout_path_count": len(historical_paths),
        "historical_side_aggregate": historical_side_audit["aggregate_by_side"],
        "fast60_complete_settlements_path": str(complete_path),
        "fast60_complete_settlements_sha256": _sha256_file(complete_path),
        "fast60_incomplete_signals_path": str(incomplete_path),
        "fast60_incomplete_signals_sha256": _sha256_file(incomplete_path),
        "fast60_filter_candidates_path": str(filter_candidates_path),
        "fast60_filter_candidates_sha256": _sha256_file(filter_candidates_path),
        "historical_side_survivorship_path": str(historical_side_path),
        "historical_side_survivorship_sha256": _sha256_file(historical_side_path),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def discover_historical_holdout_trade_paths(
    data_root: Path,
    *,
    fast_start: int,
    fast_end: int,
) -> tuple[Path, ...]:
    paths = []
    for path in sorted(
        (data_root / "reports").glob(
            "latency-adjusted-frozen-top-p-fast*-lowcost-micro-holdout-*-100ms/"
            "frozen_holdout_trades.parquet"
        )
    ):
        match = FAST_NUMBER_PATTERN.search(str(path))
        if match is None:
            continue
        fast_number = int(match.group("number"))
        if fast_start <= fast_number <= fast_end:
            paths.append(path)
    return tuple(paths)


def historical_side_survivorship(paths: tuple[Path, ...]) -> dict[str, Any]:
    block_rows: list[dict[str, Any]] = []
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        block_name = path.parent.name
        frame = frame.assign(block=block_name)
        frames.append(frame)
        for side, side_frame in frame.groupby("side", sort=True):
            block_rows.append(_side_summary_row(block_name, str(side), side_frame))

    if not frames:
        return {
            "block_count": 0,
            "trade_count": 0,
            "aggregate_by_side": {},
            "blocks_by_side": [],
        }

    all_trades = pd.concat(frames, ignore_index=True)
    aggregate_by_side = {}
    for side, side_frame in all_trades.groupby("side", sort=True):
        side_rows = [row for row in block_rows if row["side"] == str(side)]
        aggregate_by_side[str(side)] = {
            **_side_summary_core(side_frame),
            "positive_block_count": sum(float(row["stress_total_log_return_bps"]) > 0 for row in side_rows),
            "block_count": len(side_rows),
        }
    return {
        "block_count": len({str(row["block"]) for row in block_rows}),
        "trade_count": int(len(all_trades)),
        "aggregate_by_side": aggregate_by_side,
        "blocks_by_side": sorted(block_rows, key=lambda row: (row["block"], row["side"])),
    }


def fast60_filter_candidates(
    rows: list[dict[str, Any]],
    *,
    minimum_filter_trades: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    candidates: list[dict[str, Any]] = []

    def add_candidate(name: str, mask: pd.Series) -> None:
        subset = frame[mask].to_dict("records")
        if len(subset) < minimum_filter_trades:
            return
        one_position = _apply_one_position_constraint(subset)
        summary = _summary_with_labels(one_position)
        if int(summary["trade_count"]) < minimum_filter_trades:
            return
        candidates.append(
            {
                "name": name,
                "selected_signal_count": len(subset),
                "one_position_summary": summary,
                "diagnostic_positive": (
                    float(summary["stress_total_log_return_bps"]) > 0
                    and float(summary["stress_mean_bps"]) > 0
                ),
            }
        )

    add_candidate("baseline_all_raw_top_p", pd.Series([True] * len(frame), index=frame.index))
    for side in ("BUY", "SELL"):
        add_candidate(f"side={side}", frame["side"].eq(side))
    for maximum_recent in (0, 1, 2, 3, 5, 10, 20, 50):
        add_candidate(
            f"recent_signal_count<={maximum_recent}",
            frame["recent_signal_count"].astype(float).le(maximum_recent),
        )
    for quantile in (0.50, 0.70, 0.80, 0.90, 0.95, 0.98):
        threshold = float(frame["probability"].quantile(quantile))
        add_candidate(
            f"probability>=q{quantile:.2f}:{threshold:.6f}",
            frame["probability"].ge(threshold),
        )
    for side in ("BUY", "SELL"):
        for maximum_recent in (0, 1, 2, 5, 10):
            add_candidate(
                f"side={side}&recent_signal_count<={maximum_recent}",
                frame["side"].eq(side)
                & frame["recent_signal_count"].astype(float).le(maximum_recent),
            )

    return sorted(
        candidates,
        key=lambda item: (
            float(item["one_position_summary"]["stress_total_log_return_bps"]),
            float(item["one_position_summary"]["stress_mean_bps"]),
        ),
        reverse=True,
    )


def fast62_status(
    *,
    filter_candidates: list[dict[str, Any]],
    historical_side_audit: dict[str, Any],
    complete_count: int,
    historical_path_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "diagnostic_only", "no_order_execution"]
    if complete_count <= 0:
        return "FAST62_NO_COMPLETE_FAST60_RAW_OUTCOMES", reasons + [
            "fast60_raw_signals_have_no_complete_settlements"
        ]
    if historical_path_count <= 0:
        return "FAST62_MISSING_HISTORICAL_SURVIVORSHIP_DATA", reasons + [
            "historical_fast43_fast53_holdout_trades_not_found"
        ]
    positive_candidates = [
        candidate for candidate in filter_candidates if bool(candidate["diagnostic_positive"])
    ]
    aggregate_by_side = historical_side_audit.get("aggregate_by_side", {})
    side_failures = [
        side
        for side, summary in aggregate_by_side.items()
        if float(summary["stress_total_log_return_bps"]) <= 0
    ]
    if positive_candidates and side_failures:
        return "FAST62_FAST60_FILTER_POSITIVE_BUT_HISTORICAL_SIDE_SURVIVORSHIP_FAILED", (
            reasons
            + [
                f"fast60_positive_diagnostic_filters:{len(positive_candidates)}",
                f"historical_side_failures:{','.join(sorted(side_failures))}",
                "requires_payoff_target_rebuild_not_side_filter_promotion",
            ]
        )
    return "FAST62_NO_SURVIVING_PAYOFF_FILTER_STRATEGY", reasons + [
        "no_simple_filter_passed_current_and_historical_survivorship"
    ]


def _settle_fast60_raw_decisions(
    decisions: list[dict[str, Any]],
    *,
    fast60_report: dict[str, Any],
    policy_name: str,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    expected_product: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = _policy_by_name(policy_name)
    run_wal_roots = _run_wal_roots(fast60_report)
    complete_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    for label, label_decisions in _group_by_label(decisions).items():
        wal_root_raw = run_wal_roots.get(label)
        if wal_root_raw is None:
            incomplete_rows.extend(
                {
                    **record,
                    "settlement_complete": False,
                    "incomplete_reason": "wal_root_not_found_in_fast60_report",
                }
                for record in label_decisions
            )
            continue
        complete, incomplete = _settle_label_decisions(
            label_decisions,
            wal_root=Path(wal_root_raw),
            policy=policy,
            latency_ms=latency_ms,
            expected_cost_bps=expected_cost_bps,
            stress_cost_bps=stress_cost_bps,
            expected_product=expected_product,
        )
        complete_rows.extend(complete)
        incomplete_rows.extend(incomplete)
    return (
        sorted(complete_rows, key=lambda row: int(row["decision_time_ms"])),
        sorted(incomplete_rows, key=lambda row: int(row["decision_time_ms"])),
    )


def _add_recent_signal_counts(records: list[dict[str, Any]], *, lookback_seconds: int) -> None:
    lookback_ms = lookback_seconds * 1_000
    for records_for_label in _group_by_label(records).values():
        times: list[int] = []
        for record in sorted(
            records_for_label,
            key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"])),
        ):
            decision_time_ms = int(record["decision_time_ms"])
            recent_count = len(times) - bisect.bisect_left(times, decision_time_ms - lookback_ms)
            record["recent_signal_count"] = recent_count
            bisect.insort(times, decision_time_ms)


def _resolve_raw_decisions_path(report_path: Path, report: dict[str, Any]) -> Path:
    raw_path = Path(str(report["raw_decisions_path"]))
    if raw_path.exists():
        return raw_path
    candidate = report_path.parent / raw_path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Fast60 raw decisions file not found: {raw_path}")


def _side_summary_row(block: str, side: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {"block": block, "side": side, **_side_summary_core(frame)}


def _side_summary_core(frame: pd.DataFrame) -> dict[str, float | int]:
    stress = frame["stress_net_bps"]
    expected = frame["expected_net_bps"]
    return {
        "trade_count": int(len(frame)),
        "expected_total_log_return_bps": float(expected.sum()),
        "expected_mean_bps": float(expected.mean()),
        "stress_total_log_return_bps": float(stress.sum()),
        "stress_mean_bps": float(stress.mean()),
        "expected_win_rate": float((expected > 0).mean()),
    }


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
        "fast60_raw_signal_count": report["fast60_raw_signal_count"],
        "fast60_complete_outcome_count": report["fast60_complete_outcome_count"],
        "fast60_baseline_one_position_summary": report["fast60_baseline_one_position_summary"],
        "top_fast60_filter_candidates": report["top_fast60_filter_candidates"][:5],
        "historical_side_aggregate": report["historical_side_aggregate"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

"""Settle fixed-gate shadow decisions with executable bid/ask replay.

This consumes the JSONL produced by run_book_fixed_gate_shadow_policy.py and
settles those emitted decisions without re-running the signal rule. It is the
dry-run outcome counterpart for a prospective shadow stream and never submits
orders or writes to the research ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.event_replay import OrderSide  # noqa: E402
from app.services.research.research_dataset import load_book_interval  # noqa: E402
from scripts.research.run_book_executable_replay import (  # noqa: E402
    _apply_one_position_constraint,
    _book_arrays,
    simulate_executable_trade,
    summarize_executable_trades,
)
from scripts.research.run_book_rule_confirmation import _parse_float_list  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-report", type=Path, required=True)
    parser.add_argument("--wal-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-fixed-gate-shadow-settlement",
    )
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--target-bps", type=float, default=5.0)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--minimum-executable-count", type=int, default=10)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    arguments = parser.parse_args()
    report = settle_book_fixed_gate_shadow_decisions(
        shadow_report_path=arguments.shadow_report,
        wal_root=arguments.wal_root,
        output_root=arguments.output_root,
        latency_ms=arguments.latency_ms,
        target_bps=arguments.target_bps,
        initial_stop_bps=arguments.initial_stop_bps,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        minimum_executable_count=arguments.minimum_executable_count,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def settle_book_fixed_gate_shadow_decisions(
    *,
    shadow_report_path: Path,
    wal_root: Path,
    output_root: Path,
    latency_ms: int = 100,
    target_bps: float = 5.0,
    initial_stop_bps: float = 20.0,
    max_book_staleness_ms: int = 1_000,
    minimum_executable_count: int = 10,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, object]:
    _validate_parameters(
        latency_ms=latency_ms,
        target_bps=target_bps,
        initial_stop_bps=initial_stop_bps,
        max_book_staleness_ms=max_book_staleness_ms,
        minimum_executable_count=minimum_executable_count,
        cost_levels_bps=cost_levels_bps,
    )
    shadow_report = json.loads(shadow_report_path.read_text(encoding="utf-8"))
    _validate_shadow_report_boundary(shadow_report)
    decisions_path = _resolve_decisions_path(shadow_report_path, shadow_report)
    decision_records = _load_decision_records(decisions_path)
    entry_records = [record for record in decision_records if record["would_enter"] is True]
    if not entry_records:
        return _write_report(
            output_root,
            _base_report(
                shadow_report_path=shadow_report_path,
                shadow_report=shadow_report,
                decisions_path=decisions_path,
                latency_ms=latency_ms,
                target_bps=target_bps,
                initial_stop_bps=initial_stop_bps,
                max_book_staleness_ms=max_book_staleness_ms,
                cost_levels_bps=cost_levels_bps,
            )
            | {
                "status": "SHADOW_SETTLEMENT_NO_ENTRY_DECISIONS",
                "status_reasons": ["shadow_decisions_jsonl_has_no_entries"],
                "modeling_ready": False,
                "entry_decision_count": 0,
            },
        )
    book_events = _load_book_events_for_decisions(
        entry_records,
        wal_root=wal_root,
        latency_ms=latency_ms,
    )
    book = _book_arrays(book_events)
    settlements = [
        _settle_decision_record(
            record,
            book=book,
            latency_ms=latency_ms,
            target_bps=target_bps,
            initial_stop_bps=initial_stop_bps,
            max_book_staleness_ms=max_book_staleness_ms,
        )
        for record in entry_records
    ]
    trades = pd.DataFrame([row for row in settlements if row["would_enter"]])
    one_position = _apply_one_position_constraint(trades)
    independent_summary = summarize_executable_trades(
        trades,
        selected_signal_count=len(entry_records),
        cost_levels_bps=cost_levels_bps,
    )
    one_position_summary = summarize_executable_trades(
        one_position,
        selected_signal_count=len(entry_records),
        cost_levels_bps=cost_levels_bps,
    )
    status_reasons = []
    if independent_summary["executable_count"] < minimum_executable_count:
        status_reasons.append(
            "independent_executable_count_below_minimum:"
            f"{independent_summary['executable_count']}<{minimum_executable_count}"
        )
    if one_position_summary["executable_count"] < minimum_executable_count:
        status_reasons.append(
            "one_position_executable_count_below_minimum:"
            f"{one_position_summary['executable_count']}<{minimum_executable_count}"
        )
    if not _positive_at_cost(independent_summary, 1.0):
        status_reasons.append("independent_net_at_1bps_non_positive")
    if not _positive_at_cost(one_position_summary, 1.0):
        status_reasons.append("one_position_net_at_1bps_non_positive")
    modeling_ready = not status_reasons
    if modeling_ready:
        status_reasons.append("prospective_shadow_stream_required_before_testnet")

    output_root.mkdir(parents=True, exist_ok=True)
    settlements_path = output_root / "shadow_settlements.parquet"
    trades.to_parquet(settlements_path, index=False, compression="zstd")
    one_position_path = output_root / "one_position_settlements.parquet"
    one_position.to_parquet(one_position_path, index=False, compression="zstd")
    return _write_report(
        output_root,
        _base_report(
            shadow_report_path=shadow_report_path,
            shadow_report=shadow_report,
            decisions_path=decisions_path,
            latency_ms=latency_ms,
            target_bps=target_bps,
            initial_stop_bps=initial_stop_bps,
            max_book_staleness_ms=max_book_staleness_ms,
            cost_levels_bps=cost_levels_bps,
        )
        | {
            "status": (
                "SHADOW_SETTLEMENT_READY_FOR_PROSPECTIVE_STREAM"
                if modeling_ready
                else "SHADOW_SETTLEMENT_FAILED"
            ),
            "status_reasons": status_reasons,
            "modeling_ready": modeling_ready,
            "entry_decision_count": len(entry_records),
            "settlement_count": int(len(settlements)),
            "independent_summary": independent_summary,
            "one_position_summary": one_position_summary,
            "settlements_path": str(settlements_path),
            "one_position_settlements_path": str(one_position_path),
        },
    )


def _settle_decision_record(
    record: dict[str, Any],
    *,
    book: dict[str, Any],
    latency_ms: int,
    target_bps: float,
    initial_stop_bps: float,
    max_book_staleness_ms: int,
) -> dict[str, object]:
    result = simulate_executable_trade(
        book=book,
        decision_time_ms=int(record["decision_time_ms"]),
        side=OrderSide(str(record["side"])),
        horizon_seconds=int(record["horizon_seconds"]),
        target_bps=target_bps,
        initial_stop_bps=initial_stop_bps,
        latency_ms=latency_ms,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    result["policy_sha256"] = record["policy_sha256"]
    result["feature_vector_sha256"] = record["feature_vector_sha256"]
    result["shadow_decision_time"] = record["decision_time"]
    return result


def _base_report(
    *,
    shadow_report_path: Path,
    shadow_report: dict[str, Any],
    decisions_path: Path,
    latency_ms: int,
    target_bps: float,
    initial_stop_bps: float,
    max_book_staleness_ms: int,
    cost_levels_bps: tuple[float, ...],
) -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "committed": False,
        "dry_run": True,
        "experiment_stage": "book_fixed_gate_shadow_settlement",
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "settles_shadow_decisions_without_exchange_execution",
            "no_ledger_commit",
            "no_order_execution",
        ],
        "shadow_report_path": str(shadow_report_path),
        "shadow_report_sha256": _sha256_file(shadow_report_path),
        "shadow_decisions_path": str(decisions_path),
        "shadow_decisions_sha256": _sha256_file(decisions_path),
        "shadow_policy_sha256": shadow_report["policy_sha256"],
        "shadow_selected_count": shadow_report["selected_count"],
        "execution_replay": {
            "entry_exit_model": "marketable_taker_top_of_book",
            "latency_ms": latency_ms,
            "target_bps": target_bps,
            "initial_stop_bps": initial_stop_bps,
            "max_book_staleness_ms": max_book_staleness_ms,
            "explicit_cost_levels_bps": list(cost_levels_bps),
            "bid_ask_spread_included_in_gross": True,
            "one_position_constraint_reported": True,
        },
    }


def _load_book_events_for_decisions(
    records: list[dict[str, Any]],
    *,
    wal_root: Path,
    latency_ms: int,
) -> pd.DataFrame:
    start_ms = min(int(record["decision_time_ms"]) for record in records) + latency_ms - 2_000
    end_ms = max(
        int(record["decision_time_ms"]) + int(record["horizon_seconds"]) * 1000
        for record in records
    ) + 2_000
    return load_book_interval(
        wal_root / "depth",
        datetime.fromtimestamp(start_ms / 1000, tz=UTC),
        datetime.fromtimestamp(end_ms / 1000, tz=UTC),
    )


def _load_decision_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _resolve_decisions_path(shadow_report_path: Path, report: dict[str, Any]) -> Path:
    raw_path = Path(str(report["decisions_path"]))
    if raw_path.exists():
        return raw_path
    candidate = shadow_report_path.parent / raw_path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"shadow decisions file not found: {raw_path}")


def _validate_shadow_report_boundary(report: dict[str, Any]) -> None:
    if report.get("research_only") is not True:
        raise ValueError("shadow report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError("shadow report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError("shadow report must have execution_authorization=none")
    if report.get("committed") is not False or report.get("dry_run") is not True:
        raise ValueError("shadow settlement expects a dry-run report")


def _positive_at_cost(summary: dict[str, object], cost_bps: float) -> bool:
    for item in summary["cost_sensitivity"]:
        if float(item["explicit_cost_bps"]) == cost_bps:
            return bool(item["net_positive"])
    return False


def _validate_parameters(
    *,
    latency_ms: int,
    target_bps: float,
    initial_stop_bps: float,
    max_book_staleness_ms: int,
    minimum_executable_count: int,
    cost_levels_bps: tuple[float, ...],
) -> None:
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if target_bps <= 0:
        raise ValueError("target_bps must be positive")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if max_book_staleness_ms < 0:
        raise ValueError("max_book_staleness_ms cannot be negative")
    if minimum_executable_count < 1:
        raise ValueError("minimum_executable_count must be positive")
    if not cost_levels_bps or any(cost_bps < 0 for cost_bps in cost_levels_bps):
        raise ValueError("cost_levels_bps must contain non-negative values")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "status": report["status"],
        "shadow_selected_count": report.get("shadow_selected_count"),
        "independent_summary": report.get("independent_summary"),
        "one_position_summary": report.get("one_position_summary"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

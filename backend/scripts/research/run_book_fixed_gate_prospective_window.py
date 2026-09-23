"""Run the fixed book gate on one prospective WAL window, shadow-only.

This wrapper removes a manual source of drift in the research loop:

prospective WAL -> fixed-gate shadow decisions -> executable settlement

It never commits to the research ledger, never approves a strategy, and never
submits orders. The output is a compact top-level report that points to the two
underlying research-only artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_book_fixed_gate_shadow_policy import (  # noqa: E402
    _policy_from_report,
    run_book_fixed_gate_shadow_policy,
)
from scripts.research.run_book_rule_confirmation import _parse_float_list  # noqa: E402
from scripts.research.run_book_wal_scout import (  # noqa: E402
    StreamSummary,
    summarize_stream,
)
from scripts.research.settle_book_fixed_gate_shadow_decisions import (  # noqa: E402
    settle_book_fixed_gate_shadow_decisions,
)


@dataclass(frozen=True)
class FixedGateShadowWindow:
    start: datetime
    end: datetime
    decision_count: int
    first_decision_time: str
    last_decision_time: str
    history_seconds: int
    horizon_seconds: int
    decision_stride_seconds: int
    settlement_tail_guard_seconds: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--wal-root", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--side", choices=("BUY", "SELL"), default="SELL")
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--target-bps", type=float, default=5.0)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--minimum-executable-count", type=int, default=10)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    parser.add_argument("--settlement-tail-guard-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-fixed-gate-prospective-window",
    )
    arguments = parser.parse_args()
    if arguments.decision_stride_seconds <= 0:
        parser.error("--decision-stride-seconds must be positive")
    if arguments.settlement_tail_guard_seconds < 0:
        parser.error("--settlement-tail-guard-seconds cannot be negative")

    report = run_book_fixed_gate_prospective_window(
        policy_report_path=arguments.policy_report,
        wal_root=arguments.wal_root,
        output_root=arguments.output_root,
        symbol=arguments.symbol.upper(),
        decision_stride_seconds=arguments.decision_stride_seconds,
        side=arguments.side,
        latency_ms=arguments.latency_ms,
        target_bps=arguments.target_bps,
        initial_stop_bps=arguments.initial_stop_bps,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        minimum_executable_count=arguments.minimum_executable_count,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
        settlement_tail_guard_seconds=arguments.settlement_tail_guard_seconds,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_fixed_gate_prospective_window(
    *,
    policy_report_path: Path,
    wal_root: Path,
    output_root: Path,
    symbol: str = "BTCUSDT",
    decision_stride_seconds: int = 5,
    side: str = "SELL",
    latency_ms: int = 100,
    target_bps: float = 5.0,
    initial_stop_bps: float = 20.0,
    max_book_staleness_ms: int = 1_000,
    minimum_executable_count: int = 10,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    settlement_tail_guard_seconds: float = 2.0,
) -> dict[str, object]:
    policy_report = json.loads(policy_report_path.read_text(encoding="utf-8"))
    policy = _policy_from_report(policy_report)
    streams = summarize_required_streams(wal_root)
    window = derive_fixed_gate_shadow_window(
        streams=streams,
        policy=policy,
        decision_stride_seconds=decision_stride_seconds,
        settlement_tail_guard_seconds=settlement_tail_guard_seconds,
    )

    shadow_output_root = output_root / "shadow-policy"
    shadow_report = run_book_fixed_gate_shadow_policy(
        policy_report_path=policy_report_path,
        wal_root=wal_root,
        symbol=symbol,
        start=window.start,
        end=window.end,
        decision_stride_seconds=decision_stride_seconds,
        side=side,
        include_non_entries=False,
        output_root=shadow_output_root,
    )
    settlement_output_root = output_root / f"settlement-{latency_ms}ms"
    settlement_report = settle_book_fixed_gate_shadow_decisions(
        shadow_report_path=Path(str(shadow_report["report_path"])),
        wal_root=wal_root,
        output_root=settlement_output_root,
        latency_ms=latency_ms,
        target_bps=target_bps,
        initial_stop_bps=initial_stop_bps,
        max_book_staleness_ms=max_book_staleness_ms,
        minimum_executable_count=minimum_executable_count,
        cost_levels_bps=cost_levels_bps,
    )

    return _write_report(
        output_root,
        {
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
            "committed": False,
            "dry_run": True,
            "approved_strategy": False,
            "experiment_stage": "book_fixed_gate_prospective_shadow_window",
            "wal_root": str(wal_root),
            "policy_report_path": str(policy_report_path),
            "policy_report_sha256": _sha256_file(policy_report_path),
            "policy_sha256": shadow_report["policy_sha256"],
            "symbol": symbol,
            "side": side,
            "stream_summaries": {
                name: asdict(summary) for name, summary in streams.items()
            },
            "decision_window": {
                **asdict(window),
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "shadow_report_path": shadow_report["report_path"],
            "shadow_report_sha256": _sha256_file(Path(str(shadow_report["report_path"]))),
            "shadow_decisions_path": shadow_report["decisions_path"],
            "shadow_selected_count": shadow_report["selected_count"],
            "shadow_decision_count": shadow_report["decision_count"],
            "settlement_report_path": settlement_report["report_path"],
            "settlement_report_sha256": _sha256_file(
                Path(str(settlement_report["report_path"]))
            ),
            "settlement_status": settlement_report["status"],
            "status": _status_from_settlement(settlement_report),
            "status_reasons": settlement_report.get("status_reasons", []),
            "independent_summary": settlement_report.get("independent_summary"),
            "one_position_summary": settlement_report.get("one_position_summary"),
        },
    )


def summarize_required_streams(wal_root: Path) -> dict[str, StreamSummary]:
    streams = {
        "trade": summarize_stream(wal_root / "trade"),
        "depth": summarize_stream(wal_root / "depth"),
        "spot_trade": summarize_stream(wal_root / "spot_trade"),
        "mark_price": summarize_stream(wal_root / "mark_price"),
    }
    missing = [
        name
        for name in ("trade", "depth", "spot_trade")
        if streams[name].present is not True
    ]
    if missing:
        raise ValueError(
            "fixed gate prospective window requires streams: " + ", ".join(missing)
        )
    return streams


def derive_fixed_gate_shadow_window(
    *,
    streams: dict[str, StreamSummary],
    policy: dict[str, Any],
    decision_stride_seconds: int,
    settlement_tail_guard_seconds: float = 2.0,
) -> FixedGateShadowWindow:
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if settlement_tail_guard_seconds < 0:
        raise ValueError("settlement_tail_guard_seconds cannot be negative")
    required = ("trade", "depth", "spot_trade")
    missing = [name for name in required if not streams[name].present]
    if missing:
        raise ValueError("missing required streams: " + ", ".join(missing))

    history_seconds = max(int(value) for value in policy["feature_windows_seconds"])
    horizon_seconds = int(policy["horizon_seconds"])
    stride_ms = decision_stride_seconds * 1_000
    history_ms = history_seconds * 1_000
    horizon_ms = horizon_seconds * 1_000
    tail_guard_ms = int(settlement_tail_guard_seconds * 1_000)

    first_available_ms = max(_stream_first_ms(streams[name]) for name in required)
    first_decision_ms = _ceil_to_stride(first_available_ms + history_ms, stride_ms)

    last_feature_ms = min(
        _stream_last_ms(streams["trade"]),
        _stream_last_ms(streams["spot_trade"]),
    )
    last_settle_safe_ms = _stream_last_ms(streams["depth"]) - horizon_ms - tail_guard_ms
    last_decision_limit_ms = min(last_feature_ms, last_settle_safe_ms)
    last_decision_ms = _floor_to_stride(last_decision_limit_ms, stride_ms)
    if last_decision_ms < first_decision_ms:
        raise ValueError("WAL interval has no complete fixed-gate shadow decisions")

    decision_count = ((last_decision_ms - first_decision_ms) // stride_ms) + 1
    return FixedGateShadowWindow(
        start=_dt(first_decision_ms),
        end=_dt(last_decision_ms + stride_ms),
        decision_count=int(decision_count),
        first_decision_time=_iso(first_decision_ms),
        last_decision_time=_iso(last_decision_ms),
        history_seconds=history_seconds,
        horizon_seconds=horizon_seconds,
        decision_stride_seconds=decision_stride_seconds,
        settlement_tail_guard_seconds=settlement_tail_guard_seconds,
    )


def _status_from_settlement(settlement_report: dict[str, object]) -> str:
    if settlement_report.get("status") == "SHADOW_SETTLEMENT_READY_FOR_PROSPECTIVE_STREAM":
        return "PROSPECTIVE_SHADOW_WINDOW_SETTLED_POSITIVE_AT_1BPS"
    if settlement_report.get("status") == "SHADOW_SETTLEMENT_NO_ENTRY_DECISIONS":
        return "PROSPECTIVE_SHADOW_WINDOW_NO_ENTRIES"
    return "PROSPECTIVE_SHADOW_WINDOW_SETTLED_FAILED_GATE"


def _stream_first_ms(summary: StreamSummary) -> int:
    if summary.first_event_time_ms is None:
        raise ValueError(f"{summary.name} stream has no first_event_time_ms")
    return summary.first_event_time_ms


def _stream_last_ms(summary: StreamSummary) -> int:
    if summary.last_event_time_ms is None:
        raise ValueError(f"{summary.name} stream has no last_event_time_ms")
    return summary.last_event_time_ms


def _ceil_to_stride(value_ms: int, stride_ms: int) -> int:
    return int(math.ceil(value_ms / stride_ms) * stride_ms)


def _floor_to_stride(value_ms: int, stride_ms: int) -> int:
    return int(math.floor(value_ms / stride_ms) * stride_ms)


def _dt(event_time_ms: int) -> datetime:
    return datetime.fromtimestamp(event_time_ms / 1000, tz=UTC)


def _iso(event_time_ms: int) -> str:
    return _dt(event_time_ms).isoformat().replace("+00:00", "Z")


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
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "shadow_decision_count": report["shadow_decision_count"],
        "shadow_selected_count": report["shadow_selected_count"],
        "settlement_report_path": report["settlement_report_path"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

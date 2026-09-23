"""Fast81 updater for the preregistered Fast80 observation.

Fast81 is an orchestration guard, not a strategy. It finds WAL captures that
extend beyond the Fast80 preregistration timestamp and, only when they exist,
chains Fast60 -> Fast61 -> Fast80 into a single research-only report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast57_prospective_update import DEFAULT_ARTIFACT_REPORT  # noqa: E402
from scripts.research.run_fast60_fast57_shadow_dry_run import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    InferredWalInterval,
    WalRootRef,
    discover_default_wal_roots,
    infer_wal_trade_interval,
    parse_dates,
    parse_wal_root_refs,
    run_fast60_fast57_shadow_dry_run,
)
from scripts.research.run_fast61_fast60_shadow_settlement import (  # noqa: E402
    DEFAULT_POLICY_NAME,
    run_fast61_fast60_shadow_settlement,
)
from scripts.research.run_fast80_rare_vol_flow_observation import (  # noqa: E402
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_TIMESTAMP,
    run_fast80_rare_vol_flow_observation,
)


DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT / "reports" / "fast81-fast80-prospective-update-current"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, default=DEFAULT_ARTIFACT_REPORT)
    parser.add_argument(
        "--wal-roots",
        nargs="+",
        help="Optional LABEL=path WAL roots. Defaults to prospective-wal-fast* roots.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dates", help="Optional comma-separated UTC dates to inspect/score.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--lookback-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-complete-outcomes", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    report = run_fast81_fast80_prospective_update(
        artifact_report=arguments.artifact_report,
        wal_roots=(
            parse_wal_root_refs(arguments.wal_roots)
            if arguments.wal_roots
            else discover_default_wal_roots(arguments.data_root)
        ),
        dates=parse_dates(arguments.dates) if arguments.dates else None,
        symbol=arguments.symbol.upper(),
        decision_stride_seconds=arguments.decision_stride_seconds,
        warmup_seconds=arguments.warmup_seconds,
        lookback_seconds=arguments.lookback_seconds,
        limit=arguments.limit,
        policy_name=arguments.policy_name,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_complete_outcomes=arguments.minimum_complete_outcomes,
        output_root=arguments.output_root,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast81_fast80_prospective_update(
    *,
    artifact_report: Path = DEFAULT_ARTIFACT_REPORT,
    wal_roots: tuple[WalRootRef, ...],
    dates: tuple[date, ...] | None = None,
    symbol: str = "BTCUSDT",
    decision_stride_seconds: int = 5,
    warmup_seconds: int = 60,
    lookback_seconds: int = 300,
    limit: int = 0,
    policy_name: str = DEFAULT_POLICY_NAME,
    latency_ms: int = 100,
    expected_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    minimum_complete_outcomes: int = 10,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    _validate_parameters(
        decision_stride_seconds=decision_stride_seconds,
        warmup_seconds=warmup_seconds,
        lookback_seconds=lookback_seconds,
        limit=limit,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cutoff = fast80_preregistration_datetime()
    wal_audit = audit_wal_roots_for_fast80(
        wal_roots=wal_roots,
        dates=dates,
        preregistration_timestamp=cutoff,
    )
    selected_roots = tuple(item.root for item in wal_audit if item.selected)
    if not selected_roots:
        status, status_reasons = fast81_no_selection_status(wal_roots=wal_roots, audit=wal_audit)
        return _write_report(
            output_root,
            base_report(
                status=status,
                status_reasons=status_reasons,
                wal_roots=wal_roots,
                wal_audit=wal_audit,
            ),
        )

    fast60_report = run_fast60_fast57_shadow_dry_run(
        artifact_report=artifact_report,
        wal_roots=selected_roots,
        dates=dates,
        symbol=symbol,
        decision_stride_seconds=decision_stride_seconds,
        warmup_seconds=warmup_seconds,
        lookback_seconds=lookback_seconds,
        limit=limit,
        output_root=output_root / "fast60",
    )
    fast61_report = run_fast61_fast60_shadow_settlement(
        fast60_report_path=Path(str(fast60_report["report_path"])),
        output_root=output_root / "fast61",
        policy_name=policy_name,
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_complete_outcomes=minimum_complete_outcomes,
    )
    fast80_report = run_fast80_rare_vol_flow_observation(
        input_path=Path(str(fast61_report["one_position_settlements_path"])),
        output_root=output_root / "fast80",
    )
    status, status_reasons = fast81_chained_status(fast80_report)
    return _write_report(
        output_root,
        {
            **base_report(
                status=status,
                status_reasons=status_reasons,
                wal_roots=wal_roots,
                wal_audit=wal_audit,
            ),
            "fast60": child_report_summary(fast60_report),
            "fast61": child_report_summary(fast61_report),
            "fast80": child_report_summary(fast80_report),
            "fast80_gates": fast80_report.get("gates"),
            "fast80_summary": fast80_report.get("summary"),
        },
    )


class WalAuditItem:
    def __init__(
        self,
        *,
        root: WalRootRef,
        interval: InferredWalInterval | None,
        selected: bool,
        status: str,
    ) -> None:
        self.root = root
        self.interval = interval
        self.selected = selected
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.root.label,
            "wal_root": str(self.root.path),
            "selected_for_fast80_update": self.selected,
            "status": self.status,
        }
        if self.interval is not None:
            payload["interval"] = {
                "first_trade": self.interval.start.isoformat(),
                "last_trade": self.interval.end.isoformat(),
                "trade_event_count": self.interval.trade_event_count,
                "dates": list(self.interval.dates),
            }
        return payload


def audit_wal_roots_for_fast80(
    *,
    wal_roots: tuple[WalRootRef, ...],
    dates: tuple[date, ...] | None,
    preregistration_timestamp: datetime,
) -> tuple[WalAuditItem, ...]:
    audit: list[WalAuditItem] = []
    for root in wal_roots:
        interval = infer_wal_trade_interval(root.path, dates=dates)
        if interval is None:
            audit.append(
                WalAuditItem(
                    root=root,
                    interval=None,
                    selected=False,
                    status="FAST81_WAL_HAS_NO_TRADE_EVENTS",
                )
            )
            continue
        selected = interval.end > preregistration_timestamp
        audit.append(
            WalAuditItem(
                root=root,
                interval=interval,
                selected=selected,
                status=(
                    "FAST81_WAL_OVERLAPS_POST_PREREGISTRATION"
                    if selected
                    else "FAST81_WAL_ENDS_BEFORE_PREREGISTRATION"
                ),
            )
        )
    return tuple(audit)


def fast81_no_selection_status(
    *,
    wal_roots: tuple[WalRootRef, ...],
    audit: tuple[WalAuditItem, ...],
) -> tuple[str, list[str]]:
    base = ["not_approved_strategy", "research_only", "no_order_execution"]
    if not wal_roots:
        return "FAST81_NO_WAL_ROOTS_FOUND", base + ["no_wal_roots_to_score"]
    if not any(item.interval is not None for item in audit):
        return "FAST81_WAL_ROOTS_HAVE_NO_TRADE_EVENTS", base + ["no_trade_events_in_wal_roots"]
    return "FAST81_NO_POST_PREREG_WAL_ROOTS", base + [
        "waiting_for_wal_capture_after_fast80_preregistration"
    ]


def fast81_chained_status(fast80_report: dict[str, Any]) -> tuple[str, list[str]]:
    fast80_status = str(fast80_report["status"])
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        f"fast80_status:{fast80_status}",
    ]
    if fast80_status == "FAST80_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW":
        return "FAST81_FAST80_SHADOW_CANDIDATE_REQUIRES_REVIEW", reasons + [
            "testnet_still_requires_new_preregistration_and_explicit_authorization"
        ]
    return "FAST81_FAST80_UPDATE_RESEARCH_ONLY", reasons


def base_report(
    *,
    status: str,
    status_reasons: list[str],
    wal_roots: tuple[WalRootRef, ...],
    wal_audit: tuple[WalAuditItem, ...],
) -> dict[str, Any]:
    selected = [item for item in wal_audit if item.selected]
    latest_trade = max(
        (item.interval.end for item in wal_audit if item.interval is not None),
        default=None,
    )
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast81_orchestrates_research_only_observation",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_preregistration_and_explicit_authorization",
        ],
        "experiment_stage": "fast81_fast80_prospective_update",
        "status": status,
        "status_reasons": status_reasons,
        "fast80_preregistration_commit": PREREGISTRATION_COMMIT,
        "fast80_preregistration_timestamp": fast80_preregistration_datetime().isoformat(),
        "wal_root_count": len(wal_roots),
        "wal_roots_with_trade_events": sum(item.interval is not None for item in wal_audit),
        "post_preregistration_wal_root_count": len(selected),
        "latest_local_wal_trade_time": latest_trade.isoformat() if latest_trade else None,
        "wal_audit": [item.to_dict() for item in wal_audit],
        "created_at": datetime.now(tz=UTC).isoformat(),
    }


def child_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(str(report["report_path"]))
    summary = {
        "status": report.get("status"),
        "status_reasons": report.get("status_reasons"),
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
    }
    for key in (
        "aggregate",
        "input_record_count",
        "post_preregistration_record_count",
        "candidate_universe_count",
        "rule_match_count",
        "eligible_observation_count",
    ):
        if key in report:
            summary[key] = report[key]
    return summary


def fast80_preregistration_datetime() -> datetime:
    return PREREGISTRATION_TIMESTAMP.to_pydatetime().astimezone(UTC)


def _validate_parameters(
    *,
    decision_stride_seconds: int,
    warmup_seconds: int,
    lookback_seconds: int,
    limit: int,
) -> None:
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds cannot be negative")
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if limit < 0:
        raise ValueError("limit cannot be negative")


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


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
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "wal_root_count": report["wal_root_count"],
        "post_preregistration_wal_root_count": report["post_preregistration_wal_root_count"],
        "latest_local_wal_trade_time": report["latest_local_wal_trade_time"],
        "fast80": report.get("fast80"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

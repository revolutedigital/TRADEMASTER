"""Fast69 frozen directional gate shadow dry-run.

Fast68 found a diagnostic survivor, but it is not a strategy because the
extension block participated in the search. Fast69 freezes that survivor into a
deterministic policy artifact and applies it to WAL data without changing the
rule, writing local research-only shadow decisions. No ledger rows and no
orders are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast60_fast57_shadow_dry_run import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    WalRootRef,
    align_datetime_to_stride,
    book_context_for_decision,
    build_book_context,
    causal_feature_values,
    discover_default_wal_roots,
    infer_wal_trade_interval,
    parse_dates,
    parse_wal_root_refs,
    write_jsonl,
)
from scripts.research.run_fast68_directional_liquidity_volatility_gate_diagnostic import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as DEFAULT_FAST68_OUTPUT_ROOT,
)
from scripts.research.run_online_shadow_policy import (  # noqa: E402
    EventRoots,
    decision_times_between,
    load_online_microstructure_events,
)
from app.services.research.online_shadow_policy import (  # noqa: E402
    score_online_frozen_top_p_shadow_events,
)


DEFAULT_FAST56_REPORT = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms"
    / "report.json"
)
DEFAULT_FAST68_REPORT = DEFAULT_FAST68_OUTPUT_ROOT / "report.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast69-directional-shadow-dry-run"
DEFAULT_POLICY_ARTIFACT = (
    DEFAULT_DATA_ROOT / "models" / "fast69-directional-liquidity-volatility-shadow-policy.json"
)


@dataclass(frozen=True)
class DirectionalGate:
    quote_volume_floor: float
    realized_vol_floor: float
    quote_volume_cap: float | None
    realized_vol_cap: float | None
    directional_feature: str
    directional_floor: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-p-report", type=Path, default=DEFAULT_FAST56_REPORT)
    parser.add_argument("--fast68-report", type=Path, default=DEFAULT_FAST68_REPORT)
    parser.add_argument("--policy-artifact", type=Path, default=DEFAULT_POLICY_ARTIFACT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--wal-roots", nargs="+")
    parser.add_argument("--dates", help="Comma-separated UTC dates to score.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    if arguments.decision_stride_seconds <= 0:
        parser.error("--decision-stride-seconds must be positive")
    if arguments.warmup_seconds < 0:
        parser.error("--warmup-seconds cannot be negative")

    wal_roots = (
        parse_wal_root_refs(arguments.wal_roots)
        if arguments.wal_roots
        else discover_default_wal_roots(arguments.data_root)
    )
    if not wal_roots:
        parser.error("no WAL roots found")

    report = run_fast69_directional_shadow_dry_run(
        top_p_report_path=arguments.top_p_report,
        fast68_report_path=arguments.fast68_report,
        policy_artifact_path=arguments.policy_artifact,
        wal_roots=wal_roots,
        dates=parse_dates(arguments.dates) if arguments.dates else None,
        symbol=arguments.symbol.upper(),
        decision_stride_seconds=arguments.decision_stride_seconds,
        warmup_seconds=arguments.warmup_seconds,
        output_root=arguments.output_root,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast69_directional_shadow_dry_run(
    *,
    top_p_report_path: Path,
    fast68_report_path: Path,
    policy_artifact_path: Path,
    wal_roots: tuple[WalRootRef, ...],
    dates: tuple[date, ...] | None,
    symbol: str,
    decision_stride_seconds: int,
    warmup_seconds: int,
    output_root: Path,
) -> dict[str, Any]:
    top_p_report = json.loads(top_p_report_path.read_text(encoding="utf-8"))
    fast68_report = json.loads(fast68_report_path.read_text(encoding="utf-8"))
    artifact = build_fast69_policy_artifact(
        top_p_report=top_p_report,
        fast68_report=fast68_report,
        top_p_report_path=top_p_report_path,
        fast68_report_path=fast68_report_path,
    )
    policy_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    policy_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = directional_gate_from_artifact(artifact)
    top_p_artifact = artifact["base_top_p_artifact"]

    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    all_raw_decisions: list[dict[str, Any]] = []
    all_gated_decisions: list[dict[str, Any]] = []
    for wal_root in wal_roots:
        interval = infer_wal_trade_interval(wal_root.path, dates=dates)
        if interval is None:
            runs.append(
                {
                    "label": wal_root.label,
                    "wal_root": str(wal_root.path),
                    "status": "FAST69_WAL_HAS_NO_TRADE_EVENTS",
                    "status_reasons": ["trade_wal_missing_or_empty"],
                    "raw_top_p_count": 0,
                    "gated_selected_count": 0,
                }
            )
            continue
        start = align_datetime_to_stride(
            interval.start + timedelta(seconds=warmup_seconds),
            stride_seconds=decision_stride_seconds,
        )
        end = interval.end
        if end <= start:
            runs.append(
                {
                    "label": wal_root.label,
                    "wal_root": str(wal_root.path),
                    "status": "FAST69_WAL_INTERVAL_TOO_SHORT",
                    "status_reasons": ["warmup_consumes_interval"],
                    "raw_top_p_count": 0,
                    "gated_selected_count": 0,
                }
            )
            continue
        loaded = load_online_microstructure_events(
            artifact=top_p_artifact,
            symbol=symbol,
            start=start,
            end=end,
            history_seconds=warmup_seconds,
            roots=EventRoots(
                trade=wal_root.path / "trade",
                book=wal_root.path / "depth",
                mark=wal_root.path / "mark_price",
                liquidation=wal_root.path / "liquidation",
                spot=wal_root.path / "spot_trade",
            ),
        )
        decision_times_ms = decision_times_between(
            start,
            end,
            stride_seconds=decision_stride_seconds,
        )
        selection = score_online_frozen_top_p_shadow_events(
            loaded.events,
            artifact=top_p_artifact,
            decision_times_ms=decision_times_ms,
            sides=("BUY", "SELL"),
            include_non_entries=False,
        )
        book_context = build_book_context(loaded.events)
        raw_decisions = [
            shadow_decision_to_fast69_row(decision, label=wal_root.label, gate=gate)
            for decision in selection.decisions
        ]
        for row in raw_decisions:
            row.update(book_context_for_decision(book_context, int(row["decision_time_ms"])))
        gated_decisions = [
            row for row in raw_decisions if bool(row["directional_gate_passed"])
        ]
        all_raw_decisions.extend(raw_decisions)
        all_gated_decisions.extend(gated_decisions)
        run_status, run_reasons = fast69_run_status(
            raw_top_p_count=len(raw_decisions),
            gated_selected_count=len(gated_decisions),
        )
        runs.append(
            {
                "label": wal_root.label,
                "wal_root": str(wal_root.path),
                "status": run_status,
                "status_reasons": run_reasons,
                "dates": list(interval.dates),
                "interval": {
                    "first_trade": interval.start.isoformat(),
                    "last_trade": interval.end.isoformat(),
                    "decision_start": start.isoformat(),
                    "decision_end": end.isoformat(),
                },
                "trade_event_count": interval.trade_event_count,
                "event_counts": loaded.counts,
                "decision_count": len(decision_times_ms),
                "scored_rows": selection.scored_rows,
                "raw_top_p_count": len(raw_decisions),
                "gated_selected_count": len(gated_decisions),
            }
        )

    raw_path = output_root / "raw_top_p_decisions.jsonl"
    gated_path = output_root / "gated_shadow_decisions.jsonl"
    write_jsonl(raw_path, all_raw_decisions)
    write_jsonl(gated_path, all_gated_decisions)
    aggregate = {
        "wal_root_count": len(wal_roots),
        "run_count": len(runs),
        "raw_top_p_count": len(all_raw_decisions),
        "gated_selected_count": len(all_gated_decisions),
        "runs_with_gated_signals": sum(run.get("gated_selected_count", 0) > 0 for run in runs),
    }
    status, status_reasons = fast69_status(aggregate)
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
            "fast69_shadow_dry_run_research_only",
            "local_wal_scoring_does_not_write_ledger",
            "fast68_survivor_requires_clean_future_shadow",
            "shadow_outcomes_still_required",
        ],
        "experiment_stage": "fast69_directional_shadow_dry_run",
        "status": status,
        "status_reasons": status_reasons,
        "candidate": {
            "model_sha256": artifact["model_sha256"],
            "threshold": float(top_p_artifact["probability_threshold"]),
            "tail_fraction": float(top_p_artifact["tail_fraction"]),
            "horizon_seconds": int(top_p_artifact["horizon_seconds"]),
            "directional_gate": artifact["directional_gate"],
        },
        "policy_artifact_path": str(policy_artifact_path),
        "policy_artifact_sha256": _sha256_file(policy_artifact_path),
        "top_p_report_path": str(top_p_report_path),
        "top_p_report_sha256": _sha256_file(top_p_report_path),
        "fast68_report_path": str(fast68_report_path),
        "fast68_report_sha256": _sha256_file(fast68_report_path),
        "decision_stride_seconds": decision_stride_seconds,
        "warmup_seconds": warmup_seconds,
        "aggregate": aggregate,
        "runs": runs,
        "raw_decisions_path": str(raw_path),
        "raw_decisions_sha256": _sha256_file(raw_path),
        "gated_decisions_path": str(gated_path),
        "gated_decisions_sha256": _sha256_file(gated_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_fast69_policy_artifact(
    *,
    top_p_report: dict[str, Any],
    fast68_report: dict[str, Any],
    top_p_report_path: Path,
    fast68_report_path: Path,
) -> dict[str, Any]:
    _validate_top_p_report(top_p_report)
    _validate_fast68_report(fast68_report)
    top_candidate = fast68_report["top_candidate"]
    top_p_artifact = top_p_report["frozen_artifact"]
    if top_candidate["spec"]["directional_feature"] is None:
        raise ValueError("Fast69 requires a directional Fast68 candidate")
    return {
        "artifact_kind": "research_directional_liquidity_volatility_shadow_policy",
        "schema_version": 1,
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "orders_sent": False,
        "source": {
            "top_p_report_path": str(top_p_report_path),
            "top_p_report_sha256": _sha256_file(top_p_report_path),
            "fast68_report_path": str(fast68_report_path),
            "fast68_report_sha256": _sha256_file(fast68_report_path),
            "fast68_status": fast68_report["status"],
        },
        "model_sha256": str(top_p_report["model_sha256"]),
        "base_top_p_artifact": top_p_artifact,
        "directional_gate": {
            "spec": top_candidate["spec"],
            "thresholds": top_candidate["frozen_thresholds"],
            "historical_aggregate": top_candidate["historical_aggregate"],
            "extension_aggregate": top_candidate["extension_aggregate"],
        },
        "clean_shadow_requirements": {
            "minimum_complete_utc_days": 20,
            "maximum_complete_utc_days": 30,
            "minimum_one_position_trades": 100,
            "stress_total_log_return_bps_must_be_positive": True,
            "stress_mean_bps_must_be_positive": True,
            "negative_sufficient_block_allowed": False,
            "clean_data_starts_after": "2026-09-23",
        },
    }


def directional_gate_from_artifact(artifact: dict[str, Any]) -> DirectionalGate:
    thresholds = artifact["directional_gate"]["thresholds"]
    return DirectionalGate(
        quote_volume_floor=float(thresholds["quote_volume_floor"]),
        realized_vol_floor=float(thresholds["realized_vol_floor"]),
        quote_volume_cap=(
            None if thresholds.get("quote_volume_cap") is None else float(thresholds["quote_volume_cap"])
        ),
        realized_vol_cap=(
            None if thresholds.get("realized_vol_cap") is None else float(thresholds["realized_vol_cap"])
        ),
        directional_feature=str(thresholds["directional_feature"]),
        directional_floor=float(thresholds["directional_floor"]),
    )


def shadow_decision_to_fast69_row(
    decision: Any,
    *,
    label: str,
    gate: DirectionalGate,
) -> dict[str, Any]:
    decision_time_ms = int(decision.decision_time.timestamp() * 1000)
    feature_values = causal_feature_values(decision)
    gate_values = {
        "quote_volume_30s": float(feature_values["quote_volume_30s"]),
        "realized_vol_30s_bps": float(feature_values["realized_vol_30s_bps"]),
        gate.directional_feature: float(feature_values[gate.directional_feature]),
    }
    gate_passed = directional_gate_passed(gate_values, gate)
    row = {
        "label": label,
        "decision_time": decision.decision_time.isoformat(),
        "decision_time_ms": decision_time_ms,
        "side": decision.side,
        "horizon_seconds": decision.horizon_seconds,
        "probability": decision.probability,
        "threshold": decision.threshold,
        "would_enter": bool(decision.would_enter and gate_passed),
        "top_p_would_enter": decision.would_enter,
        "directional_gate_passed": gate_passed,
        "directional_gate_values": gate_values,
        "model_sha256": decision.model_sha256,
        "feature_vector_sha256": decision.feature_vector_sha256,
        "causal_feature_values": feature_values,
    }
    for feature in (
        "quote_volume_30s",
        "realized_vol_30s_bps",
        "directed_flow_imbalance_30s",
        "flow_imbalance_5s",
    ):
        if feature in feature_values:
            row[feature] = feature_values[feature]
    return row


def directional_gate_passed(values: dict[str, float], gate: DirectionalGate) -> bool:
    if values["quote_volume_30s"] < gate.quote_volume_floor:
        return False
    if values["realized_vol_30s_bps"] < gate.realized_vol_floor:
        return False
    if gate.quote_volume_cap is not None and values["quote_volume_30s"] > gate.quote_volume_cap:
        return False
    if gate.realized_vol_cap is not None and values["realized_vol_30s_bps"] > gate.realized_vol_cap:
        return False
    return values[gate.directional_feature] >= gate.directional_floor


def fast69_run_status(
    *,
    raw_top_p_count: int,
    gated_selected_count: int,
) -> tuple[str, list[str]]:
    if raw_top_p_count <= 0:
        return "FAST69_SHADOW_DRY_RUN_NO_TOP_P_SIGNALS", ["no_top_p_signals"]
    if gated_selected_count <= 0:
        return (
            "FAST69_DIRECTIONAL_GATE_FILTERED_ALL",
            [f"raw_top_p_signals:{raw_top_p_count}", "directional_gate_filtered_all"],
        )
    return (
        "FAST69_DIRECTIONAL_SHADOW_SIGNALS_RESEARCH_ONLY",
        ["not_approved_strategy", "dry_run_no_ledger_write"],
    )


def fast69_status(aggregate: dict[str, int]) -> tuple[str, list[str]]:
    raw_count = int(aggregate["raw_top_p_count"])
    gated_count = int(aggregate["gated_selected_count"])
    if raw_count <= 0:
        return "FAST69_NO_TOP_P_SIGNALS", ["no_top_p_signals_in_wal_roots"]
    if gated_count <= 0:
        return (
            "FAST69_DIRECTIONAL_GATE_FILTERED_ALL_SIGNALS",
            [f"raw_top_p_signals:{raw_count}", "do_not_promote_to_strategy"],
        )
    return (
        "FAST69_DIRECTIONAL_SHADOW_SIGNALS_RESEARCH_ONLY",
        [f"gated_signals:{gated_count}", "not_approved_strategy", "shadow_outcomes_required"],
    )


def _validate_top_p_report(report: dict[str, Any]) -> None:
    if report.get("research_only") is not True:
        raise ValueError("top-p report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError("top-p report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError("top-p report must have execution_authorization=none")
    if report.get("approved_strategy") is not False:
        raise ValueError("top-p report must not be an approved strategy")
    artifact = report.get("frozen_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("top-p report is missing frozen_artifact")
    if artifact.get("research_only") is not True:
        raise ValueError("top-p artifact must be research_only")
    if artifact.get("order_submission_allowed") is not False:
        raise ValueError("top-p artifact must not allow orders")
    if artifact.get("execution_authorization") != "none":
        raise ValueError("top-p artifact must have execution_authorization=none")


def _validate_fast68_report(report: dict[str, Any]) -> None:
    if report.get("research_only") is not True:
        raise ValueError("Fast68 report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError("Fast68 report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError("Fast68 report must have execution_authorization=none")
    if report.get("approved_strategy") is not False:
        raise ValueError("Fast68 report must not be an approved strategy")
    if not report.get("top_candidate"):
        raise ValueError("Fast68 report is missing top_candidate")


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
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate": report["candidate"],
        "aggregate": report["aggregate"],
        "policy_artifact_path": report["policy_artifact_path"],
        "policy_artifact_sha256": report["policy_artifact_sha256"],
        "raw_decisions_path": report["raw_decisions_path"],
        "gated_decisions_path": report["gated_decisions_path"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

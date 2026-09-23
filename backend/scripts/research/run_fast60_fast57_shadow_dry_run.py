"""Fast60 live-like dry-run for the frozen Fast57 candidate over WAL captures.

Fast60 scores the Fast57 frozen top-p artifact on chronological WAL events,
applies the fixed density gate (`recent_300s_lte_0`), and writes only local
research artifacts. It never writes to the shadow ledger and never sends orders.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.microstructure import MarketEventType, MicrostructureEvent  # noqa: E402
from scripts.research.run_fast57_prospective_update import DEFAULT_ARTIFACT_REPORT  # noqa: E402
from scripts.research.run_online_shadow_policy import (  # noqa: E402
    EventRoots,
    decision_times_between,
    load_online_microstructure_events,
)
from app.services.research.online_shadow_policy import (  # noqa: E402
    score_online_frozen_top_p_shadow_events,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast60-fast57-shadow-dry-run"


@dataclass(frozen=True)
class WalRootRef:
    label: str
    path: Path


@dataclass(frozen=True)
class InferredWalInterval:
    start: datetime
    end: datetime
    trade_event_count: int
    dates: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, default=DEFAULT_ARTIFACT_REPORT)
    parser.add_argument(
        "--wal-roots",
        nargs="+",
        help="Optional LABEL=path WAL roots. Defaults to prospective-wal-fast* roots.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dates", help="Comma-separated UTC dates to score.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--lookback-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    if arguments.decision_stride_seconds <= 0:
        parser.error("--decision-stride-seconds must be positive")
    if arguments.warmup_seconds < 0:
        parser.error("--warmup-seconds cannot be negative")
    if arguments.lookback_seconds <= 0:
        parser.error("--lookback-seconds must be positive")
    if arguments.limit < 0:
        parser.error("--limit cannot be negative")

    wal_roots = (
        parse_wal_root_refs(arguments.wal_roots)
        if arguments.wal_roots
        else discover_default_wal_roots(arguments.data_root)
    )
    if not wal_roots:
        parser.error("no WAL roots found")
    report = run_fast60_fast57_shadow_dry_run(
        artifact_report=arguments.artifact_report,
        wal_roots=wal_roots,
        dates=parse_dates(arguments.dates) if arguments.dates else None,
        symbol=arguments.symbol.upper(),
        decision_stride_seconds=arguments.decision_stride_seconds,
        warmup_seconds=arguments.warmup_seconds,
        lookback_seconds=arguments.lookback_seconds,
        limit=arguments.limit,
        output_root=arguments.output_root,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast60_fast57_shadow_dry_run(
    *,
    artifact_report: Path,
    wal_roots: tuple[WalRootRef, ...],
    dates: tuple[date, ...] | None,
    symbol: str,
    decision_stride_seconds: int,
    warmup_seconds: int,
    lookback_seconds: int,
    limit: int,
    output_root: Path,
) -> dict[str, Any]:
    artifact_payload = json.loads(artifact_report.read_text(encoding="utf-8"))
    artifact = artifact_payload["frozen_artifact"]
    runs = []
    all_raw_decisions: list[dict[str, Any]] = []
    all_gated_decisions: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for wal_root in wal_roots:
        interval = infer_wal_trade_interval(wal_root.path, dates=dates)
        if interval is None:
            runs.append(
                {
                    "label": wal_root.label,
                    "wal_root": str(wal_root.path),
                    "status": "FAST60_WAL_HAS_NO_TRADE_EVENTS",
                    "status_reasons": ["trade_wal_missing_or_empty"],
                    "raw_selected_count": 0,
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
                    "status": "FAST60_WAL_INTERVAL_TOO_SHORT",
                    "status_reasons": ["warmup_consumes_interval"],
                    "raw_selected_count": 0,
                    "gated_selected_count": 0,
                }
            )
            continue
        loaded = load_online_microstructure_events(
            artifact=artifact,
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
            artifact=artifact,
            decision_times_ms=decision_times_ms,
            sides=("BUY", "SELL"),
            include_non_entries=False,
        )
        raw_decisions = [
            shadow_decision_to_row(decision, label=wal_root.label)
            for decision in selection.decisions
        ]
        gated_decisions = apply_fast57_density_gate(
            raw_decisions,
            lookback_seconds=lookback_seconds,
            limit=limit,
        )
        book_context = build_book_context(loaded.events)
        for row in raw_decisions:
            row.update(book_context_for_decision(book_context, int(row["decision_time_ms"])))
        for row in gated_decisions:
            row.update(book_context_for_decision(book_context, int(row["decision_time_ms"])))
        all_raw_decisions.extend(raw_decisions)
        all_gated_decisions.extend(gated_decisions)
        run_status, run_reasons = fast60_run_status(
            raw_selected_count=len(raw_decisions),
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
                "raw_selected_count": len(raw_decisions),
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
        "raw_selected_count": len(all_raw_decisions),
        "gated_selected_count": len(all_gated_decisions),
        "runs_with_gated_signals": sum(run.get("gated_selected_count", 0) > 0 for run in runs),
    }
    status, status_reasons = fast60_status(aggregate)
    report_path = output_root / "report.json"
    report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "committed": False,
        "dry_run": True,
        "approval_reasons": [
            "fast60_shadow_dry_run_research_only",
            "local_wal_scoring_does_not_write_ledger",
            "shadow_outcomes_and_book_evidence_still_required",
        ],
        "experiment_stage": "fast60_fast57_shadow_dry_run",
        "status": status,
        "status_reasons": status_reasons,
        "candidate": {
            "model_sha256": artifact_payload["model_sha256"],
            "threshold": float(artifact["probability_threshold"]),
            "tail_fraction": float(artifact["tail_fraction"]),
            "horizon_seconds": int(artifact["horizon_seconds"]),
            "density_gate": {
                "lookback_seconds": lookback_seconds,
                "limit": limit,
            },
        },
        "decision_stride_seconds": decision_stride_seconds,
        "warmup_seconds": warmup_seconds,
        "aggregate": aggregate,
        "runs": runs,
        "raw_decisions_path": str(raw_path),
        "gated_decisions_path": str(gated_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def discover_default_wal_roots(data_root: Path) -> tuple[WalRootRef, ...]:
    roots = []
    for path in sorted(data_root.glob("prospective-wal-fast*")):
        if path.is_dir() and (path / "trade").exists():
            roots.append(WalRootRef(label=path.name, path=path))
    return tuple(roots)


def parse_wal_root_refs(raw_roots: list[str]) -> tuple[WalRootRef, ...]:
    refs = []
    for raw in raw_roots:
        label, separator, path = raw.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid WAL root spec: {raw!r}")
        refs.append(WalRootRef(label=label, path=Path(path)))
    return tuple(refs)


def parse_dates(raw: str) -> tuple[date, ...]:
    parsed = []
    for part in raw.split(","):
        item = part.strip()
        if item:
            parsed.append(date.fromisoformat(item))
    if not parsed:
        raise ValueError("dates cannot be empty")
    return tuple(dict.fromkeys(parsed))


def infer_wal_trade_interval(
    wal_root: Path,
    *,
    dates: tuple[date, ...] | None,
) -> InferredWalInterval | None:
    paths = trade_partition_paths(wal_root, dates=dates)
    first: datetime | None = None
    last: datetime | None = None
    rows = 0
    observed_dates: set[str] = set()
    for path in paths:
        observed_dates.add(path.parent.name.removeprefix("date="))
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                event_time = parse_datetime(str(payload["event_time"]))
                first = event_time if first is None or event_time < first else first
                last = event_time if last is None or event_time > last else last
                rows += 1
    if first is None or last is None:
        return None
    return InferredWalInterval(
        start=first,
        end=last,
        trade_event_count=rows,
        dates=tuple(sorted(observed_dates)),
    )


def trade_partition_paths(wal_root: Path, *, dates: tuple[date, ...] | None) -> tuple[Path, ...]:
    if dates is None:
        return tuple(sorted((wal_root / "trade").glob("date=*/events.jsonl.gz")))
    return tuple(
        path
        for utc_date in dates
        if (path := wal_root / "trade" / f"date={utc_date.isoformat()}" / "events.jsonl.gz").exists()
    )


def align_datetime_to_stride(value: datetime, *, stride_seconds: int) -> datetime:
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    normalized = value.astimezone(UTC)
    stride_ms = stride_seconds * 1000
    timestamp_ms = int(normalized.timestamp() * 1000)
    aligned_ms = ((timestamp_ms + stride_ms - 1) // stride_ms) * stride_ms
    return datetime.fromtimestamp(aligned_ms / 1000, tz=UTC)


def apply_fast57_density_gate(
    decisions: list[dict[str, Any]],
    *,
    lookback_seconds: int,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        decisions,
        key=lambda row: (int(row["decision_time_ms"]), -float(row["probability"])),
    )
    times: list[int] = []
    gated = []
    lookback_ms = lookback_seconds * 1000
    for row in ordered:
        decision_time_ms = int(row["decision_time_ms"])
        recent_count = len(times) - bisect.bisect_left(times, decision_time_ms - lookback_ms)
        enriched = {**row, "recent_signal_count": recent_count}
        if recent_count <= limit:
            gated.append(enriched)
        bisect.insort(times, decision_time_ms)
    return gated


def shadow_decision_to_row(decision: Any, *, label: str) -> dict[str, Any]:
    decision_time_ms = int(decision.decision_time.timestamp() * 1000)
    return {
        "label": label,
        "decision_time": decision.decision_time.isoformat(),
        "decision_time_ms": decision_time_ms,
        "side": decision.side,
        "horizon_seconds": decision.horizon_seconds,
        "probability": decision.probability,
        "threshold": decision.threshold,
        "would_enter": decision.would_enter,
        "model_sha256": decision.model_sha256,
        "feature_vector_sha256": decision.feature_vector_sha256,
    }


def build_book_context(events: tuple[MicrostructureEvent, ...]) -> tuple[dict[str, Any], ...]:
    rows = []
    for event in events:
        if event.event_type != MarketEventType.DEPTH:
            continue
        if event.bid_price is None or event.ask_price is None:
            continue
        event_time_ms = int(event.event_time.timestamp() * 1000)
        spread_bps = math.log(float(event.ask_price) / float(event.bid_price)) * 10_000
        rows.append(
            {
                "event_time_ms": event_time_ms,
                "bid_price": float(event.bid_price),
                "ask_price": float(event.ask_price),
                "spread_bps": spread_bps,
                "top_bid_notional_usdt": float(event.bid_price) * float(event.bid_quantity or 0.0),
                "top_ask_notional_usdt": float(event.ask_price) * float(event.ask_quantity or 0.0),
            }
        )
    return tuple(sorted(rows, key=lambda row: int(row["event_time_ms"])))


def book_context_for_decision(
    book_context: tuple[dict[str, Any], ...],
    decision_time_ms: int,
) -> dict[str, Any]:
    if not book_context:
        return {
            "book_available": False,
            "book_age_ms": None,
            "spread_bps": None,
            "top_bid_notional_usdt": None,
            "top_ask_notional_usdt": None,
        }
    times = [int(row["event_time_ms"]) for row in book_context]
    index = bisect.bisect_right(times, decision_time_ms) - 1
    if index < 0:
        return {
            "book_available": False,
            "book_age_ms": None,
            "spread_bps": None,
            "top_bid_notional_usdt": None,
            "top_ask_notional_usdt": None,
        }
    row = book_context[index]
    return {
        "book_available": True,
        "book_age_ms": decision_time_ms - int(row["event_time_ms"]),
        "spread_bps": row["spread_bps"],
        "top_bid_notional_usdt": row["top_bid_notional_usdt"],
        "top_ask_notional_usdt": row["top_ask_notional_usdt"],
    }


def fast60_run_status(
    *,
    raw_selected_count: int,
    gated_selected_count: int,
) -> tuple[str, list[str]]:
    if raw_selected_count <= 0:
        return "FAST60_SHADOW_DRY_RUN_NO_TOP_P_SIGNALS", ["no_top_p_signals"]
    if gated_selected_count <= 0:
        return (
            "FAST60_SHADOW_DRY_RUN_DENSITY_FILTERED_ALL",
            [f"raw_top_p_signals:{raw_selected_count}", "density_gate_filtered_all"],
        )
    return (
        "FAST60_SHADOW_DRY_RUN_GATED_SIGNALS_RESEARCH_ONLY",
        ["not_approved_strategy", "dry_run_no_ledger_write"],
    )


def fast60_status(aggregate: dict[str, int]) -> tuple[str, list[str]]:
    raw_count = int(aggregate["raw_selected_count"])
    gated_count = int(aggregate["gated_selected_count"])
    if raw_count <= 0:
        return "FAST60_NO_TOP_P_SIGNALS", ["no_top_p_signals_in_wal_roots"]
    if gated_count <= 0:
        return (
            "FAST60_DENSITY_GATE_FILTERED_ALL_SIGNALS",
            [f"raw_top_p_signals:{raw_count}", "do_not_promote_to_strategy"],
        )
    return (
        "FAST60_GATED_SHADOW_SIGNALS_RESEARCH_ONLY",
        [f"gated_signals:{gated_count}", "not_approved_strategy", "shadow_outcomes_required"],
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_datetime(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "approved_strategy": report["approved_strategy"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate": report["candidate"],
        "aggregate": report["aggregate"],
        "raw_decisions_path": report["raw_decisions_path"],
        "gated_decisions_path": report["gated_decisions_path"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

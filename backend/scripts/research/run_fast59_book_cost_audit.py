"""Fast59 top-of-book cost audit for prospective WAL captures.

Fast59 does not approve a strategy and does not send orders. It answers a
narrow execution question raised by Fast58: do our captured book streams show
top-of-book spread/freshness compatible with a very low-cost edge?
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast59-book-cost-audit"


@dataclass(frozen=True)
class WalRootRef:
    label: str
    path: Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wal-roots",
        nargs="+",
        help=(
            "Optional LABEL=path WAL roots. Defaults to discovered "
            "backend/data/microstructure_v1/prospective-wal-fast* roots."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root used only for default WAL discovery.",
    )
    parser.add_argument("--dates", help="Comma-separated UTC dates to audit. Defaults to all dates.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-roundtrip-book-cost-bps", type=float, default=2.0)
    parser.add_argument("--max-p95-receive-delay-ms", type=float, default=1_000.0)
    parser.add_argument("--min-p05-top-notional-usdt", type=float, default=10_000.0)
    parser.add_argument("--min-depth-events", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.max_roundtrip_book_cost_bps <= 0:
        parser.error("--max-roundtrip-book-cost-bps must be positive")
    if arguments.max_p95_receive_delay_ms < 0:
        parser.error("--max-p95-receive-delay-ms cannot be negative")
    if arguments.min_p05_top_notional_usdt < 0:
        parser.error("--min-p05-top-notional-usdt cannot be negative")
    if arguments.min_depth_events <= 0:
        parser.error("--min-depth-events must be positive")

    wal_roots = (
        parse_wal_root_refs(arguments.wal_roots)
        if arguments.wal_roots
        else discover_default_wal_roots(arguments.data_root)
    )
    if not wal_roots:
        parser.error("no WAL roots found")
    dates = parse_dates(arguments.dates) if arguments.dates else None
    report = run_fast59_book_cost_audit(
        wal_roots=wal_roots,
        dates=dates,
        output_root=arguments.output_root,
        max_roundtrip_book_cost_bps=arguments.max_roundtrip_book_cost_bps,
        max_p95_receive_delay_ms=arguments.max_p95_receive_delay_ms,
        min_p05_top_notional_usdt=arguments.min_p05_top_notional_usdt,
        min_depth_events=arguments.min_depth_events,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast59_book_cost_audit(
    *,
    wal_roots: tuple[WalRootRef, ...],
    dates: tuple[date, ...] | None,
    output_root: Path,
    max_roundtrip_book_cost_bps: float,
    max_p95_receive_delay_ms: float,
    min_p05_top_notional_usdt: float,
    min_depth_events: int,
) -> dict[str, Any]:
    partitions: list[dict[str, Any]] = []
    for wal_root in wal_roots:
        for depth_path in depth_partition_paths(wal_root.path, dates=dates):
            utc_date = partition_date(depth_path)
            partitions.append(
                audit_depth_partition(
                    label=wal_root.label,
                    wal_root=wal_root.path,
                    depth_path=depth_path,
                    utc_date=utc_date,
                    max_roundtrip_book_cost_bps=max_roundtrip_book_cost_bps,
                    max_p95_receive_delay_ms=max_p95_receive_delay_ms,
                    min_p05_top_notional_usdt=min_p05_top_notional_usdt,
                    min_depth_events=min_depth_events,
                )
            )

    aggregate = summarize_book_cost_partitions(partitions)
    status, status_reasons = fast59_status(aggregate)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "fast59_book_cost_audit_research_only",
            "wal_book_cost_sample_cannot_approve_strategy",
            "prospective_shadow_and_60_day_book_evidence_still_required",
        ],
        "experiment_stage": "fast59_book_cost_audit",
        "status": status,
        "status_reasons": status_reasons,
        "thresholds": {
            "max_roundtrip_book_cost_bps": max_roundtrip_book_cost_bps,
            "max_p95_receive_delay_ms": max_p95_receive_delay_ms,
            "min_p05_top_notional_usdt": min_p05_top_notional_usdt,
            "min_depth_events": min_depth_events,
        },
        "aggregate": aggregate,
        "partitions": partitions,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def discover_default_wal_roots(data_root: Path) -> tuple[WalRootRef, ...]:
    refs = []
    for path in sorted(data_root.glob("prospective-wal-fast*")):
        if path.is_dir() and any((path / "depth").glob("date=*/events.jsonl.gz")):
            refs.append(WalRootRef(label=path.name, path=path))
    return tuple(refs)


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


def depth_partition_paths(root: Path, *, dates: tuple[date, ...] | None) -> tuple[Path, ...]:
    if dates is None:
        return tuple(sorted((root / "depth").glob("date=*/events.jsonl.gz")))
    paths = []
    for utc_date in dates:
        path = root / "depth" / f"date={utc_date.isoformat()}" / "events.jsonl.gz"
        if path.exists():
            paths.append(path)
    return tuple(paths)


def partition_date(path: Path) -> date:
    for parent in path.parents:
        if parent.name.startswith("date="):
            return date.fromisoformat(parent.name.removeprefix("date="))
    raise ValueError(f"cannot infer date partition from {path}")


def audit_depth_partition(
    *,
    label: str,
    wal_root: Path,
    depth_path: Path,
    utc_date: date,
    max_roundtrip_book_cost_bps: float,
    max_p95_receive_delay_ms: float,
    min_p05_top_notional_usdt: float,
    min_depth_events: int,
) -> dict[str, Any]:
    spreads: list[float] = []
    receive_delays: list[float] = []
    bid_notional: list[float] = []
    ask_notional: list[float] = []
    event_times: list[datetime] = []
    crossed_or_invalid = 0
    rows = 0
    with gzip.open(depth_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            payload = json.loads(line)
            bid_price = _positive_float(payload.get("bid_price"))
            ask_price = _positive_float(payload.get("ask_price"))
            bid_qty = _nonnegative_float(payload.get("bid_quantity"))
            ask_qty = _nonnegative_float(payload.get("ask_quantity"))
            if bid_price is None or ask_price is None or ask_price < bid_price:
                crossed_or_invalid += 1
                continue
            spreads.append(math.log(ask_price / bid_price) * 10_000)
            if bid_qty is not None:
                bid_notional.append(bid_price * bid_qty)
            if ask_qty is not None:
                ask_notional.append(ask_price * ask_qty)
            event_time = parse_datetime(str(payload["event_time"]))
            event_times.append(event_time)
            receive_time_raw = payload.get("receive_time")
            if receive_time_raw is not None:
                receive_time = parse_datetime(str(receive_time_raw))
                receive_delays.append(max(0.0, (receive_time - event_time).total_seconds() * 1000))

    spread_summary = numeric_summary(spreads)
    receive_delay_summary = numeric_summary(receive_delays)
    bid_notional_summary = numeric_summary(bid_notional)
    ask_notional_summary = numeric_summary(ask_notional)
    p95_spread = float(spread_summary["p95"])
    p95_receive_delay = float(receive_delay_summary["p95"])
    p05_bid_notional = float(bid_notional_summary["p05"])
    p05_ask_notional = float(ask_notional_summary["p05"])
    first_event = min(event_times) if event_times else None
    last_event = max(event_times) if event_times else None
    reasons = []
    if rows < min_depth_events:
        reasons.append(f"depth_events_below_minimum:{rows}<{min_depth_events}")
    if crossed_or_invalid:
        reasons.append(f"invalid_or_crossed_quotes:{crossed_or_invalid}")
    if p95_spread > max_roundtrip_book_cost_bps:
        reasons.append(
            f"p95_roundtrip_book_cost_above_threshold:{p95_spread:.6f}>{max_roundtrip_book_cost_bps}"
        )
    if p95_receive_delay > max_p95_receive_delay_ms:
        reasons.append(
            f"p95_receive_delay_above_threshold:{p95_receive_delay:.3f}>{max_p95_receive_delay_ms}"
        )
    if min(p05_bid_notional, p05_ask_notional) < min_p05_top_notional_usdt:
        reasons.append(
            "p05_top_notional_below_threshold:"
            f"{min(p05_bid_notional, p05_ask_notional):.2f}<{min_p05_top_notional_usdt}"
        )
    return {
        "label": label,
        "wal_root": str(wal_root),
        "utc_date": utc_date.isoformat(),
        "relative_path": str(depth_path.relative_to(wal_root)),
        "eligible_sample": not reasons,
        "reasons": reasons,
        "row_count": rows,
        "valid_quote_count": len(spreads),
        "invalid_or_crossed_quote_count": crossed_or_invalid,
        "first_event_time": first_event.isoformat() if first_event else None,
        "last_event_time": last_event.isoformat() if last_event else None,
        "duration_seconds": (last_event - first_event).total_seconds()
        if first_event and last_event
        else 0.0,
        "roundtrip_book_cost_bps": spread_summary,
        "receive_delay_ms": receive_delay_summary,
        "top_bid_notional_usdt": bid_notional_summary,
        "top_ask_notional_usdt": ask_notional_summary,
    }


def summarize_book_cost_partitions(partitions: list[dict[str, Any]]) -> dict[str, Any]:
    if not partitions:
        return {
            "partition_count": 0,
            "eligible_partition_count": 0,
            "unique_utc_dates": [],
            "max_p95_roundtrip_book_cost_bps": None,
            "max_p95_receive_delay_ms": None,
            "min_p05_top_notional_usdt": None,
        }
    p95_costs = [
        float(partition["roundtrip_book_cost_bps"]["p95"]) for partition in partitions
    ]
    p95_delays = [float(partition["receive_delay_ms"]["p95"]) for partition in partitions]
    p05_notional = [
        min(
            float(partition["top_bid_notional_usdt"]["p05"]),
            float(partition["top_ask_notional_usdt"]["p05"]),
        )
        for partition in partitions
    ]
    return {
        "partition_count": len(partitions),
        "eligible_partition_count": sum(bool(partition["eligible_sample"]) for partition in partitions),
        "unique_utc_dates": sorted({str(partition["utc_date"]) for partition in partitions}),
        "max_p95_roundtrip_book_cost_bps": max(p95_costs),
        "max_p95_receive_delay_ms": max(p95_delays),
        "min_p05_top_notional_usdt": min(p05_notional),
    }


def fast59_status(aggregate: dict[str, Any]) -> tuple[str, list[str]]:
    partition_count = int(aggregate["partition_count"])
    eligible_count = int(aggregate["eligible_partition_count"])
    if partition_count == 0:
        return "FAST59_NO_BOOK_WAL_PARTITIONS", ["book_wal_partitions_missing"]
    if eligible_count == partition_count:
        return (
            "FAST59_BOOK_COST_SAMPLE_OK_RESEARCH_ONLY",
            ["not_approved_strategy", "full_60_day_book_evidence_still_required"],
        )
    if eligible_count > 0:
        return (
            "FAST59_BOOK_COST_SAMPLE_PARTIAL_RESEARCH_ONLY",
            [
                f"eligible_partitions:{eligible_count}/{partition_count}",
                "do_not_promote_to_strategy",
            ],
        )
    return (
        "FAST59_BOOK_COST_SAMPLE_FAILED",
        [
            f"eligible_partitions:{eligible_count}/{partition_count}",
            "do_not_promote_to_strategy",
        ],
    )


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p05": quantile(ordered, 0.05),
        "p50": quantile(ordered, 0.50),
        "p90": quantile(ordered, 0.90),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def quantile(ordered_values: list[float], fraction: float) -> float:
    if not ordered_values:
        return 0.0
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered_values[int(position)]
    weight = position - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def parse_datetime(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or not math.isfinite(parsed):
        return None
    return parsed


def _nonnegative_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if parsed < 0 or not math.isfinite(parsed):
        return None
    return parsed


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "thresholds": report["thresholds"],
        "aggregate": report["aggregate"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())

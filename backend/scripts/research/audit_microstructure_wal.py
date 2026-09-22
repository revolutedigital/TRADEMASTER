"""Audit prospective recorder WAL partitions for the 60-day evidence gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.market.microstructure_wal_audit import BookEvidenceGate, DailyWalAudit
from app.services.market.microstructure_wal_audit import (
    ProspectiveWalAuditor,
    count_complete_days,
    evaluate_book_evidence_gate,
)


DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1" / "prospective-wal"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--date", type=_parse_date)
    parser.add_argument("--start-date", type=_parse_date)
    parser.add_argument("--end-date", type=_parse_date)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    parser.add_argument("--required-complete-days", type=int, default=60)
    arguments = parser.parse_args()

    start_date, end_date = _resolve_range(arguments.date, arguments.start_date, arguments.end_date)
    auditor = ProspectiveWalAuditor(arguments.root)
    audits = auditor.audit_range(start_date, end_date)
    gate = evaluate_book_evidence_gate(
        audits,
        required_complete_days=arguments.required_complete_days,
    )
    if arguments.format == "json":
        _write_stdout(
            json.dumps(
                {
                    "book_evidence_gate": gate.to_dict(),
                    "daily_audits": [audit.to_dict() for audit in audits],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_text(audits, gate=gate)
    if arguments.fail_on_incomplete and not gate.eligible:
        return 2
    return 0


def _resolve_range(
    single_date: date | None,
    start_date: date | None,
    end_date: date | None,
) -> tuple[date, date]:
    if single_date and (start_date or end_date):
        raise SystemExit("--date cannot be combined with --start-date or --end-date")
    if single_date:
        return single_date, single_date
    today = datetime.now(UTC).date()
    resolved_end = end_date or (today - timedelta(days=1))
    resolved_start = start_date or resolved_end
    if resolved_end < resolved_start:
        raise SystemExit("--end-date must be on or after --start-date")
    return resolved_start, resolved_end


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def _print_text(audits: Sequence[DailyWalAudit], *, gate: BookEvidenceGate) -> None:
    complete_days = count_complete_days(audits)
    _write_stdout(
        "Microstructure WAL audit: "
        f"{complete_days}/{len(audits)} complete UTC days; "
        "research_only=true order_submission_allowed=false execution_authorization=none"
    )
    _write_stdout(
        "Book evidence gate: "
        f"eligible={str(gate.eligible).lower()} "
        f"longest_complete_streak_days={gate.longest_complete_streak_days} "
        f"required_complete_days={gate.required_complete_days} "
        f"manifest={gate.manifest_sha256}"
    )
    for reason in gate.reasons:
        _write_stdout(f"  - {reason}")
    for audit in audits:
        _write_stdout(f"{audit.utc_date.isoformat()} {audit.status} {audit.manifest_sha256}")
        if audit.reasons:
            for reason in audit.reasons:
                _write_stdout(f"  - {reason}")
        for stream in audit.streams:
            max_gap = (
                "n/a"
                if stream.max_receive_gap_seconds is None
                else f"{stream.max_receive_gap_seconds:.3f}s"
            )
            _write_stdout(
                "  "
                f"{stream.event_type.value}: rows={stream.row_count} "
                f"bytes={stream.byte_size} gaps={stream.sequence_gap_count} "
                f"dupes={stream.duplicate_sequence_count} max_receive_gap={max_gap}"
            )


def _write_stdout(line: str) -> None:
    sys.stdout.write(f"{line}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Completeness and integrity checks for prospective microstructure WAL files."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

from app.schemas.microstructure import MarketEventType


WalAuditStatus = Literal["VALID", "PARTIAL", "MISSING", "INVALID"]


DEFAULT_REQUIRED_EVENT_TYPES = (
    MarketEventType.TRADE,
    MarketEventType.DEPTH,
    MarketEventType.MARK_PRICE,
)
DEFAULT_OPTIONAL_EVENT_TYPES = (MarketEventType.LIQUIDATION,)
DEFAULT_MIN_ROWS_BY_TYPE = {
    MarketEventType.TRADE: 100_000,
    MarketEventType.DEPTH: 100_000,
    MarketEventType.MARK_PRICE: 60_000,
}
DEFAULT_MAX_RECEIVE_GAP_SECONDS = {
    MarketEventType.TRADE: 30.0,
    MarketEventType.DEPTH: 30.0,
    MarketEventType.MARK_PRICE: 10.0,
}
CONTIGUOUS_SEQUENCE_TYPES = {MarketEventType.AGG_TRADE, MarketEventType.DEPTH}


@dataclass(frozen=True)
class DailyCompletenessPolicy:
    """Rules a UTC day must pass before it can count toward prospective evidence."""

    required_event_types: tuple[MarketEventType, ...] = DEFAULT_REQUIRED_EVENT_TYPES
    optional_event_types: tuple[MarketEventType, ...] = DEFAULT_OPTIONAL_EVENT_TYPES
    min_rows_by_type: Mapping[MarketEventType, int] = field(
        default_factory=lambda: dict(DEFAULT_MIN_ROWS_BY_TYPE)
    )
    max_receive_gap_seconds_by_type: Mapping[MarketEventType, float] = field(
        default_factory=lambda: dict(DEFAULT_MAX_RECEIVE_GAP_SECONDS)
    )
    max_start_delay: timedelta = timedelta(minutes=5)
    max_end_lag: timedelta = timedelta(minutes=5)


@dataclass(frozen=True)
class EventStreamWalAudit:
    event_type: MarketEventType
    utc_date: date
    relative_path: str
    exists: bool
    byte_size: int
    gzip_sha256: str | None
    row_count: int
    first_event_time: datetime | None
    last_event_time: datetime | None
    first_receive_time: datetime | None
    last_receive_time: datetime | None
    max_receive_gap_seconds: float | None
    duplicate_sequence_count: int
    sequence_regression_count: int
    sequence_gap_count: int
    json_error_count: int
    wrong_event_type_count: int
    reasons: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "utc_date": self.utc_date.isoformat(),
            "relative_path": self.relative_path,
            "exists": self.exists,
            "byte_size": self.byte_size,
            "gzip_sha256": self.gzip_sha256,
            "row_count": self.row_count,
            "first_event_time": _datetime_json(self.first_event_time),
            "last_event_time": _datetime_json(self.last_event_time),
            "first_receive_time": _datetime_json(self.first_receive_time),
            "last_receive_time": _datetime_json(self.last_receive_time),
            "max_receive_gap_seconds": self.max_receive_gap_seconds,
            "duplicate_sequence_count": self.duplicate_sequence_count,
            "sequence_regression_count": self.sequence_regression_count,
            "sequence_gap_count": self.sequence_gap_count,
            "json_error_count": self.json_error_count,
            "wrong_event_type_count": self.wrong_event_type_count,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DailyWalAudit:
    utc_date: date
    status: WalAuditStatus
    complete_day: bool
    manifest_sha256: str
    streams: tuple[EventStreamWalAudit, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "utc_date": self.utc_date.isoformat(),
            "status": self.status,
            "complete_day": self.complete_day,
            "manifest_sha256": self.manifest_sha256,
            "streams": [stream.to_dict() for stream in self.streams],
            "reasons": list(self.reasons),
            "safety": {
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
            },
        }


class ProspectiveWalAuditor:
    """Audit recorder output without touching exchange accounts or trading paths."""

    def __init__(
        self,
        root: Path,
        *,
        policy: DailyCompletenessPolicy | None = None,
    ) -> None:
        self._root = root
        self._policy = policy or DailyCompletenessPolicy()

    def audit_range(self, start_date: date, end_date: date) -> tuple[DailyWalAudit, ...]:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")
        days = (end_date - start_date).days + 1
        return tuple(self.audit_date(start_date + timedelta(days=offset)) for offset in range(days))

    def audit_date(self, utc_date: date) -> DailyWalAudit:
        stream_types = self._policy.required_event_types + self._policy.optional_event_types
        streams = tuple(self._audit_stream(event_type, utc_date) for event_type in stream_types)
        reasons: list[str] = []
        required_by_type = {
            stream.event_type: stream
            for stream in streams
            if stream.event_type in self._policy.required_event_types
        }
        for event_type in self._policy.required_event_types:
            stream = required_by_type[event_type]
            for reason in stream.reasons:
                reasons.append(f"{event_type.value}: {reason}")

        required_missing = any(
            not required_by_type[event_type].exists for event_type in self._policy.required_event_types
        )
        required_empty = any(
            required_by_type[event_type].exists and required_by_type[event_type].row_count == 0
            for event_type in self._policy.required_event_types
        )
        integrity_failure = any(_has_integrity_failure(stream) for stream in required_by_type.values())
        coverage_failure = any(_has_coverage_failure(stream) for stream in required_by_type.values())

        if not reasons:
            status: WalAuditStatus = "VALID"
        elif required_missing or required_empty:
            status = "MISSING"
        elif integrity_failure:
            status = "INVALID"
        elif coverage_failure:
            status = "PARTIAL"
        else:
            status = "INVALID"

        manifest_sha256 = _stable_sha256(
            {
                "utc_date": utc_date.isoformat(),
                "status": status,
                "streams": [stream.to_dict() for stream in streams],
            }
        )
        return DailyWalAudit(
            utc_date=utc_date,
            status=status,
            complete_day=status == "VALID",
            manifest_sha256=manifest_sha256,
            streams=streams,
            reasons=tuple(reasons),
        )

    def _audit_stream(self, event_type: MarketEventType, utc_date: date) -> EventStreamWalAudit:
        relative_path = _relative_event_path(event_type, utc_date)
        path = self._root / relative_path
        if not path.exists():
            return EventStreamWalAudit(
                event_type=event_type,
                utc_date=utc_date,
                relative_path=relative_path,
                exists=False,
                byte_size=0,
                gzip_sha256=None,
                row_count=0,
                first_event_time=None,
                last_event_time=None,
                first_receive_time=None,
                last_receive_time=None,
                max_receive_gap_seconds=None,
                duplicate_sequence_count=0,
                sequence_regression_count=0,
                sequence_gap_count=0,
                json_error_count=0,
                wrong_event_type_count=0,
                reasons=("missing WAL file",),
            )

        state = _StreamAuditState(event_type=event_type)
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    state.observe(line)
        except (EOFError, OSError, gzip.BadGzipFile, UnicodeDecodeError) as error:
            state.reasons.append(f"unreadable gzip WAL: {error}")

        byte_size = path.stat().st_size
        reasons = state.reasons
        self._append_policy_reasons(event_type, utc_date, state, reasons)
        return EventStreamWalAudit(
            event_type=event_type,
            utc_date=utc_date,
            relative_path=relative_path,
            exists=True,
            byte_size=byte_size,
            gzip_sha256=_sha256_file(path),
            row_count=state.row_count,
            first_event_time=state.first_event_time,
            last_event_time=state.last_event_time,
            first_receive_time=state.first_receive_time,
            last_receive_time=state.last_receive_time,
            max_receive_gap_seconds=state.max_receive_gap_seconds,
            duplicate_sequence_count=state.duplicate_sequence_count,
            sequence_regression_count=state.sequence_regression_count,
            sequence_gap_count=state.sequence_gap_count,
            json_error_count=state.json_error_count,
            wrong_event_type_count=state.wrong_event_type_count,
            reasons=tuple(reasons),
        )

    def _append_policy_reasons(
        self,
        event_type: MarketEventType,
        utc_date: date,
        state: "_StreamAuditState",
        reasons: list[str],
    ) -> None:
        if event_type in self._policy.optional_event_types and state.row_count == 0:
            return
        if state.row_count == 0:
            reasons.append("empty WAL file")
            return

        minimum_rows = self._policy.min_rows_by_type.get(event_type, 0)
        if state.row_count < minimum_rows:
            reasons.append(f"row_count {state.row_count} below minimum {minimum_rows}")

        day_start = datetime.combine(utc_date, time.min, tzinfo=UTC)
        day_end = day_start + timedelta(days=1)
        first_observed = state.first_receive_time or state.first_event_time
        last_observed = state.last_receive_time or state.last_event_time
        if first_observed and first_observed > day_start + self._policy.max_start_delay:
            reasons.append(
                "coverage starts at "
                f"{first_observed.isoformat()} after allowed boundary "
                f"{(day_start + self._policy.max_start_delay).isoformat()}"
            )
        if last_observed and last_observed < day_end - self._policy.max_end_lag:
            reasons.append(
                "coverage ends at "
                f"{last_observed.isoformat()} before allowed boundary "
                f"{(day_end - self._policy.max_end_lag).isoformat()}"
            )

        max_allowed_gap = self._policy.max_receive_gap_seconds_by_type.get(event_type)
        if (
            max_allowed_gap is not None
            and state.max_receive_gap_seconds is not None
            and state.max_receive_gap_seconds > max_allowed_gap
        ):
            reasons.append(
                f"max receive gap {state.max_receive_gap_seconds:.3f}s exceeds "
                f"{max_allowed_gap:.3f}s"
            )

        if state.json_error_count:
            reasons.append(f"{state.json_error_count} JSON parse errors")
        if state.wrong_event_type_count:
            reasons.append(f"{state.wrong_event_type_count} rows with another event_type")
        if state.duplicate_sequence_count:
            reasons.append(f"{state.duplicate_sequence_count} duplicate sequence IDs")
        if state.sequence_regression_count:
            reasons.append(f"{state.sequence_regression_count} sequence regressions")
        if state.sequence_gap_count:
            reasons.append(f"{state.sequence_gap_count} sequence gaps")


@dataclass
class _StreamAuditState:
    event_type: MarketEventType
    row_count: int = 0
    first_event_time: datetime | None = None
    last_event_time: datetime | None = None
    first_receive_time: datetime | None = None
    last_receive_time: datetime | None = None
    max_receive_gap_seconds: float | None = None
    duplicate_sequence_count: int = 0
    sequence_regression_count: int = 0
    sequence_gap_count: int = 0
    json_error_count: int = 0
    wrong_event_type_count: int = 0
    reasons: list[str] = field(default_factory=list)
    _previous_sequence_end: int | None = None
    _previous_receive_time: datetime | None = None

    def observe(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            self.json_error_count += 1
            return
        if not isinstance(row, dict):
            self.json_error_count += 1
            return
        if row.get("event_type") != self.event_type.value:
            self.wrong_event_type_count += 1
            return

        event_time = _parse_datetime(row.get("event_time"))
        receive_time = _parse_datetime(row.get("receive_time"))
        self.row_count += 1
        self.first_event_time = self.first_event_time or event_time
        self.last_event_time = event_time or self.last_event_time
        self.first_receive_time = self.first_receive_time or receive_time
        self.last_receive_time = receive_time or self.last_receive_time
        if receive_time is not None and self._previous_receive_time is not None:
            receive_gap = (receive_time - self._previous_receive_time).total_seconds()
            if receive_gap < 0:
                self.sequence_regression_count += 1
            else:
                current_max = self.max_receive_gap_seconds or 0.0
                self.max_receive_gap_seconds = max(current_max, receive_gap)
        if receive_time is not None:
            self._previous_receive_time = receive_time

        sequence_start = _optional_int(row.get("first_sequence_id") or row.get("sequence_id"))
        sequence_end = _optional_int(row.get("last_sequence_id") or row.get("sequence_id"))
        if sequence_end is None:
            return

        if self._previous_sequence_end is not None:
            if sequence_end == self._previous_sequence_end:
                self.duplicate_sequence_count += 1
            elif sequence_end < self._previous_sequence_end:
                self.sequence_regression_count += 1
            elif self.event_type in CONTIGUOUS_SEQUENCE_TYPES:
                if self.event_type == MarketEventType.DEPTH:
                    previous_final_update_id = _optional_int(
                        (row.get("payload") or {}).get("previous_final_update_id")
                    )
                    if previous_final_update_id != self._previous_sequence_end:
                        self.sequence_gap_count += 1
                elif sequence_start is not None and sequence_start != self._previous_sequence_end + 1:
                    self.sequence_gap_count += 1
        self._previous_sequence_end = max(sequence_end, self._previous_sequence_end or sequence_end)


def _has_integrity_failure(stream: EventStreamWalAudit) -> bool:
    return any(
        (
            stream.json_error_count,
            stream.wrong_event_type_count,
            stream.duplicate_sequence_count,
            stream.sequence_regression_count,
            stream.sequence_gap_count,
            any("unreadable gzip WAL" in reason for reason in stream.reasons),
        )
    )


def _has_coverage_failure(stream: EventStreamWalAudit) -> bool:
    return any(
        reason.startswith("coverage starts")
        or reason.startswith("coverage ends")
        or reason.startswith("row_count")
        or reason.startswith("max receive gap")
        for reason in stream.reasons
    )


def _relative_event_path(event_type: MarketEventType, utc_date: date) -> str:
    return f"{event_type.value.lower()}/date={utc_date.isoformat()}/events.jsonl.gz"


def _datetime_json(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_complete_days(audits: Iterable[DailyWalAudit]) -> int:
    return sum(1 for audit in audits if audit.complete_day)

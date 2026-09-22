"""Prospective WAL audit proves daily completeness before evidence is counted."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from app.services.market.microstructure_wal_audit import (
    DailyCompletenessPolicy,
    ProspectiveWalAuditor,
    count_complete_days,
)


def test_audit_accepts_clean_complete_utc_day(tmp_path: Path) -> None:
    utc_date = date(2026, 1, 1)
    policy = DailyCompletenessPolicy(
        min_rows_by_type={
            MarketEventType.TRADE: 2,
            MarketEventType.DEPTH: 2,
            MarketEventType.MARK_PRICE: 2,
        },
        max_receive_gap_seconds_by_type={
            MarketEventType.TRADE: 90_000,
            MarketEventType.DEPTH: 90_000,
            MarketEventType.MARK_PRICE: 90_000,
        },
    )
    start = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 23, 59, tzinfo=UTC)
    _write_events(
        tmp_path,
        [
            _event(MarketEventType.TRADE, start, sequence_id=10),
            _event(MarketEventType.TRADE, end, sequence_id=12),
            _event(
                MarketEventType.DEPTH,
                start,
                first_sequence_id=100,
                last_sequence_id=100,
                payload={"previous_final_update_id": 99, "bids": [["99", "2"]], "asks": []},
            ),
            _event(
                MarketEventType.DEPTH,
                end,
                first_sequence_id=101,
                last_sequence_id=101,
                payload={"previous_final_update_id": 100, "bids": [], "asks": [["101", "1"]]},
            ),
            _event(MarketEventType.MARK_PRICE, start, price=100),
            _event(MarketEventType.MARK_PRICE, end, price=101),
        ],
    )

    audit = ProspectiveWalAuditor(tmp_path, policy=policy).audit_date(utc_date)

    assert audit.status == "VALID"
    assert audit.complete_day is True
    assert count_complete_days((audit,)) == 1
    assert len(audit.manifest_sha256) == 64


def test_audit_rejects_missing_required_stream(tmp_path: Path) -> None:
    utc_date = date(2026, 1, 1)
    _write_events(
        tmp_path,
        [
            _event(MarketEventType.TRADE, datetime(2026, 1, 1, 0, 1, tzinfo=UTC)),
        ],
    )

    audit = ProspectiveWalAuditor(
        tmp_path,
        policy=DailyCompletenessPolicy(
            min_rows_by_type={MarketEventType.TRADE: 1},
            max_receive_gap_seconds_by_type={},
        ),
    ).audit_date(utc_date)

    assert audit.status == "MISSING"
    assert audit.complete_day is False
    assert any("DEPTH: missing WAL file" in reason for reason in audit.reasons)


def test_audit_detects_depth_sequence_gap(tmp_path: Path) -> None:
    utc_date = date(2026, 1, 1)
    start = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    _write_events(
        tmp_path,
        [
            _event(MarketEventType.TRADE, start, sequence_id=1),
            _event(
                MarketEventType.DEPTH,
                start,
                first_sequence_id=100,
                last_sequence_id=100,
                payload={"previous_final_update_id": 99},
            ),
            _event(
                MarketEventType.DEPTH,
                start + timedelta(seconds=1),
                first_sequence_id=102,
                last_sequence_id=102,
                payload={"previous_final_update_id": 101},
            ),
            _event(MarketEventType.MARK_PRICE, start),
        ],
    )

    audit = ProspectiveWalAuditor(
        tmp_path,
        policy=DailyCompletenessPolicy(
            min_rows_by_type={
                MarketEventType.TRADE: 1,
                MarketEventType.DEPTH: 1,
                MarketEventType.MARK_PRICE: 1,
            },
            max_receive_gap_seconds_by_type={},
            max_end_lag=timedelta(days=1),
        ),
    ).audit_date(utc_date)

    assert audit.status == "INVALID"
    assert any("DEPTH: 1 sequence gaps" in reason for reason in audit.reasons)


def test_audit_marks_boundary_shortfall_as_partial(tmp_path: Path) -> None:
    utc_date = date(2026, 1, 1)
    middle = datetime(2026, 1, 1, 12, tzinfo=UTC)
    _write_events(
        tmp_path,
        [
            _event(MarketEventType.TRADE, middle),
            _event(MarketEventType.DEPTH, middle),
            _event(MarketEventType.MARK_PRICE, middle),
        ],
    )

    audit = ProspectiveWalAuditor(
        tmp_path,
        policy=DailyCompletenessPolicy(
            min_rows_by_type={
                MarketEventType.TRADE: 1,
                MarketEventType.DEPTH: 1,
                MarketEventType.MARK_PRICE: 1,
            },
            max_receive_gap_seconds_by_type={},
        ),
    ).audit_date(utc_date)

    assert audit.status == "PARTIAL"
    assert audit.complete_day is False
    assert any(reason.startswith("TRADE: coverage starts") for reason in audit.reasons)


def _event(
    event_type: MarketEventType,
    when: datetime,
    *,
    sequence_id: int | None = None,
    first_sequence_id: int | None = None,
    last_sequence_id: int | None = None,
    price: float = 100.0,
    payload: dict | None = None,
) -> MicrostructureEvent:
    return MicrostructureEvent(
        product="usdm_perpetual",
        symbol="BTCUSDT",
        event_type=event_type,
        event_time=when,
        receive_time=when,
        sequence_id=sequence_id,
        first_sequence_id=first_sequence_id,
        last_sequence_id=last_sequence_id,
        price=price,
        quantity=1.0 if event_type == MarketEventType.TRADE else None,
        bid_price=99.0 if event_type == MarketEventType.DEPTH else None,
        ask_price=101.0 if event_type == MarketEventType.DEPTH else None,
        payload=payload,
    )


def _write_events(root: Path, events: list[MicrostructureEvent]) -> None:
    for event in events:
        event_date = event.event_time.astimezone(UTC).date().isoformat()
        path = root / event.event_type.value.lower() / f"date={event_date}" / "events.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at", encoding="utf-8") as output:
            output.write(json.dumps(event.model_dump(mode="json"), separators=(",", ":")))
            output.write("\n")

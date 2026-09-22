"""Recorder entrypoint keeps WAL collection and evidence artifacts fail-closed."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from scripts.research import record_microstructure


def test_write_audit_status_emits_small_research_only_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls(2026, 1, 2, tzinfo=tz or UTC)

    monkeypatch.setattr(record_microstructure, "datetime", FixedDatetime)
    wal_root = tmp_path / "wal"
    status_path = tmp_path / "audits" / "evidence-gate-status.json"
    event_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    _write_events(
        wal_root,
        [
            _event(MarketEventType.TRADE, event_time),
            _event(MarketEventType.TRADE, event_time, product="spot"),
            _event(MarketEventType.DEPTH, event_time),
            _event(MarketEventType.MARK_PRICE, event_time),
        ],
    )

    record_microstructure._write_audit_status(
        root=wal_root,
        status_path=status_path,
        rolling_days=1,
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["artifact_available"] is True
    assert payload["audited_start_date"] == "2026-01-01"
    assert payload["audited_end_date"] == "2026-01-01"
    assert payload["book_evidence_gate"]["required_streams"] == [
        "TRADE",
        "DEPTH",
        "MARK_PRICE",
        "SPOT_TRADE",
    ]
    assert payload["book_evidence_gate"]["eligible"] is False
    assert payload["safety"] == {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }
    assert "daily_audits" not in payload


def test_status_artifact_reader_fails_closed_when_artifact_is_missing(tmp_path: Path) -> None:
    payload = record_microstructure._read_status_artifact(tmp_path / "missing.json")

    assert payload["artifact_available"] is False
    assert payload["book_evidence_gate"]["eligible"] is False
    assert payload["status_reasons"] == ["evidence_gate_artifact_missing"]
    assert payload["safety"] == {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def test_status_artifact_reader_fails_closed_when_artifact_is_invalid(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text("{not-json", encoding="utf-8")

    payload = record_microstructure._read_status_artifact(artifact)

    assert payload["artifact_available"] is False
    assert payload["book_evidence_gate"]["eligible"] is False
    assert payload["status_reasons"] == [
        "evidence_gate_artifact_unreadable:JSONDecodeError"
    ]


def test_health_payload_never_grants_execution(tmp_path: Path) -> None:
    payload = record_microstructure._health_payload(tmp_path / "status.json")

    assert payload["status"] == "healthy"
    assert payload["artifact_exists"] is False
    assert payload["research_only"] is True
    assert payload["order_submission_allowed"] is False
    assert payload["execution_authorization"] == "none"


def _event(
    event_type: MarketEventType,
    when: datetime,
    *,
    product: str = "usdm_perpetual",
) -> MicrostructureEvent:
    return MicrostructureEvent(
        product=product,
        symbol="BTCUSDT",
        event_type=event_type,
        event_time=when,
        receive_time=when,
        sequence_id=1,
        first_sequence_id=1 if event_type == MarketEventType.DEPTH else None,
        last_sequence_id=1 if event_type == MarketEventType.DEPTH else None,
        price=100 if event_type in {MarketEventType.TRADE, MarketEventType.MARK_PRICE} else None,
        quantity=1 if event_type == MarketEventType.TRADE else None,
        bid_price=99 if event_type == MarketEventType.DEPTH else None,
        ask_price=101 if event_type == MarketEventType.DEPTH else None,
        payload={"previous_final_update_id": 0} if event_type == MarketEventType.DEPTH else None,
    )


def _write_events(root: Path, events: list[MicrostructureEvent]) -> None:
    for event in events:
        event_date = event.event_time.astimezone(UTC).date().isoformat()
        event_directory = event.event_type.value.lower()
        if event.product != "usdm_perpetual":
            event_directory = f"{event.product}_{event_directory}"
        path = root / event_directory / f"date={event_date}" / "events.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at", encoding="utf-8") as output:
            output.write(json.dumps(event.model_dump(mode="json"), separators=(",", ":")))
            output.write("\n")

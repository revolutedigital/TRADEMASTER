"""Research-only evidence-gate API never grants execution access."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from app.api.v1 import research
from app.services.market.microstructure_wal_audit import build_evidence_gate_status


async def test_evidence_gate_api_returns_ineligible_placeholder_when_artifact_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_artifact = tmp_path / "missing.json"
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(missing_artifact),
    )

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is False
    assert response.book_evidence_gate.eligible is False
    assert response.status_reasons == ["evidence_gate_artifact_missing"]
    assert response.safety.order_submission_allowed is False
    assert response.safety.execution_authorization == "none"


async def test_evidence_gate_api_reads_valid_small_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    payload = build_evidence_gate_status(
        (),
        artifact_available=True,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_reasons=("no_complete_days_yet",),
    )
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is True
    assert response.audited_start_date is None
    assert response.audited_end_date is None
    assert response.book_evidence_gate.required_complete_days == 60
    assert response.book_evidence_gate.streak_start is None
    assert response.book_evidence_gate.streak_end is None
    assert response.status_reasons == ["no_complete_days_yet"]


async def test_evidence_gate_api_rejects_unreadable_artifact_as_ineligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "evidence-gate-status.json"
    artifact.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        research.settings,
        "microstructure_evidence_status_path",
        str(artifact),
    )

    response = await research.get_evidence_gate_status(_user={"sub": "operator"})

    assert response.artifact_available is False
    assert response.book_evidence_gate.eligible is False
    assert response.status_reasons == ["evidence_gate_artifact_unreadable:JSONDecodeError"]


def test_evidence_gate_status_accepts_iso_dates_in_artifact() -> None:
    payload = build_evidence_gate_status(
        (),
        artifact_available=False,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    payload["audited_start_date"] = "2026-01-01"
    payload["audited_end_date"] = "2026-01-02"
    payload["book_evidence_gate"]["streak_start"] = "2026-01-01"
    payload["book_evidence_gate"]["streak_end"] = "2026-01-02"
    payload["book_evidence_gate"]["incomplete_days"] = ["2026-01-01"]

    parsed = research.EvidenceGateStatusResponse.model_validate(payload)

    assert parsed.audited_start_date == date(2026, 1, 1)
    assert parsed.audited_end_date == date(2026, 1, 2)
    assert parsed.book_evidence_gate.incomplete_days == [date(2026, 1, 1)]

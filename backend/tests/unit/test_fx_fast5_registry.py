"""Round-5 protocol guard tests."""

import json
from pathlib import Path

import pytest

from scripts.research import fx_fast5_registry as registry


def created(digest: str) -> dict[str, object]:
    return {"date": "2026-09-21", "event": "registry_created", "preregistration_sha256": digest}


def event(kind: str, **extra: object) -> dict[str, object]:
    return {"date": "2026-09-21", "event": kind, "code_commit": "abc123", **extra}


def test_blind_2022_refuses_to_open_before_approved_development() -> None:
    digest = "a" * 64
    history = [
        created(digest),
        event("factory_validated"),
        event("labels_materialized"),
        event("policies_frozen", artifact_sha256="b" * 64),
        event("development_validation_run"),
        event("development_report", approved=False),
        event("blind_2022_run"),
    ]
    assert any(
        "2022 requires approved" in problem for problem in registry.problems(history, digest)
    )


def test_complete_development_sequence_is_valid() -> None:
    digest = "a" * 64
    history = [
        created(digest),
        event("factory_validated"),
        event("labels_materialized"),
        event("policies_frozen", artifact_sha256="b" * 64),
        event("development_validation_run"),
        event("development_report", approved=True),
        event("blind_2022_run"),
        event("blind_2022_report", approved=False),
    ]
    assert registry.problems(history, digest) == []


def test_append_rejects_invalid_transition_without_writing(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.md"
    preregistration.write_text("fixed", encoding="utf-8")
    digest = registry.sha256_of(preregistration)
    path = tmp_path / "registry.jsonl"
    path.write_text(json.dumps(created(digest)) + "\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="labels require"):
        registry.append_event(
            {"event": "labels_materialized", "code_commit": "abc"},
            registry=path,
            preregistration=preregistration,
        )
    assert path.read_bytes() == before

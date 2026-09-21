"""Round-6 ranked-entry registry boundaries."""

from pathlib import Path

from scripts.research.fx_fast5_registry import append_event, problems, read_events, sha256_of


def test_round6_cannot_freeze_before_labels(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    preregistration.write_text("ranked runner", encoding="utf-8")
    append_event(
        {"event": "registry_created", "preregistration_sha256": sha256_of(preregistration)},
        registry=registry,
        preregistration=preregistration,
    )
    events = [
        *read_events(registry),
        {
            "date": "2026-09-21",
            "event": "policies_frozen",
            "code_commit": "abc",
            "artifact_sha256": "0" * 64,
        },
    ]
    assert "policies require materialized labels" in "; ".join(
        problems(events, sha256_of(preregistration))
    )

"""Round-10 registry holds every protected sample closed."""

from pathlib import Path

from scripts.research.fx_fast5_registry import append_event, problems, read_events, sha256_of


def test_tournament_cannot_open_h2_without_a_frozen_policy(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    preregistration.write_text("tournament", encoding="utf-8")
    append_event(
        {"event": "registry_created", "preregistration_sha256": sha256_of(preregistration)},
        registry=registry,
        preregistration=preregistration,
    )
    events = [
        *read_events(registry),
        {"date": "2026-09-21", "event": "development_validation_run", "code_commit": "abc"},
    ]
    assert "2021-H2 requires frozen policies" in "; ".join(
        problems(events, sha256_of(preregistration))
    )

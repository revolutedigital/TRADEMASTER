"""Round-11 registry remains fail-closed."""

from pathlib import Path

from scripts.research.fx_fast5_registry import append_event, problems, read_events, sha256_of


def test_unknown_execution_event_is_rejected(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    preregistration.write_text("centralized flow", encoding="utf-8")
    append_event(
        {"event": "registry_created", "preregistration_sha256": sha256_of(preregistration)},
        registry=registry,
        preregistration=preregistration,
    )
    events = [*read_events(registry), {"date": "2026-09-21", "event": "live_run"}]
    assert "unknown event" in "; ".join(problems(events, sha256_of(preregistration)))

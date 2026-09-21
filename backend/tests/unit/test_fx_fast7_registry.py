"""Round-7 registry uses the established protected-sample gates."""

from pathlib import Path

from scripts.research.fx_fast5_registry import append_event, problems, read_events, sha256_of


def test_round7_rejects_h2_before_policy_freeze(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    preregistration.write_text("early protection", encoding="utf-8")
    append_event(
        {"event": "registry_created", "preregistration_sha256": sha256_of(preregistration)},
        registry=registry,
        preregistration=preregistration,
    )
    events = [
        *read_events(registry),
        {
            "date": "2026-09-21",
            "event": "development_validation_run",
            "code_commit": "abc",
        },
    ]
    assert "2021-H2 requires frozen policies" in "; ".join(
        problems(events, sha256_of(preregistration))
    )

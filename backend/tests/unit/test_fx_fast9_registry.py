"""Round-9 registry protects validation and holdouts."""

from pathlib import Path

from scripts.research.fx_fast5_registry import append_event, problems, read_events, sha256_of


def test_round9_rejects_h2_before_a_frozen_q2_policy(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    preregistration.write_text("managed payoff", encoding="utf-8")
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


def test_round9_rejects_2022_without_h2_approval(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    preregistration.write_text("managed payoff", encoding="utf-8")
    append_event(
        {"event": "registry_created", "preregistration_sha256": sha256_of(preregistration)},
        registry=registry,
        preregistration=preregistration,
    )
    events = [
        *read_events(registry),
        {"date": "2026-09-21", "event": "blind_2022_run", "code_commit": "abc"},
    ]
    assert "2022 requires approved 2021-H2" in "; ".join(
        problems(events, sha256_of(preregistration))
    )

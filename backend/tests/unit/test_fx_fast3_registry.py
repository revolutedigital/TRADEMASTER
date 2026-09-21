"""Protocol guards for the conditional short-trade research round."""

from pathlib import Path

import pytest

from scripts.research import fx_fast3_registry as registry

DIGEST = "a" * 64
DAY = "2026-09-21"


def event(kind: str, **fields: object) -> dict[str, object]:
    return {"event": kind, "date": DAY, **fields}


def created() -> dict[str, object]:
    return event("registry_created", preregistration_sha256=DIGEST)


def test_complete_history_is_valid() -> None:
    history = [
        created(),
        event("development_run", code_commit="abc"),
        event("model_locked", code_commit="abc", policy_id="M1.H60", artifact_sha256="b" * 64),
        event("development_holdout_run", code_commit="def"),
        event("development_report", code_commit="def", approved=True),
        event("confirmation_run", code_commit="ghi"),
        event("confirmation_report", code_commit="ghi", approved=True),
    ]

    assert registry.problems(history, DIGEST) == []


def test_holdout_requires_a_locked_model_and_runs_once() -> None:
    without_lock = [created(), event("development_holdout_run", code_commit="abc")]
    assert any("before model_locked" in problem for problem in registry.problems(without_lock, DIGEST))

    locked = event("model_locked", code_commit="abc", policy_id="B2.H15", artifact_sha256="b" * 64)
    twice = [created(), locked, event("development_holdout_run", code_commit="abc"),
             event("development_holdout_run", code_commit="abc")]
    assert any("only once" in problem for problem in registry.problems(twice, DIGEST))


def test_confirmation_requires_an_approved_development_report_and_runs_once() -> None:
    locked = event("model_locked", code_commit="abc", policy_id="B2.H15", artifact_sha256="b" * 64)
    d3 = event("development_holdout_run", code_commit="abc")
    rejected = [created(), locked, d3, event("development_report", code_commit="abc", approved=False),
                event("confirmation_run", code_commit="abc")]
    assert any("approved development report" in problem for problem in registry.problems(rejected, DIGEST))

    approved = [created(), locked, d3, event("development_report", code_commit="abc", approved=True),
                event("confirmation_run", code_commit="abc"), event("confirmation_run", code_commit="abc")]
    assert any("only once" in problem for problem in registry.problems(approved, DIGEST))


def test_document_drift_requires_an_amendment() -> None:
    assert registry.problems([created()], "c" * 64) == [
        "preregistration changed without an amendment carrying its new SHA-256"
    ]
    amended = [created(), event("amendment", reason="clarify", preregistration_sha256="c" * 64)]
    assert registry.problems(amended, "c" * 64) == []


def test_invalid_append_does_not_touch_registry(tmp_path: Path) -> None:
    document = tmp_path / "prereg.md"
    output = tmp_path / "registry.jsonl"
    document.write_text("contract", encoding="utf-8")
    registry.append_event(
        {"event": "registry_created", "preregistration_sha256": registry.sha256_of(document)},
        registry=output,
        preregistration=document,
    )

    with pytest.raises(ValueError, match="before model_locked"):
        registry.append_event(
            {"event": "development_holdout_run", "code_commit": "abc"},
            registry=output,
            preregistration=document,
        )

    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_committed_registry_matches_the_document() -> None:
    if not registry.DEFAULT_REGISTRY.exists():
        pytest.skip("registry is created in the preregistration commit")
    assert registry.problems(
        registry.read_events(), registry.sha256_of(registry.DEFAULT_PREREGISTRATION)
    ) == []

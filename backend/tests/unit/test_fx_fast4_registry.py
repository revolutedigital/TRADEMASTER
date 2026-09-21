"""Protocol guards for the quote-microstructure research round."""

from pathlib import Path

import pytest

from scripts.research import fx_fast4_registry as registry

DIGEST = "a" * 64
DAY = "2026-09-21"


def event(kind: str, **fields: object) -> dict[str, object]:
    return {"event": kind, "date": DAY, **fields}


def created() -> dict[str, object]:
    return event("registry_created", preregistration_sha256=DIGEST)


def validated() -> dict[str, object]:
    return event("factory_validated", code_commit="abc", artifact_sha256="b" * 64)


def locked() -> dict[str, object]:
    return event("model_locked", code_commit="abc", policy_id="M1.H120", artifact_sha256="c" * 64)


def test_complete_history_is_valid() -> None:
    history = [
        created(),
        validated(),
        event("development_run", code_commit="abc"),
        locked(),
        event("development_holdout_run", code_commit="def"),
        event("development_report", code_commit="def", approved=True),
        event("confirmation_run", code_commit="ghi"),
        event("confirmation_report", code_commit="ghi", approved=True),
        event("broker_portability_run", code_commit="jkl"),
        event("broker_portability_report", code_commit="jkl", approved=True),
    ]
    assert registry.problems(history, DIGEST) == []


def test_development_requires_validated_factory_and_model_requires_run() -> None:
    no_factory = [created(), event("development_run", code_commit="abc")]
    assert any("before factory_validated" in problem for problem in registry.problems(no_factory, DIGEST))

    no_run = [created(), validated(), locked()]
    assert any("requires a development run" in problem for problem in registry.problems(no_run, DIGEST))


def test_broker_week_requires_approved_s3_and_runs_once() -> None:
    premature = [created(), event("broker_portability_run", code_commit="abc")]
    assert any("approved S3" in problem for problem in registry.problems(premature, DIGEST))

    prefix = [
        created(), validated(), event("development_run", code_commit="abc"), locked(),
        event("development_holdout_run", code_commit="def"),
        event("development_report", code_commit="def", approved=True),
        event("confirmation_run", code_commit="ghi"),
        event("confirmation_report", code_commit="ghi", approved=True),
    ]
    twice = prefix + [
        event("broker_portability_run", code_commit="jkl"),
        event("broker_portability_run", code_commit="jkl"),
    ]
    assert any("only once" in problem for problem in registry.problems(twice, DIGEST))


def test_document_drift_requires_an_amendment() -> None:
    assert registry.problems([created()], "d" * 64) == [
        "preregistration changed without an amendment carrying its new SHA-256"
    ]
    amended = [created(), event("amendment", reason="clarify", preregistration_sha256="d" * 64)]
    assert registry.problems(amended, "d" * 64) == []


def test_invalid_append_does_not_touch_registry(tmp_path: Path) -> None:
    document = tmp_path / "prereg.md"
    output = tmp_path / "registry.jsonl"
    document.write_text("contract", encoding="utf-8")
    registry.append_event(
        {"event": "registry_created", "preregistration_sha256": registry.sha256_of(document)},
        registry=output,
        preregistration=document,
    )
    with pytest.raises(ValueError, match="approved S3"):
        registry.append_event(
            {"event": "broker_portability_run", "code_commit": "abc"},
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


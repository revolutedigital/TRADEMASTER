"""The fast-strategy registry enforces its pre-registration."""

import json
import re
from pathlib import Path

import pytest

from scripts.research import fx_fast_registry as registry

HASH = "a" * 64
OTHER_HASH = "b" * 64


def created(digest: str = HASH) -> dict:
    return {"event": "registry_created", "date": "2026-09-20", "preregistration_sha256": digest}


def calibration() -> dict:
    return {"event": "calibration_report", "date": "2026-09-21", "false_pass_rate": 0.04}


def discovery_report(approved: list[str]) -> dict:
    return {"event": "discovery_report", "date": "2026-09-22", "approved": approved}


def run(configuration: str, sample: str) -> dict:
    return {
        "event": "run",
        "date": "2026-09-22",
        "configuration": configuration,
        "sample": sample,
        "code_commit": "abc1234",
        "result": {"trades": 812, "mean_r": 0.03},
    }


def test_the_committed_registry_is_valid_for_the_committed_preregistration() -> None:
    events = registry.read_events()

    assert registry.problems(events, registry.sha256_of(registry.DEFAULT_PREREGISTRATION)) == []


def test_every_declared_configuration_is_defined_in_the_preregistration_and_no_other() -> None:
    text = registry.DEFAULT_PREREGISTRATION.read_text(encoding="utf-8")

    defined = set(re.findall(r"^\*\*([FC]\d[ab]?)\.", text, flags=re.MULTILINE))

    assert defined == registry.CONFIGURATIONS


def test_a_valid_full_history_has_no_problems() -> None:
    events = [
        created(),
        calibration(),
        run("F1a", "discovery"),
        run("C1", "discovery"),
        discovery_report(["F1a"]),
        run("F1a", "confirmation"),
    ]

    assert registry.problems(events, HASH) == []


def test_an_empty_registry_is_rejected() -> None:
    assert registry.problems([], HASH)


def test_the_registry_must_start_with_its_creation_event() -> None:
    found = registry.problems([calibration(), created()], HASH)

    assert any("first event" in reason for reason in found)


def test_editing_the_preregistration_without_an_amendment_is_caught() -> None:
    found = registry.problems([created(HASH)], OTHER_HASH)

    assert any("changed without an amendment" in reason for reason in found)


def test_an_amendment_with_a_reason_and_the_new_hash_is_accepted() -> None:
    amendment = {
        "event": "amendment",
        "date": "2026-09-21",
        "preregistration_sha256": OTHER_HASH,
        "reason": "typo in the F5a description, no rule changed",
    }

    assert registry.problems([created(HASH), amendment], OTHER_HASH) == []


def test_an_amendment_without_a_reason_is_rejected() -> None:
    amendment = {"event": "amendment", "date": "2026-09-21", "preregistration_sha256": OTHER_HASH}

    found = registry.problems([created(HASH), amendment], OTHER_HASH)

    assert any("needs a reason" in reason for reason in found)


@pytest.mark.parametrize("digest", ["", "abc", "G" * 64, None])
def test_a_document_event_needs_a_real_sha256(digest) -> None:
    event = created()
    event["preregistration_sha256"] = digest

    assert any("SHA-256" in reason for reason in registry.problems([event], HASH))


def test_a_run_of_an_undeclared_configuration_is_rejected() -> None:
    found = registry.problems([created(), calibration(), run("F9z", "discovery")], HASH)

    assert any("not declared" in reason for reason in found)


def test_a_discovery_run_before_the_calibration_report_is_rejected() -> None:
    found = registry.problems([created(), run("F1a", "discovery")], HASH)

    assert any("before the calibration report" in reason for reason in found)


def test_a_discovery_run_after_the_discovery_report_is_rejected() -> None:
    events = [created(), calibration(), discovery_report([]), run("F1a", "discovery")]

    assert any("after the discovery report" in r for r in registry.problems(events, HASH))


def test_the_frozen_sample_stays_closed_until_the_discovery_report() -> None:
    events = [created(), calibration(), run("F1a", "confirmation")]

    found = registry.problems(events, HASH)

    assert any("confirmation run before the discovery report" in reason for reason in found)


def test_only_approved_configurations_reach_the_frozen_sample() -> None:
    events = [created(), calibration(), discovery_report(["F1a"]), run("F2a", "confirmation")]

    found = registry.problems(events, HASH)

    assert any("was not approved" in reason for reason in found)


def test_a_discovery_report_before_the_calibration_report_is_rejected() -> None:
    found = registry.problems([created(), discovery_report([])], HASH)

    assert any("before the calibration report" in reason for reason in found)


def test_a_discovery_report_can_only_approve_declared_configurations() -> None:
    found = registry.problems([created(), calibration(), discovery_report(["ZZ"])], HASH)

    assert any("approved must list declared" in reason for reason in found)


@pytest.mark.parametrize("missing", ["code_commit", "result", "sample", "configuration"])
def test_a_run_needs_its_provenance(missing) -> None:
    event = run("F1a", "discovery")
    del event[missing]

    assert registry.problems([created(), calibration(), event], HASH)


def test_an_unknown_event_type_and_a_bad_date_are_rejected() -> None:
    found = registry.problems([created(), {"event": "note", "date": "2026-09-20"}], HASH)
    assert any("unknown event type" in reason for reason in found)

    undated = {**calibration(), "date": "yesterday"}
    assert any("ISO date" in reason for reason in registry.problems([created(), undated], HASH))


def test_a_line_that_is_not_json_is_an_error_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text(json.dumps(created()) + "\n{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        registry.read_events(path)


def test_a_line_that_is_not_an_object_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        registry.read_events(path)


def test_appending_builds_the_registry_and_refuses_a_rule_break_without_writing(
    tmp_path: Path,
) -> None:
    document = tmp_path / "preregistration.md"
    document.write_text("frozen text", encoding="utf-8")
    path = tmp_path / "registry.jsonl"
    digest = registry.sha256_of(document)

    registry.append_event(
        {"event": "registry_created", "preregistration_sha256": digest},
        registry=path,
        preregistration=document,
    )
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="before the calibration report"):
        registry.append_event(run("F1a", "discovery"), registry=path, preregistration=document)

    assert path.read_text(encoding="utf-8") == before
    assert registry.problems(registry.read_events(path), digest) == []


def test_appending_after_the_document_drifted_is_refused(tmp_path: Path) -> None:
    document = tmp_path / "preregistration.md"
    document.write_text("frozen text", encoding="utf-8")
    path = tmp_path / "registry.jsonl"
    registry.append_event(
        {"event": "registry_created", "preregistration_sha256": registry.sha256_of(document)},
        registry=path,
        preregistration=document,
    )
    document.write_text("frozen text, edited", encoding="utf-8")

    with pytest.raises(ValueError, match="changed without an amendment"):
        registry.append_event(calibration(), registry=path, preregistration=document)

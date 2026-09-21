"""The round-2 registry refuses what the pre-registration forbids: order, undeclared configurations, drift."""

import json
from pathlib import Path

import pytest

from scripts.research import fx_fast2_registry as reg

DECLARED = frozenset({"A", "B", "C"})
DOC, GRID = "a" * 64, "b" * 64
DAY = "2026-09-20"


def created() -> dict:
    return {"event": "registry_created", "date": DAY, "preregistration_sha256": DOC, "grid_sha256": GRID}


def event(kind: str, **fields) -> dict:
    return {"event": kind, "date": DAY, **fields}


def run(sample: str, configuration: str = "A") -> dict:
    return event("run", sample=sample, configuration=configuration, code_commit="abc", result={"t": 1.0})


def check(events: list[dict]) -> list[str]:
    return reg.problems(events, DOC, GRID, DECLARED)


def test_a_full_valid_history_passes() -> None:
    history = [created(), event("calibration_report", false_pass_rate=0.01), run("discovery", "A"), run("discovery", "B"),
               event("discovery_report", approved=["A", "B"]), run("replication", "A"),
               event("replication_report", approved=["A"]), run("confirmation", "A")]

    assert check(history) == []


def test_discovery_needs_the_calibration_first_and_closes_with_its_report() -> None:
    assert any("before the calibration" in p for p in check([created(), run("discovery")]))
    late = [created(), event("calibration_report"), event("discovery_report", approved=[]), run("discovery")]
    assert any("after the discovery report" in p for p in check(late))


def test_a_second_calibration_needs_a_documented_correction_first() -> None:
    twice = [created(), event("calibration_report"), event("calibration_report")]
    corrected = [created(), event("calibration_report"),
                 event("amendment", reason="fix", preregistration_sha256=DOC, grid_sha256=GRID), event("calibration_report")]

    assert any("second calibration report" in p for p in check(twice))
    assert check(corrected) == []


def test_a_configuration_outside_the_grid_or_that_did_not_pass_the_stage_before_is_refused() -> None:
    base = [created(), event("calibration_report")]
    assert any("not in the grid" in p for p in check([*base, run("discovery", "Z")]))
    after_discovery = [*base, event("discovery_report", approved=["A"]), run("replication", "B")]
    assert any("did not pass discovery" in p for p in check(after_discovery))
    after_replication = [*base, event("discovery_report", approved=["A", "B"]), event("replication_report", approved=["A"]),
                         run("confirmation", "B")]
    assert any("did not pass replication" in p for p in check(after_replication))


def test_confirmation_cannot_run_before_the_replication_report() -> None:
    history = [created(), event("calibration_report"), event("discovery_report", approved=["A"]), run("confirmation", "A")]

    assert any("before the replication report" in p for p in check(history))


def test_a_report_may_only_list_what_passed_the_stage_before() -> None:
    history = [created(), event("calibration_report"), event("discovery_report", approved=["A"]),
               event("replication_report", approved=["A", "B"])]

    assert any("must list configurations that passed" in p for p in check(history))


def test_the_document_and_the_grid_cannot_drift_without_an_amendment() -> None:
    assert reg.problems([created()], "c" * 64, GRID, DECLARED) == ["preregistration changed without an amendment event carrying its new SHA-256"]
    assert reg.problems([created()], DOC, "c" * 64, DECLARED) == ["grid changed without an amendment event carrying its new SHA-256"]
    amended = [created(), event("amendment", reason="typo", preregistration_sha256="c" * 64, grid_sha256=GRID)]
    assert reg.problems(amended, "c" * 64, GRID, DECLARED) == []
    assert any("needs a reason" in p for p in reg.problems([created(), event("amendment", preregistration_sha256=DOC, grid_sha256=GRID)], DOC, GRID, DECLARED))


def test_append_writes_nothing_when_the_result_would_break_the_rules(tmp_path: Path) -> None:
    grid, doc, registry = tmp_path / "grid.json", tmp_path / "doc.md", tmp_path / "registry.jsonl"
    grid.write_text(json.dumps([{"key": "A"}]))
    doc.write_text("doc")
    reg.append_event({"event": "registry_created", "preregistration_sha256": reg.sha256_of(doc),
                      "grid_sha256": reg.sha256_of(grid)}, registry=registry, preregistration=doc, grid=grid)

    with pytest.raises(ValueError, match="before the calibration"):
        reg.append_event({"event": "run", "sample": "discovery", "configuration": "A", "code_commit": "x", "result": {}},
                         registry=registry, preregistration=doc, grid=grid)

    assert len(registry.read_text().splitlines()) == 1


def test_the_committed_registry_matches_the_committed_document_and_grid() -> None:
    if not reg.DEFAULT_REGISTRY.exists():
        pytest.skip("the registry is created when the pre-registration is committed")

    assert reg.problems(reg.read_events(), reg.sha256_of(reg.DEFAULT_PREREGISTRATION), reg.sha256_of(reg.DEFAULT_GRID),
                        reg.declared_keys()) == []

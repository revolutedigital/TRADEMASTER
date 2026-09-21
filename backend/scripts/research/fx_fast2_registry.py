"""Append-only registry that enforces the round-2 pre-registration (docs/forex/fast2-preregistration.md).

Every step on real data is one JSON line in `docs/forex/fast2-registry.jsonl`. The registry stores the
SHA-256 of the pre-registration and of the grid file, so neither can change without a dated amendment,
and it refuses a configuration that is not in the grid, a stage out of order (calibration, then
discovery on S0, then replication on S1, then confirmation on S2) and a configuration that did not pass
the stage before.

    python -m scripts.research.fx_fast2_registry --create     # the first line, once
    python -m scripts.research.fx_fast2_registry --amend "why" # after a documented change, before any result
    python -m scripts.research.fx_fast2_registry              # check the registry

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREREGISTRATION = REPO_ROOT / "docs" / "forex" / "fast2-preregistration.md"
DEFAULT_GRID = REPO_ROOT / "docs" / "forex" / "fast2-grid.json"
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "forex" / "fast2-registry.jsonl"

SAMPLES = ("discovery", "replication", "confirmation")
DOCUMENT_EVENTS = frozenset({"registry_created", "amendment"})
REPORTS = {"discovery": "discovery_report", "replication": "replication_report"}
EVENTS = DOCUMENT_EVENTS | {"calibration_report", "discovery_report", "replication_report", "run"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

Event = dict[str, object]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_keys(grid: Path = DEFAULT_GRID) -> frozenset[str]:
    return frozenset(point["key"] for point in json.loads(grid.read_text(encoding="utf-8")))


def read_events(path: Path = DEFAULT_REGISTRY) -> list[Event]:
    """Parse the registry; a line that is not a JSON object is an error, never skipped."""
    events: list[Event] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"registry line {number} is not valid JSON: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"registry line {number} is not a JSON object")
        events.append(event)
    return events


def _is_iso_date(value: object) -> bool:
    try:
        date.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_PATTERN.match(value))


def problems(events: Sequence[Event], preregistration_sha256: str, grid_sha256: str,
             declared: frozenset[str]) -> list[str]:
    """Every way the registry breaks the pre-registration; an empty list means it is valid."""
    if not events:
        return ["the registry is empty: it must start with a registry_created event"]
    found: list[str] = []
    if events[0].get("event") != "registry_created":
        found.append("the first event must be registry_created")
    calibrated = False
    amended_since_calibration = False  # a second calibration is only for a documented correction
    approved: dict[str, set[str]] = {}  # stage -> configurations that passed it (once its report exists)
    registered: dict[str, str] = {}
    for position, event in enumerate(events, start=1):
        kind, where = event.get("event"), f"event {position}"
        if kind not in EVENTS:
            found.append(f"{where}: unknown event type {kind!r}")
            continue
        if not _is_iso_date(event.get("date")):
            found.append(f"{where}: date must be an ISO date")
        if kind in DOCUMENT_EVENTS:
            for field in ("preregistration_sha256", "grid_sha256"):
                if _is_digest(event.get(field)):
                    registered[field] = event[field]  # type: ignore[assignment]
                else:
                    found.append(f"{where}: {field} must be a SHA-256 hex digest")
            if kind == "amendment" and not _has_text(event.get("reason")):
                found.append(f"{where}: an amendment needs a reason")
            amended_since_calibration = True
        elif kind == "calibration_report":
            if calibrated and not amended_since_calibration:
                found.append(f"{where}: a second calibration report without an amendment after the first")
            calibrated, amended_since_calibration = True, False
        elif kind in ("discovery_report", "replication_report"):
            stage = "discovery" if kind == "discovery_report" else "replication"
            previous = None if stage == "discovery" else approved.get("discovery")
            listed = event.get("approved")
            if stage == "discovery" and not calibrated:
                found.append(f"{where}: discovery report before the calibration report")
            if stage == "replication" and previous is None:
                found.append(f"{where}: replication report before the discovery report")
            if stage in approved:
                found.append(f"{where}: a second {kind}")
            if not isinstance(listed, list) or not set(listed) <= (declared if previous is None else previous):
                found.append(f"{where}: approved must list configurations that passed the stage before")
            else:
                approved[stage] = set(listed)
        else:
            found.extend(_run_problems(event, where, calibrated, approved, declared))
    for field, current in (("preregistration_sha256", preregistration_sha256), ("grid_sha256", grid_sha256)):
        if registered.get(field) != current:
            found.append(f"{field.removesuffix('_sha256')} changed without an amendment event carrying its new SHA-256")
    return found


def _run_problems(event: Event, where: str, calibrated: bool, approved: dict[str, set[str]],
                  declared: frozenset[str]) -> list[str]:
    found: list[str] = []
    configuration, sample = event.get("configuration"), event.get("sample")
    if configuration not in declared:
        found.append(f"{where}: configuration {configuration!r} is not in the grid")
    if sample not in SAMPLES:
        found.append(f"{where}: sample must be one of {list(SAMPLES)}")
    elif sample == "discovery":
        if not calibrated:
            found.append(f"{where}: discovery run before the calibration report")
        if "discovery" in approved:
            found.append(f"{where}: discovery run after the discovery report")
    else:
        stage_before = "discovery" if sample == "replication" else "replication"
        if stage_before not in approved:
            found.append(f"{where}: {sample} run before the {stage_before} report")
        elif configuration not in approved[stage_before]:
            found.append(f"{where}: {configuration!r} did not pass {stage_before}")
        if sample == "replication" and "replication" in approved:
            found.append(f"{where}: replication run after the replication report")
    if not _has_text(event.get("code_commit")):
        found.append(f"{where}: a run needs the code_commit that produced it")
    if not isinstance(event.get("result"), dict):
        found.append(f"{where}: a run needs a result object")
    return found


def append_event(event: Event, *, registry: Path = DEFAULT_REGISTRY, preregistration: Path = DEFAULT_PREREGISTRATION,
                 grid: Path = DEFAULT_GRID) -> None:
    """Append one event, or raise without writing anything if the result would break the rules."""
    candidate: Event = {"date": date.today().isoformat(), **event}
    existing = read_events(registry) if registry.exists() else []
    found = problems([*existing, candidate], sha256_of(preregistration), sha256_of(grid), declared_keys(grid))
    if found:
        raise ValueError("; ".join(found))
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--create", action="store_true", help="write the registry_created line")
    parser.add_argument("--amend", metavar="REASON", help="record the current hashes of the document and the grid, with the reason")
    arguments = parser.parse_args(argv)
    if arguments.amend:
        append_event({"event": "amendment", "reason": arguments.amend, "preregistration_sha256": sha256_of(DEFAULT_PREREGISTRATION),
                      "grid_sha256": sha256_of(DEFAULT_GRID)})
    if arguments.create:
        append_event({"event": "registry_created", "preregistration_sha256": sha256_of(DEFAULT_PREREGISTRATION),
                      "grid_sha256": sha256_of(DEFAULT_GRID)})
    found = problems(read_events(), sha256_of(DEFAULT_PREREGISTRATION), sha256_of(DEFAULT_GRID), declared_keys())
    sys.stdout.write("\n".join(found) + "\n" if found else "registry ok\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())

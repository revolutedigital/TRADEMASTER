"""Append-only registry that enforces the fast-strategy pre-registration.

Every run on real data is one JSON line in `docs/forex/fast-registry.jsonl`. The registry stores the
SHA-256 of the pre-registration, so the document cannot change without a dated amendment line, and it
refuses a configuration the document does not declare, a discovery run before the calibration report
(or after the discovery report), and a confirmation run before the discovery report or for a
configuration that was not approved there.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREREGISTRATION = REPO_ROOT / "docs" / "forex" / "fast-preregistration.md"
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "forex" / "fast-registry.jsonl"

CONFIGURATIONS = frozenset(
    {"F1a", "F1b", "F2a", "F2b", "F3a", "F3b", "F4", "F5a", "F5b", "C1", "C2"}
)
SAMPLES = frozenset({"discovery", "confirmation"})
DOCUMENT_EVENTS = frozenset({"registry_created", "amendment"})
EVENTS = DOCUMENT_EVENTS | {"calibration_report", "discovery_report", "run"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

Event = dict[str, object]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def problems(events: Sequence[Event], preregistration_sha256: str) -> list[str]:
    """Every way the registry breaks the pre-registration; an empty list means it is valid."""
    if not events:
        return ["the registry is empty: it must start with a registry_created event"]
    found: list[str] = []
    if events[0].get("event") != "registry_created":
        found.append("the first event must be registry_created")
    calibrated = False
    approved: set[str] | None = None
    registered_hash: str | None = None
    for position, event in enumerate(events, start=1):
        kind = event.get("event")
        where = f"event {position}"
        if kind not in EVENTS:
            found.append(f"{where}: unknown event type {kind!r}")
            continue
        if not _is_iso_date(event.get("date")):
            found.append(f"{where}: date must be an ISO date")
        if kind in DOCUMENT_EVENTS:
            digest = event.get("preregistration_sha256")
            if isinstance(digest, str) and SHA256_PATTERN.match(digest):
                registered_hash = digest
            else:
                found.append(f"{where}: preregistration_sha256 must be a SHA-256 hex digest")
            if kind == "amendment" and not _has_text(event.get("reason")):
                found.append(f"{where}: an amendment needs a reason")
        elif kind == "calibration_report":
            calibrated = True
        elif kind == "discovery_report":
            listed = event.get("approved")
            if not calibrated:
                found.append(f"{where}: discovery report before the calibration report")
            if not isinstance(listed, list) or not set(listed) <= CONFIGURATIONS:
                found.append(f"{where}: approved must list declared configurations")
            else:
                approved = set(listed)
        else:
            found.extend(_run_problems(event, where, calibrated=calibrated, approved=approved))
    if registered_hash != preregistration_sha256:
        found.append(
            "the pre-registration changed without an amendment event carrying its new SHA-256"
        )
    return found


def _run_problems(
    event: Event, where: str, *, calibrated: bool, approved: set[str] | None
) -> list[str]:
    found: list[str] = []
    configuration = event.get("configuration")
    sample = event.get("sample")
    if configuration not in CONFIGURATIONS:
        found.append(f"{where}: configuration {configuration!r} is not declared")
    if sample not in SAMPLES:
        found.append(f"{where}: sample must be one of {sorted(SAMPLES)}")
    elif sample == "discovery":
        if not calibrated:
            found.append(f"{where}: discovery run before the calibration report")
        if approved is not None:
            found.append(f"{where}: discovery run after the discovery report")
    elif approved is None:
        found.append(f"{where}: confirmation run before the discovery report")
    elif configuration not in approved:
        found.append(f"{where}: {configuration!r} was not approved in the discovery report")
    if not _has_text(event.get("code_commit")):
        found.append(f"{where}: a run needs the code_commit that produced it")
    if not isinstance(event.get("result"), dict):
        found.append(f"{where}: a run needs a result object")
    return found


def append_event(
    event: Event,
    *,
    registry: Path = DEFAULT_REGISTRY,
    preregistration: Path = DEFAULT_PREREGISTRATION,
) -> None:
    """Append one event, or raise without writing anything if the result would break the rules."""
    candidate: Event = {"date": date.today().isoformat(), **event}
    existing = read_events(registry) if registry.exists() else []
    found = problems([*existing, candidate], sha256_of(preregistration))
    if found:
        raise ValueError("; ".join(found))
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    found = problems(read_events(), sha256_of(DEFAULT_PREREGISTRATION))
    sys.stdout.write("\n".join(found) + "\n" if found else "registry ok\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())

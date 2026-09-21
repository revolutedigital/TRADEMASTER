"""Append-only guard for the round-3 conditional short-trade research protocol."""

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
DEFAULT_PREREGISTRATION = REPO_ROOT / "docs" / "forex" / "fast3-preregistration.md"
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "forex" / "fast3-registry.jsonl"

DOCUMENT_EVENTS = frozenset({"registry_created", "amendment"})
EVENTS = DOCUMENT_EVENTS | {
    "development_run",
    "model_locked",
    "development_holdout_run",
    "development_report",
    "confirmation_run",
    "confirmation_report",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
Event = dict[str, object]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(path: Path = DEFAULT_REGISTRY) -> list[Event]:
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
    return isinstance(value, str) and bool(SHA256_PATTERN.fullmatch(value))


def problems(events: Sequence[Event], preregistration_sha256: str) -> list[str]:  # noqa: PLR0912
    """Return every protocol violation; an empty list means the history is valid."""
    if not events:
        return ["the registry is empty: it must start with registry_created"]
    found: list[str] = []
    if events[0].get("event") != "registry_created":
        found.append("the first event must be registry_created")
    registered_hash: str | None = None
    locked_policy: str | None = None
    holdout_ran = False
    development_passed = False
    confirmation_ran = False
    for position, event in enumerate(events, start=1):
        kind, where = event.get("event"), f"event {position}"
        if kind not in EVENTS:
            found.append(f"{where}: unknown event type {kind!r}")
            continue
        if not _is_iso_date(event.get("date")):
            found.append(f"{where}: date must be an ISO date")
        if kind in DOCUMENT_EVENTS:
            digest = event.get("preregistration_sha256")
            if _is_digest(digest):
                registered_hash = str(digest)
            else:
                found.append(f"{where}: preregistration_sha256 must be a SHA-256 digest")
            if kind == "amendment" and not _has_text(event.get("reason")):
                found.append(f"{where}: an amendment needs a reason")
            continue
        if not _has_text(event.get("code_commit")):
            found.append(f"{where}: execution needs code_commit")
        if kind == "development_run":
            if locked_policy is not None:
                found.append(f"{where}: development run after model_locked")
        elif kind == "model_locked":
            if locked_policy is not None:
                found.append(f"{where}: only one policy may be locked")
            if not _has_text(event.get("policy_id")) or not _is_digest(event.get("artifact_sha256")):
                found.append(f"{where}: model_locked needs policy_id and artifact_sha256")
            else:
                locked_policy = str(event["policy_id"])
        elif kind == "development_holdout_run":
            if locked_policy is None:
                found.append(f"{where}: D3 cannot run before model_locked")
            if holdout_ran:
                found.append(f"{where}: D3 may run only once")
            holdout_ran = True
        elif kind == "development_report":
            if not holdout_ran:
                found.append(f"{where}: development report before the D3 run")
            if not isinstance(event.get("approved"), bool):
                found.append(f"{where}: development report needs boolean approved")
            development_passed = event.get("approved") is True
        elif kind == "confirmation_run":
            if not development_passed:
                found.append(f"{where}: S2 cannot run before an approved development report")
            if confirmation_ran:
                found.append(f"{where}: S2 may run only once")
            confirmation_ran = True
        elif kind == "confirmation_report":
            if not confirmation_ran:
                found.append(f"{where}: confirmation report before the S2 run")
            if not isinstance(event.get("approved"), bool):
                found.append(f"{where}: confirmation report needs boolean approved")
    if registered_hash != preregistration_sha256:
        found.append("preregistration changed without an amendment carrying its new SHA-256")
    return found


def append_event(
    event: Event,
    *,
    registry: Path = DEFAULT_REGISTRY,
    preregistration: Path = DEFAULT_PREREGISTRATION,
) -> None:
    candidate: Event = {"date": date.today().isoformat(), **event}
    existing = read_events(registry) if registry.exists() else []
    found = problems([*existing, candidate], sha256_of(preregistration))
    if found:
        raise ValueError("; ".join(found))
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--amend", metavar="REASON")
    arguments = parser.parse_args(argv)
    if arguments.create:
        append_event(
            {
                "event": "registry_created",
                "preregistration_sha256": sha256_of(DEFAULT_PREREGISTRATION),
            }
        )
    if arguments.amend:
        append_event(
            {
                "event": "amendment",
                "reason": arguments.amend,
                "preregistration_sha256": sha256_of(DEFAULT_PREREGISTRATION),
            }
        )
    found = problems(read_events(), sha256_of(DEFAULT_PREREGISTRATION))
    sys.stdout.write("\n".join(found) + "\n" if found else "registry ok\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())


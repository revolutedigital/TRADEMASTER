"""Append-only protocol guard for round-5 probabilistic entries and trailing runners."""

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
DEFAULT_PREREGISTRATION = REPO_ROOT / "docs" / "forex" / "fast5-preregistration.md"
DEFAULT_REGISTRY = REPO_ROOT / "docs" / "forex" / "fast5-registry.jsonl"

DOCUMENT_EVENTS = frozenset({"registry_created", "amendment"})
EXECUTION_EVENTS = frozenset(
    {
        "factory_validated",
        "labels_materialized",
        "policies_frozen",
        "development_validation_run",
        "development_report",
        "blind_2022_run",
        "blind_2022_report",
        "holdout_run",
        "holdout_report",
        "confirmation_run",
        "confirmation_report",
        "broker_portability_run",
        "broker_portability_report",
    }
)
EVENTS = DOCUMENT_EVENTS | EXECUTION_EVENTS
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
Event = dict[str, object]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(path: Path = DEFAULT_REGISTRY) -> list[Event]:
    events: list[Event] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"registry line {line_number} is invalid JSON: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"registry line {line_number} is not an object")
        events.append(event)
    return events


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_PATTERN.fullmatch(value))


def _is_date(value: object) -> bool:
    try:
        date.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


def problems(events: Sequence[Event], preregistration_sha256: str) -> list[str]:  # noqa: PLR0912
    if not events:
        return ["registry is empty"]
    found: list[str] = []
    if events[0].get("event") != "registry_created":
        found.append("first event must be registry_created")
    registered_hash: str | None = None
    factory = labels = policies = development_run = development_passed = False
    blind_ran = blind_passed = holdout_ran = holdout_passed = False
    confirmation_ran = confirmation_passed = broker_ran = False
    for position, event in enumerate(events, start=1):
        kind = event.get("event")
        where = f"event {position}"
        if kind not in EVENTS:
            found.append(f"{where}: unknown event {kind!r}")
            continue
        if not _is_date(event.get("date")):
            found.append(f"{where}: invalid date")
        if kind in DOCUMENT_EVENTS:
            digest = event.get("preregistration_sha256")
            if _is_digest(digest):
                registered_hash = str(digest)
            else:
                found.append(f"{where}: preregistration_sha256 must be SHA-256")
            if kind == "amendment" and not _has_text(event.get("reason")):
                found.append(f"{where}: amendment needs reason")
            continue
        if not _has_text(event.get("code_commit")):
            found.append(f"{where}: execution needs code_commit")
        if kind == "factory_validated":
            factory = True
        elif kind == "labels_materialized":
            if not factory:
                found.append(f"{where}: labels require validated factory")
            labels = True
        elif kind == "policies_frozen":
            if not labels:
                found.append(f"{where}: policies require materialized labels")
            if policies:
                found.append(f"{where}: policies may be frozen only once")
            if not _is_digest(event.get("artifact_sha256")):
                found.append(f"{where}: policies need artifact_sha256")
            policies = True
        elif kind == "development_validation_run":
            if not policies:
                found.append(f"{where}: 2021-H2 requires frozen policies")
            if development_run:
                found.append(f"{where}: 2021-H2 may run only once")
            development_run = True
        elif kind == "development_report":
            if not development_run:
                found.append(f"{where}: development report requires its run")
            if not isinstance(event.get("approved"), bool):
                found.append(f"{where}: development report needs approved")
            development_passed = event.get("approved") is True
        elif kind == "blind_2022_run":
            if not development_passed:
                found.append(f"{where}: 2022 requires approved 2021-H2")
            if blind_ran:
                found.append(f"{where}: 2022 may run only once")
            blind_ran = True
        elif kind == "blind_2022_report":
            if not blind_ran:
                found.append(f"{where}: 2022 report requires its run")
            if not isinstance(event.get("approved"), bool):
                found.append(f"{where}: 2022 report needs approved")
            blind_passed = event.get("approved") is True
        elif kind == "holdout_run":
            if not blind_passed:
                found.append(f"{where}: D4 requires approved 2022")
            if holdout_ran:
                found.append(f"{where}: D4 may run only once")
            holdout_ran = True
        elif kind == "holdout_report":
            if not holdout_ran:
                found.append(f"{where}: D4 report requires its run")
            if not isinstance(event.get("approved"), bool):
                found.append(f"{where}: D4 report needs approved")
            holdout_passed = event.get("approved") is True
        elif kind == "confirmation_run":
            if not holdout_passed:
                found.append(f"{where}: S3 requires approved D4")
            if confirmation_ran:
                found.append(f"{where}: S3 may run only once")
            confirmation_ran = True
        elif kind == "confirmation_report":
            if not confirmation_ran:
                found.append(f"{where}: S3 report requires its run")
            if not isinstance(event.get("approved"), bool):
                found.append(f"{where}: S3 report needs approved")
            confirmation_passed = event.get("approved") is True
        elif kind == "broker_portability_run":
            if not confirmation_passed:
                found.append(f"{where}: broker week requires approved S3")
            if broker_ran:
                found.append(f"{where}: broker week may run only once")
            broker_ran = True
        elif kind == "broker_portability_report":
            if not broker_ran:
                found.append(f"{where}: broker report requires its run")
            if not isinstance(event.get("approved"), bool):
                found.append(f"{where}: broker report needs approved")
    if registered_hash != preregistration_sha256:
        found.append("preregistration changed without matching amendment")
    return found


def append_event(
    event: Event,
    *,
    registry: Path = DEFAULT_REGISTRY,
    preregistration: Path = DEFAULT_PREREGISTRATION,
) -> None:
    candidate = {"date": date.today().isoformat(), **event}
    existing = read_events(registry) if registry.exists() else []
    found = problems([*existing, candidate], sha256_of(preregistration))
    if found:
        raise ValueError("; ".join(found))
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amend", metavar="REASON")
    arguments = parser.parse_args(argv)
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

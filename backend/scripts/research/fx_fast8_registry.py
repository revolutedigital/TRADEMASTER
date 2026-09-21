"""Append-only protocol guard for the pre-activation time-stop study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.research.fx_fast5_registry import append_event, problems, read_events, sha256_of

REPO_ROOT = Path(__file__).resolve().parents[3]
PREREGISTRATION = REPO_ROOT / "docs" / "forex" / "fast8-preregistration.md"
REGISTRY = REPO_ROOT / "docs" / "forex" / "fast8-registry.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--amend", metavar="REASON")
    arguments = parser.parse_args(argv)
    if arguments.create:
        append_event(
            {"event": "registry_created", "preregistration_sha256": sha256_of(PREREGISTRATION)},
            registry=REGISTRY,
            preregistration=PREREGISTRATION,
        )
    if arguments.amend:
        append_event(
            {
                "event": "amendment",
                "reason": arguments.amend,
                "preregistration_sha256": sha256_of(PREREGISTRATION),
            },
            registry=REGISTRY,
            preregistration=PREREGISTRATION,
        )
    found = problems(read_events(REGISTRY), sha256_of(PREREGISTRATION))
    sys.stdout.write("\n".join(found) + "\n" if found else "registry ok\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())

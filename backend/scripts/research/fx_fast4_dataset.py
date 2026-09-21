"""Fail-closed access to the raw tick samples declared for round 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.research import fx_fast4_registry as registry_protocol
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_histdata_ticks import read_tick_zip

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TICK_DIR = REPO_ROOT / "backend" / "data" / "raw" / "histdata" / "zips"
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "forex" / "fast-data-manifest.csv"


@dataclass(frozen=True)
class Sample:
    name: str
    first_month: str
    end_month_exclusive: str
    protected: bool


SAMPLES = {
    "development": Sample("development", "2019-01", "2023-01", False),
    "holdout": Sample("holdout", "2023-01", "2024-09", True),
    "confirmation": Sample("confirmation", "2024-09", "2026-06", True),
}


class ProtectedSampleError(PermissionError):
    """The requested sample has not reached its pre-registered gate."""


def _protocol_state(events: list[dict[str, object]]) -> tuple[bool, bool, bool, bool]:
    locked = any(event.get("event") == "model_locked" for event in events)
    holdout_ran = any(event.get("event") == "development_holdout_run" for event in events)
    development_passed = any(
        event.get("event") == "development_report" and event.get("approved") is True
        for event in events
    )
    confirmation_ran = any(event.get("event") == "confirmation_run" for event in events)
    return locked, holdout_ran, development_passed, confirmation_ran


def guard_sample_access(
    sample_name: str,
    *,
    unlock_protected: bool = False,
    registry: Path = registry_protocol.DEFAULT_REGISTRY,
    preregistration: Path = registry_protocol.DEFAULT_PREREGISTRATION,
) -> Sample:
    try:
        sample = SAMPLES[sample_name]
    except KeyError as error:
        raise ValueError(f"unknown sample {sample_name!r}; choose one of {sorted(SAMPLES)}") from error
    if not sample.protected:
        return sample
    if not unlock_protected:
        raise ProtectedSampleError(f"{sample_name} is protected; explicit unlock is required")
    if not registry.exists():
        raise ProtectedSampleError("the round-4 registry does not exist")
    events = registry_protocol.read_events(registry)
    violations = registry_protocol.problems(events, registry_protocol.sha256_of(preregistration))
    if violations:
        raise ProtectedSampleError("invalid round-4 registry: " + "; ".join(violations))
    locked, holdout_ran, development_passed, confirmation_ran = _protocol_state(events)
    if sample_name == "holdout":
        if not locked:
            raise ProtectedSampleError("D4 is locked until model_locked is registered")
        if holdout_ran:
            raise ProtectedSampleError("D4 has already been run")
    elif sample_name == "confirmation":
        if not development_passed:
            raise ProtectedSampleError("S3 is locked until D4 has an approved development report")
        if confirmation_ran:
            raise ProtectedSampleError("S3 has already been run")
    return sample


def _included_months(pair: str, manifest: Path) -> frozenset[str]:
    frame = pd.read_csv(manifest, dtype={"pair": str, "month": str})
    if not {"pair", "month", "included"} <= set(frame.columns):
        raise ValueError(f"manifest {manifest} lacks pair, month, or included")
    included = frame[(frame["pair"] == pair) & frame["included"].astype(bool)]
    return frozenset(included["month"].tolist())


def _months(sample: Sample) -> tuple[str, ...]:
    start = pd.Period(sample.first_month, freq="M")
    stop = pd.Period(sample.end_month_exclusive, freq="M")
    return tuple(str(month) for month in pd.period_range(start, stop - 1, freq="M"))


def tick_archives(
    pair: str,
    sample_name: str,
    *,
    unlock_protected: bool = False,
    tick_dir: Path = DEFAULT_TICK_DIR,
    manifest: Path = DEFAULT_MANIFEST,
    registry: Path = registry_protocol.DEFAULT_REGISTRY,
    preregistration: Path = registry_protocol.DEFAULT_PREREGISTRATION,
) -> tuple[Path, ...]:
    """Resolve allowed archives without opening a protected file."""
    if pair not in ALL_PAIRS:
        raise ValueError(f"pair {pair!r} is outside the declared round-4 universe")
    sample = guard_sample_access(
        sample_name,
        unlock_protected=unlock_protected,
        registry=registry,
        preregistration=preregistration,
    )
    included = _included_months(pair, manifest)
    paths: list[Path] = []
    for month in _months(sample):
        if month not in included:
            continue
        compact = month.replace("-", "")
        path = tick_dir / f"DAT_ASCII_{pair}_T_{compact}.zip"
        if not path.exists():
            raise FileNotFoundError(f"missing declared tick archive {path}")
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no included tick archive found for {pair} in {sample_name}")
    return tuple(paths)


def load_tick_archive(path: Path, *, sample_name: str, pair: str, **guard_kwargs: object) -> pd.DataFrame:
    """Open one archive only if it belongs to the guarded sample and pair."""
    allowed = tick_archives(pair, sample_name, **guard_kwargs)
    resolved = path.resolve()
    if resolved not in {candidate.resolve() for candidate in allowed}:
        raise ValueError(f"{path} is outside sample {sample_name} for {pair}")
    return read_tick_zip(path, rule="europe")


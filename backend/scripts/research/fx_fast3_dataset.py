"""Manifest-aware, fail-closed dataset access for round 3.

The public entry point checks the append-only registry before it opens D3 or S2. Development code
must use this loader instead of opening the matrix files directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx import strategy as fx
from scripts.research import fx_fast3_registry as registry_protocol
from scripts.research.fx_dataset import ALL_PAIRS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRE_DIR = REPO_ROOT / "backend" / "data" / "lab_pre"
DEFAULT_RECENT_DIR = REPO_ROOT / "backend" / "data" / "lab"
DEFAULT_PRE_MANIFEST = REPO_ROOT / "docs" / "forex" / "fast2-data-manifest.csv"
DEFAULT_RECENT_MANIFEST = REPO_ROOT / "docs" / "forex" / "fast-data-manifest.csv"


@dataclass(frozen=True)
class Sample:
    name: str
    first_month: str
    end_month_exclusive: str
    protected: bool


SAMPLES = {
    "development": Sample("development", "2014-11", "2023-01", False),
    "holdout": Sample("holdout", "2023-01", "2024-09", True),
    "confirmation": Sample("confirmation", "2024-09", "2026-09", True),
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
    """Fail before opening protected data unless the registry is valid and the gate is open."""
    try:
        sample = SAMPLES[sample_name]
    except KeyError as error:
        raise ValueError(f"unknown sample {sample_name!r}; choose one of {sorted(SAMPLES)}") from error
    if not sample.protected:
        return sample
    if not unlock_protected:
        raise ProtectedSampleError(f"{sample_name} is protected; explicit unlock is required")
    if not registry.exists():
        raise ProtectedSampleError("the round-3 registry does not exist")
    events = registry_protocol.read_events(registry)
    violations = registry_protocol.problems(events, registry_protocol.sha256_of(preregistration))
    if violations:
        raise ProtectedSampleError("invalid round-3 registry: " + "; ".join(violations))
    locked, holdout_ran, development_passed, confirmation_ran = _protocol_state(events)
    if sample_name == "holdout":
        if not locked:
            raise ProtectedSampleError("D3 is locked until model_locked is registered")
        if holdout_ran:
            raise ProtectedSampleError("D3 has already been run")
    elif sample_name == "confirmation":
        if not development_passed:
            raise ProtectedSampleError("S2 is locked until D3 has an approved development report")
        if confirmation_ran:
            raise ProtectedSampleError("S2 has already been run")
    return sample


def _included_months(pair: str, manifests: tuple[Path, ...]) -> frozenset[str]:
    months: set[str] = set()
    for path in manifests:
        frame = pd.read_csv(path, dtype={"pair": str, "month": str})
        if not {"pair", "month", "included"} <= set(frame.columns):
            raise ValueError(f"manifest {path} lacks pair, month, or included")
        included = frame[(frame["pair"] == pair) & frame["included"].astype(bool)]
        months.update(included["month"].tolist())
    return frozenset(months)


def _open_pair_parts(pair: str, directories: tuple[Path, ...]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for directory in directories:
        path = directory / f"{pair}.npy"
        if path.exists():
            parts.append(np.asarray(np.load(path, mmap_mode="r")))
    if not parts:
        raise FileNotFoundError(f"no matrix found for {pair} in {[str(path) for path in directories]}")
    matrix = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if matrix.ndim != 2 or matrix.shape[1] != fx.BAR_WIDTH:
        raise ValueError(f"{pair} matrix must have shape (n, {fx.BAR_WIDTH})")
    order = np.argsort(matrix[:, fx.BAR_TIME], kind="stable")
    ordered = np.ascontiguousarray(matrix[order])
    if len(ordered) > 1:
        unique = np.concatenate(([True], np.diff(ordered[:, fx.BAR_TIME]) > 0))
        ordered = ordered[unique]
    return ordered


def load_pair_sample(
    pair: str,
    sample_name: str,
    *,
    unlock_protected: bool = False,
    pre_dir: Path = DEFAULT_PRE_DIR,
    recent_dir: Path = DEFAULT_RECENT_DIR,
    pre_manifest: Path = DEFAULT_PRE_MANIFEST,
    recent_manifest: Path = DEFAULT_RECENT_MANIFEST,
    registry: Path = registry_protocol.DEFAULT_REGISTRY,
    preregistration: Path = registry_protocol.DEFAULT_PREREGISTRATION,
) -> np.ndarray:
    """Load one sample only after applying protocol and committed manifest exclusions."""
    if pair not in ALL_PAIRS:
        raise ValueError(f"pair {pair!r} is outside the declared round-3 universe")
    sample = guard_sample_access(
        sample_name,
        unlock_protected=unlock_protected,
        registry=registry,
        preregistration=preregistration,
    )
    matrix = _open_pair_parts(pair, (pre_dir, recent_dir))
    month = matrix[:, fx.BAR_TIME].astype("datetime64[s]").astype("datetime64[M]")
    lower = np.datetime64(sample.first_month, "M")
    upper = np.datetime64(sample.end_month_exclusive, "M")
    in_range = (month >= lower) & (month < upper)
    allowed = _included_months(pair, (pre_manifest, recent_manifest))
    allowed_months = np.array(sorted(allowed), dtype="datetime64[M]")
    included = np.isin(month, allowed_months)
    return np.ascontiguousarray(matrix[in_range & included])

"""Protected sample boundaries and committed-manifest filtering for round 3."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.fx import strategy as fx
from scripts.research import fx_fast3_dataset as dataset
from scripts.research import fx_fast3_registry as protocol


def write_contract(tmp_path: Path) -> tuple[Path, Path]:
    document = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    document.write_text("locked protocol", encoding="utf-8")
    digest = hashlib.sha256(document.read_bytes()).hexdigest()
    registry.write_text(
        f'{{"date":"2026-09-21","event":"registry_created","preregistration_sha256":"{digest}"}}\n',
        encoding="utf-8",
    )
    return document, registry


def append(registry: Path, document: Path, event: dict[str, object]) -> None:
    protocol.append_event(event, registry=registry, preregistration=document)


def test_development_is_open_but_d3_requires_unlock_and_a_locked_model(tmp_path: Path) -> None:
    document, registry = write_contract(tmp_path)

    assert dataset.guard_sample_access("development", registry=registry, preregistration=document).name == "development"
    with pytest.raises(dataset.ProtectedSampleError, match="explicit unlock"):
        dataset.guard_sample_access("holdout", registry=registry, preregistration=document)
    with pytest.raises(dataset.ProtectedSampleError, match="model_locked"):
        dataset.guard_sample_access(
            "holdout", unlock_protected=True, registry=registry, preregistration=document
        )

    append(
        registry,
        document,
        {"event": "model_locked", "code_commit": "abc", "policy_id": "B2.H15", "artifact_sha256": "a" * 64},
    )
    assert dataset.guard_sample_access(
        "holdout", unlock_protected=True, registry=registry, preregistration=document
    ).name == "holdout"


def test_s2_requires_d3_approval_and_both_protected_samples_run_once(tmp_path: Path) -> None:
    document, registry = write_contract(tmp_path)
    append(
        registry,
        document,
        {"event": "model_locked", "code_commit": "abc", "policy_id": "B2.H15", "artifact_sha256": "a" * 64},
    )
    with pytest.raises(dataset.ProtectedSampleError, match="approved development report"):
        dataset.guard_sample_access(
            "confirmation", unlock_protected=True, registry=registry, preregistration=document
        )

    append(registry, document, {"event": "development_holdout_run", "code_commit": "abc"})
    append(registry, document, {"event": "development_report", "code_commit": "abc", "approved": True})
    with pytest.raises(dataset.ProtectedSampleError, match="already been run"):
        dataset.guard_sample_access(
            "holdout", unlock_protected=True, registry=registry, preregistration=document
        )
    assert dataset.guard_sample_access(
        "confirmation", unlock_protected=True, registry=registry, preregistration=document
    ).name == "confirmation"

    append(registry, document, {"event": "confirmation_run", "code_commit": "abc"})
    with pytest.raises(dataset.ProtectedSampleError, match="already been run"):
        dataset.guard_sample_access(
            "confirmation", unlock_protected=True, registry=registry, preregistration=document
        )


def test_loader_keeps_only_the_requested_range_and_included_months(tmp_path: Path) -> None:
    document, registry = write_contract(tmp_path)
    pre_dir, recent_dir = tmp_path / "pre", tmp_path / "recent"
    pre_dir.mkdir()
    recent_dir.mkdir()
    timestamps = pd.to_datetime(
        ["2014-11-03", "2014-12-03", "2022-12-03", "2023-01-03"], utc=True
    )
    seconds = ((timestamps - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    matrix = np.ones((4, fx.BAR_WIDTH), dtype=float)
    matrix[:, fx.BAR_TIME] = seconds
    np.save(pre_dir / "EURUSD.npy", matrix[:2])
    np.save(recent_dir / "EURUSD.npy", matrix[2:])
    pre_manifest = tmp_path / "pre.csv"
    recent_manifest = tmp_path / "recent.csv"
    pd.DataFrame(
        {"pair": ["EURUSD", "EURUSD"], "month": ["2014-11", "2014-12"], "included": [True, False]}
    ).to_csv(pre_manifest, index=False)
    pd.DataFrame(
        {"pair": ["EURUSD", "EURUSD"], "month": ["2022-12", "2023-01"], "included": [True, True]}
    ).to_csv(recent_manifest, index=False)

    result = dataset.load_pair_sample(
        "EURUSD",
        "development",
        pre_dir=pre_dir,
        recent_dir=recent_dir,
        pre_manifest=pre_manifest,
        recent_manifest=recent_manifest,
        registry=registry,
        preregistration=document,
    )

    months = result[:, fx.BAR_TIME].astype("datetime64[s]").astype("datetime64[M]")
    assert months.tolist() == [np.datetime64("2014-11"), np.datetime64("2022-12")]

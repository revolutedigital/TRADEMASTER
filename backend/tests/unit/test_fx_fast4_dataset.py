"""Temporal access guards for round-4 raw tick data."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.research import fx_fast4_dataset as dataset
from scripts.research import fx_fast4_registry as registry_protocol


def write_manifest(path: Path, months: list[str]) -> None:
    pd.DataFrame(
        [{"pair": "EURUSD", "month": month, "included": True} for month in months]
    ).to_csv(path, index=False)


def write_tick(path: Path, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ticks.csv", f"{stamp},1.10000,1.10010,0\n")


def test_development_paths_open_without_unlock(tmp_path: Path) -> None:
    tick_dir = tmp_path / "ticks"
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["2019-01"])
    expected = tick_dir / "DAT_ASCII_EURUSD_T_201901.zip"
    write_tick(expected, "20190102 120000000")

    paths = dataset.tick_archives("EURUSD", "development", tick_dir=tick_dir, manifest=manifest)
    assert paths == (expected,)


def test_protected_paths_fail_before_archive_is_opened(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["2023-01"])
    with pytest.raises(dataset.ProtectedSampleError, match="explicit unlock"):
        dataset.tick_archives(
            "EURUSD", "holdout", tick_dir=tmp_path / "missing", manifest=manifest
        )


def test_holdout_requires_locked_policy_even_with_explicit_unlock(tmp_path: Path) -> None:
    document = tmp_path / "prereg.md"
    registry = tmp_path / "registry.jsonl"
    document.write_text("contract", encoding="utf-8")
    registry.write_text(
        json.dumps(
            {
                "date": "2026-09-21",
                "event": "registry_created",
                "preregistration_sha256": registry_protocol.sha256_of(document),
            }
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(dataset.ProtectedSampleError, match="model_locked"):
        dataset.tick_archives(
            "EURUSD",
            "holdout",
            unlock_protected=True,
            tick_dir=tmp_path / "ticks",
            manifest=tmp_path / "missing.csv",
            registry=registry,
            preregistration=document,
        )


def test_loader_rejects_archive_outside_declared_sample(tmp_path: Path) -> None:
    tick_dir = tmp_path / "ticks"
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["2019-01"])
    allowed = tick_dir / "DAT_ASCII_EURUSD_T_201901.zip"
    outside = tick_dir / "DAT_ASCII_EURUSD_T_202301.zip"
    write_tick(allowed, "20190102 120000000")
    write_tick(outside, "20230102 120000000")
    with pytest.raises(ValueError, match="outside sample"):
        dataset.load_tick_archive(
            outside,
            sample_name="development",
            pair="EURUSD",
            tick_dir=tick_dir,
            manifest=manifest,
        )

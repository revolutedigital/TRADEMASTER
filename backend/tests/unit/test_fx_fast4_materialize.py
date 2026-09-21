"""Partition and manifest behavior for round-4 materialization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.fx.instruments import ConversionRates
from scripts.research import fx_fast4_materialize as materialize


def synthetic_ticks(count: int = 3_200) -> pd.DataFrame:
    index = pd.date_range("2021-01-04", periods=count, freq="250ms", tz="UTC")
    mid = 1.10 + np.cumsum(np.where(np.arange(count) % 4, 0.00001, -0.00001))
    return pd.DataFrame({"bid": mid - 0.00004, "ask": mid + 0.00004}, index=index)


def test_materialize_pair_writes_hashed_aligned_partitions(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "DAT_ASCII_EURUSD_T_202101.zip"
    archive.touch()
    monkeypatch.setattr(materialize, "tick_archives", lambda *args, **kwargs: (archive,))
    monkeypatch.setattr(materialize, "read_tick_zip", lambda *args, **kwargs: synthetic_ticks())

    report = materialize.materialize_pair(
        "EURUSD",
        "development",
        tmp_path / "output",
        ConversionRates({"EURUSD": 1.10}),
    )

    features = pd.read_parquet(tmp_path / "output" / "EURUSD-features.parquet")
    outcomes = pd.read_parquet(tmp_path / "output" / "EURUSD-outcomes.parquet")
    assert len(features) == len(outcomes) == report["events"]
    assert features.index.equals(outcomes.index)
    assert report["features_sha256"] != report["outcomes_sha256"]
    assert report["valid_outcomes"]["h30_long_terminal_r_base"] > 0
    assert outcomes["h30_entry_index"].dtype == np.dtype("float64")
    assert outcomes["h30_exit_index"].dtype == np.dtype("float64")


def test_month_slice_is_half_open() -> None:
    paths = tuple(Path(f"DAT_ASCII_EURUSD_T_2021{month:02d}.zip") for month in range(1, 5))
    selected = materialize._select_archives(paths, "2021-02", "2021-04")
    assert [path.name for path in selected] == [
        "DAT_ASCII_EURUSD_T_202102.zip",
        "DAT_ASCII_EURUSD_T_202103.zip",
    ]


def test_protected_sample_cannot_be_sliced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be sliced"):
        materialize.materialize(
            "holdout", tmp_path, pairs=("EURUSD",), first_month="2023-01"
        )


def test_existing_manifest_is_extended_without_losing_prior_pairs(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text(
        '{"sample":"development","pairs":{"EURUSD":{"events":12}}}\n', encoding="utf-8"
    )
    monkeypatch.setattr(materialize, "_median_rates", lambda: ConversionRates({"EURUSD": 1.10}))
    monkeypatch.setattr(
        materialize,
        "materialize_pair",
        lambda pair, *args, **kwargs: {"pair": pair, "clean_ticks": 100, "events": 10},
    )

    report = materialize.materialize("development", output, pairs=("GBPUSD",))

    assert report["pairs"]["EURUSD"]["events"] == 12
    assert report["pairs"]["GBPUSD"]["events"] == 10

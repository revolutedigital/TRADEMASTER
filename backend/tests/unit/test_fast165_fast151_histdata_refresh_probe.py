from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.research.run_fast165_fast151_histdata_refresh_probe import (
    TARGET_PAIRS,
    fast165_status,
    pair_bounds_from_directory,
    parquet_bounds,
)


def _write_month(directory: Path, pair: str, end: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2026-09-01T00:00:00Z", end=end, freq="1min")
    frame = pd.DataFrame(
        {
            "bid_open": 1.0,
            "bid_high": 1.0,
            "bid_low": 1.0,
            "bid_close": 1.0,
            "ask_open": 1.0001,
            "ask_high": 1.0001,
            "ask_low": 1.0001,
            "ask_close": 1.0001,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    frame.to_parquet(directory / f"{pair}_M1_202609_202609.parquet")


def _write_all_pairs(directory: Path, end: str) -> None:
    for pair in TARGET_PAIRS:
        _write_month(directory, pair, end)


def test_parquet_bounds_accepts_timestamp_column(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-09-01T00:00:00Z", "2026-09-18T20:59:00Z"],
                utc=True,
            ),
            "bid_close": [1.0, 1.1],
        }
    ).to_parquet(path)

    bounds = parquet_bounds(path)

    assert bounds["row_count"] == 2
    assert bounds["start"] == "2026-09-01T00:00:00+00:00"
    assert bounds["end"] == "2026-09-18T20:59:00+00:00"


def test_fast165_status_confirms_source_still_before_preregistration(tmp_path: Path) -> None:
    local = tmp_path / "local"
    fresh = tmp_path / "fresh"
    _write_all_pairs(local, "2026-09-18T20:59:00Z")
    _write_all_pairs(fresh, "2026-09-18T20:59:00Z")

    status, reasons = fast165_status(
        local_bounds=pair_bounds_from_directory(local, month="2026-09"),
        fresh_bounds=pair_bounds_from_directory(fresh, month="2026-09"),
    )

    assert status == "FAST165_HISTDATA_REFRESH_CONFIRMS_SOURCE_NOT_POST_PREREGISTRATION"
    assert "fresh_histdata_min_pair_end_before_preregistration_timestamp" in reasons
    assert "not_approved_strategy" in reasons


def test_fast165_status_requires_import_when_fresh_source_is_newer(tmp_path: Path) -> None:
    local = tmp_path / "local"
    fresh = tmp_path / "fresh"
    _write_all_pairs(local, "2026-09-18T20:59:00Z")
    _write_all_pairs(fresh, "2026-09-24T12:00:00Z")

    status, reasons = fast165_status(
        local_bounds=pair_bounds_from_directory(local, month="2026-09"),
        fresh_bounds=pair_bounds_from_directory(fresh, month="2026-09"),
    )

    assert status == "FAST165_HISTDATA_REFRESH_SOURCE_NEWER_THAN_LOCAL_IMPORT_REQUIRED"
    assert "fresh_histdata_has_post_preregistration_data" in reasons
    assert "canonical_m1_import_required" in reasons

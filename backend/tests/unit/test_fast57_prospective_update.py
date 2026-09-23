from datetime import date
from pathlib import Path

from scripts.research.run_fast57_prospective_update import (
    date_range,
    fast57_block_arguments,
    fast57_research_dataset_config,
    latest_complete_utc_archive_date,
    normalized_trade_partition_exists,
    research_partition_exists,
)


def test_latest_complete_utc_archive_date_is_yesterday() -> None:
    assert latest_complete_utc_archive_date(date(2026, 9, 23)) == date(2026, 9, 22)


def test_date_range_is_inclusive_and_empty_when_reversed() -> None:
    assert date_range(date(2026, 9, 21), date(2026, 9, 23)) == (
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
    )
    assert date_range(date(2026, 9, 23), date(2026, 9, 22)) == ()


def test_partition_guards_use_expected_fast57_layout(tmp_path: Path) -> None:
    trade_root = tmp_path / "normalized" / "aggTrades"
    research_root = tmp_path / "research"
    utc_date = date(2026, 9, 23)

    assert normalized_trade_partition_exists(trade_root, utc_date) is False
    assert research_partition_exists(research_root, utc_date) is False

    (trade_root / "date=2026-09-23").mkdir(parents=True)
    (trade_root / "date=2026-09-23" / "events.parquet").write_bytes(b"")
    (research_root / "date=2026-09-23").mkdir(parents=True)
    (research_root / "date=2026-09-23" / "research_rows.parquet").write_bytes(b"")

    assert normalized_trade_partition_exists(trade_root, utc_date) is True
    assert research_partition_exists(research_root, utc_date) is True


def test_fast57_block_arguments_keep_seed_before_prospective(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    prospective = tmp_path / "prospective"
    (seed / "date=2026-09-21").mkdir(parents=True)
    (seed / "date=2026-09-21" / "research_rows.parquet").write_bytes(b"")
    (prospective / "date=2026-09-23").mkdir(parents=True)
    (prospective / "date=2026-09-23" / "research_rows.parquet").write_bytes(b"")

    assert fast57_block_arguments(
        seed_research_root=seed,
        prospective_research_root=prospective,
    ) == [
        f"seed={seed}",
        f"prospective={prospective}",
    ]


def test_fast57_research_dataset_config_is_low_cost_frozen_shape() -> None:
    config = fast57_research_dataset_config()

    assert config.decision_stride_seconds == 5
    assert config.horizons_seconds == (30, 120, 300)
    assert config.expected_cost_bps == 1.0
    assert config.stress_cost_bps == 2.0
    assert config.initial_stop_bps == 20.0

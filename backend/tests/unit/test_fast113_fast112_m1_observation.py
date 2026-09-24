from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research.run_fast112_usdjpy_breakout_observation import (
    OBSERVATION_CONFIG,
    OBSERVATION_PAIR,
)
from scripts.research.run_fast113_fast112_m1_observation import (
    SAFETY_FLAGS,
    build_report,
    empty_trade_rows,
    parse_optional_timestamp,
    source_validation_summary,
    trade_rows_from_simulated_trades,
)


def test_trade_rows_from_simulated_trades_adds_fast112_metadata() -> None:
    trades = pd.DataFrame(
        [
            {
                "entry_time": pd.Timestamp("2026-09-24T07:15:00Z"),
                "side": 1,
                "stop_pips": 100.5,
                "r_multiple": 0.25,
            }
        ]
    )

    rows = trade_rows_from_simulated_trades(
        config_key=OBSERVATION_CONFIG,
        pair=OBSERVATION_PAIR,
        scenario="stress",
        trades=trades,
    )

    assert rows == [
        {
            "stage": "m1_materialized",
            "config": OBSERVATION_CONFIG,
            "pair": OBSERVATION_PAIR,
            "scenario": "stress",
            "entry_time": "2026-09-24T07:15:00+00:00",
            "entry_epoch": 1790234100.0,
            "fx_day": rows[0]["fx_day"],
            "london_weekday": 3,
            "london_weekday_name": "Thursday",
            "london_entry_hour": 8,
            "london_month": 9,
            "year": 2026,
            "side": 1,
            "side_label": "LONG",
            "stop_pips": 100.5,
            "r_multiple": 0.25,
        }
    ]


def test_parse_optional_timestamp_requires_timezone() -> None:
    assert parse_optional_timestamp("2026-09-24T07:15:00Z", name="start") == pd.Timestamp(
        "2026-09-24T07:15:00Z"
    )

    with pytest.raises(ValueError, match="start must include a timezone"):
        parse_optional_timestamp("2026-09-24 07:15:00", name="start")


def test_empty_trade_rows_exposes_columns_required_by_fast112_filter() -> None:
    rows = empty_trade_rows()

    assert {
        "stage",
        "config",
        "pair",
        "scenario",
        "entry_time",
        "london_entry_hour",
        "stop_pips",
        "r_multiple",
        "side_label",
        "year",
    }.issubset(rows.columns)


def test_source_validation_summary_flags_unvalidated_month(tmp_path: Path) -> None:
    (tmp_path / "USDJPY_validation.csv").write_text(
        "pair,month,minutes,hours_compared,bid_match,ask_match,crossed_minutes,passes\n"
        "USDJPY,2026-09,19680,0,0.0,0.0,0,False\n",
        encoding="utf-8",
    )

    summary = source_validation_summary(tmp_path, "USDJPY")

    assert summary["available"] is True
    assert summary["all_reported_months_passed"] is False
    assert summary["failed_reported_months"] == ["2026-09"]
    assert summary["zero_oracle_hour_months"] == ["2026-09"]


def test_build_report_preserves_research_only_safety_flags(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text("{}\n", encoding="utf-8")
    materialized = tmp_path / "materialized_trade_rows.parquet"
    materialized.write_bytes(b"materialized")
    observations = tmp_path / "observations.jsonl"
    observations.write_text("", encoding="utf-8")
    raw_m1 = pd.DataFrame(index=pd.DatetimeIndex(["2026-09-01T03:59:00Z"]))

    report = build_report(
        preregistration={"preregistration_sha256": "sha"},
        raw_m1_dir=tmp_path,
        source_files=[{"path": "source.parquet", "sha256": "source-sha", "bytes": 10}],
        source_validation={"available": False},
        raw_m1=raw_m1,
        materialized_rows=empty_trade_rows(),
        materialized_path=materialized,
        observations_path=observations,
        observation_records=[],
        summary={"stress": {}, "base": {}},
        gates={"shadow_candidate": False},
        status="FAST112_OBSERVATION_NO_ELIGIBLE_TRADES_YET",
        status_reasons=["not_approved_strategy"],
        preregistration_path=tmp_path / "prereg.md",
        registry_path=registry,
        start=None,
        end=None,
    )

    for key, value in SAFETY_FLAGS.items():
        assert report[key] is value
    assert report["eligible_observation_count"] == 0
    assert report["materialization_source"] == "raw_m1"

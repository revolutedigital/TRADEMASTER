from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research.run_fast124_usdjpy_compression_observation import (
    FROZEN_POLICY,
    OBSERVATION_PAIR,
    PREREGISTRATION_TIMESTAMP,
    SAFETY_FLAGS,
    build_report,
    empty_trade_rows,
    evaluate_gates,
    fast124_status,
    parse_optional_timestamp,
    select_eligible_observations,
    source_validation_summary,
    summarize_observations,
)


def _row(
    *,
    entry_timestamp: str,
    net_base_pips: float = 4.0,
    net_stress_pips: float = 3.0,
    pair: str = OBSERVATION_PAIR,
    policy_id: str = FROZEN_POLICY.identifier,
    timeframe: str = FROZEN_POLICY.timeframe,
    channel_bars: int = FROZEN_POLICY.channel_bars,
    compression_quantile: float = FROZEN_POLICY.compression_quantile,
    stop_atr_multiple: float = FROZEN_POLICY.stop_atr_multiple,
    max_hold_minutes: int = FROZEN_POLICY.max_hold_minutes,
    management: str = FROZEN_POLICY.management.name,
) -> dict:
    entry = pd.Timestamp(entry_timestamp)
    return {
        "candidate_id": f"{pair}:{policy_id}",
        "policy_id": policy_id,
        "pair": pair,
        "timeframe": timeframe,
        "channel_bars": channel_bars,
        "compression_quantile": compression_quantile,
        "stop_atr_multiple": stop_atr_multiple,
        "max_hold_minutes": max_hold_minutes,
        "management": management,
        "signal_timestamp": entry - pd.Timedelta(minutes=1),
        "entry_timestamp": entry,
        "side": "LONG",
        "breakout_distance_pips": 2.0,
        "atr_pips": 15.0,
        "risk_pips": 22.5,
        "compression_value_pips": 10.0,
        "compression_threshold_pips": 11.0,
        "raw_base_pips": net_base_pips + 1.0,
        "net_base_pips": net_base_pips,
        "exit_base_timestamp": entry + pd.Timedelta(minutes=120),
        "exit_base_reason": "target",
        "raw_stress_pips": net_stress_pips + 1.0,
        "net_stress_pips": net_stress_pips,
        "exit_stress_timestamp": entry + pd.Timedelta(minutes=120),
        "exit_stress_reason": "target",
        "base_commission_pips": 0.2,
        "stress_commission_pips": 0.4,
    }


def test_select_eligible_observations_requires_post_preregistration_exact_frozen_rule() -> None:
    rows = pd.DataFrame(
        [
            _row(entry_timestamp="2026-09-24T04:50:13Z"),
            _row(entry_timestamp="2026-09-24T04:50:12Z"),
            _row(entry_timestamp="2026-09-24T04:50:13Z", pair="EURUSD"),
            _row(entry_timestamp="2026-09-24T04:50:13Z", policy_id="other"),
            _row(entry_timestamp="2026-09-24T04:50:13Z", management="trail_1r_1r"),
        ]
    )

    selected = select_eligible_observations(rows, prospective_start=PREREGISTRATION_TIMESTAMP)

    assert len(selected) == 1
    assert selected.iloc[0]["pair"] == OBSERVATION_PAIR
    assert selected.iloc[0]["policy_id"] == FROZEN_POLICY.identifier
    assert selected.iloc[0]["management"] == FROZEN_POLICY.management.name


def test_summarize_observations_and_gates_detect_shadow_candidate_and_early_fail() -> None:
    candidate_rows = []
    for index in range(100):
        year = 2027 + index // 50
        candidate_rows.append(
            _row(
                entry_timestamp=f"{year}-01-{(index % 28) + 1:02d}T04:50:13Z",
                net_base_pips=2.0,
                net_stress_pips=2.0,
            )
        )
    summary = summarize_observations(pd.DataFrame(candidate_rows), alpha=0.05)

    assert summary["eligible_observation_count"] == 100
    assert evaluate_gates(summary)["shadow_candidate"] is True

    fail_rows = pd.DataFrame(
        [
            _row(
                entry_timestamp=f"2027-01-{(index % 28) + 1:02d}T04:50:13Z",
                net_stress_pips=-3.0,
            )
            for index in range(50)
        ]
    )
    failed = summarize_observations(fail_rows, alpha=0.05)
    gates = evaluate_gates(failed)

    assert gates["early_fail"] is True
    assert "early_fail_stress_mean_pips" in gates["early_fail_reasons"]


def test_fast124_status_never_approves_strategy() -> None:
    empty = summarize_observations(empty_trade_rows(), alpha=0.05)
    status, reasons = fast124_status(summary=empty, gates=evaluate_gates(empty))

    assert status == "FAST124_OBSERVATION_NO_ELIGIBLE_TRADES_YET"
    assert "not_approved_strategy" in reasons

    small = summarize_observations(
        pd.DataFrame([_row(entry_timestamp="2027-01-01T04:50:13Z")]),
        alpha=0.05,
    )
    status, reasons = fast124_status(summary=small, gates=evaluate_gates(small))

    assert status == "FAST124_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    assert "insufficient_stress_trades" in reasons


def test_parse_optional_timestamp_requires_timezone() -> None:
    assert parse_optional_timestamp("2026-09-24T04:50:13Z", name="start") == pd.Timestamp(
        "2026-09-24T04:50:13Z"
    )

    with pytest.raises(ValueError, match="start must include a timezone"):
        parse_optional_timestamp("2026-09-24 04:50:13", name="start")


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


def test_source_validation_summary_treats_missing_passes_as_unverified(tmp_path: Path) -> None:
    (tmp_path / "USDJPY_validation.csv").write_text(
        "pair,month,minutes\n"
        "USDJPY,2026-09,19680\n",
        encoding="utf-8",
    )

    summary = source_validation_summary(tmp_path, "USDJPY")

    assert summary["available"] is True
    assert summary["reported_months"] == ["2026-09"]
    assert summary["all_reported_months_passed"] is False
    assert summary["note"] == "validation_csv_without_passes_column_treated_as_unverified"


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
        summary={"stress": {}, "base": {}, "eligible_observation_count": 0},
        gates={"shadow_candidate": False},
        status="FAST124_OBSERVATION_NO_ELIGIBLE_TRADES_YET",
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

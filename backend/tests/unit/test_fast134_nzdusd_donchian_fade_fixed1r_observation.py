from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research.run_fast134_nzdusd_donchian_fade_fixed1r_observation import (
    FROZEN_ENTRY_POLICY,
    FROZEN_POLICY_ID,
    FROZEN_RISK_POLICY,
    PAIR,
    PREREGISTRATION_TIMESTAMP,
    SAFETY_FLAGS,
    build_report,
    empty_trade_rows,
    evaluate_gates,
    fast134_status,
    select_eligible_observations,
    summarize_observations,
    validate_parameters,
    validate_preregistration,
)


def _row(
    *,
    entry_timestamp: str,
    net_base_pips: float = 30.0,
    net_stress_pips: float = 30.0,
    pair: str = PAIR,
    candidate_id: str = FROZEN_POLICY_ID,
    entry_policy_id: str = FROZEN_ENTRY_POLICY.identifier,
    risk_policy_id: str = FROZEN_RISK_POLICY.identifier,
    channel_days: int = FROZEN_ENTRY_POLICY.channel_days,
    hold_days: int = FROZEN_ENTRY_POLICY.hold_days,
    stop_atr_multiple: float = FROZEN_RISK_POLICY.stop_atr_multiple,
    management: str = FROZEN_RISK_POLICY.management.name,
) -> dict:
    entry = pd.Timestamp(entry_timestamp)
    return {
        "candidate_id": candidate_id,
        "pair": pair,
        "entry_policy_id": entry_policy_id,
        "risk_policy_id": risk_policy_id,
        "channel_days": channel_days,
        "hold_days": hold_days,
        "stop_atr_multiple": stop_atr_multiple,
        "management": management,
        "signal_timestamp": entry - pd.Timedelta(minutes=1),
        "entry_timestamp": entry,
        "side": "LONG",
        "daily_atr_pips": 55.0,
        "risk_pips": 110.0,
        "breakout_distance_pips": 7.0,
        "raw_base_pips": net_base_pips + 2.0,
        "net_base_pips": net_base_pips,
        "exit_base_timestamp": entry + pd.Timedelta(days=3),
        "exit_base_reason": "target",
        "raw_stress_pips": net_stress_pips + 2.0,
        "net_stress_pips": net_stress_pips,
        "exit_stress_timestamp": entry + pd.Timedelta(days=3),
        "exit_stress_reason": "target",
    }


def test_validate_preregistration_uses_real_fast134_registry() -> None:
    result = validate_preregistration(
        preregistration_path=Path(
            "docs/forex/fast134-nzdusd-donchian-fade-fixed1r-prospective-preregistration.md"
        ),
        registry_path=Path("docs/forex/fast134-registry.jsonl"),
    )

    assert result["preregistration_sha256"] == "30cba913140050354521dc0651f7b35247b3fe6f6fff6f9d5b4b38ca45c13ac0"


def test_validate_parameters_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        validate_parameters(alpha=0.0)


def test_select_eligible_observations_requires_post_preregistration_exact_frozen_rule() -> None:
    rows = pd.DataFrame(
        [
            _row(entry_timestamp="2026-09-24T07:04:41Z"),
            _row(entry_timestamp="2026-09-24T07:04:40Z"),
            _row(entry_timestamp="2026-09-24T07:04:41Z", pair="EURUSD"),
            _row(entry_timestamp="2026-09-24T07:04:41Z", candidate_id="other"),
            _row(entry_timestamp="2026-09-24T07:04:41Z", risk_policy_id="stop2_fixed_2r"),
            _row(entry_timestamp="2026-09-24T07:04:41Z", management="fixed_2r"),
        ]
    )

    selected = select_eligible_observations(rows, prospective_start=PREREGISTRATION_TIMESTAMP)

    assert len(selected) == 1
    assert selected.iloc[0]["candidate_id"] == FROZEN_POLICY_ID
    assert selected.iloc[0]["risk_policy_id"] == FROZEN_RISK_POLICY.identifier
    assert selected.iloc[0]["management"] == FROZEN_RISK_POLICY.management.name


def test_summarize_observations_and_gates_detect_shadow_candidate_and_early_fail() -> None:
    candidate_rows = []
    for index in range(20):
        year = 2027 + index // 10
        candidate_rows.append(
            _row(
                entry_timestamp=f"{year}-01-{(index % 10) + 1:02d}T07:04:41Z",
                net_base_pips=35.0,
                net_stress_pips=30.0,
            )
        )
    summary = summarize_observations(pd.DataFrame(candidate_rows), alpha=0.05)

    assert summary["eligible_observation_count"] == 20
    assert evaluate_gates(summary)["shadow_candidate"] is True

    fail_rows = pd.DataFrame(
        [
            _row(
                entry_timestamp=f"2027-01-{index + 1:02d}T07:04:41Z",
                net_base_pips=-11.0,
                net_stress_pips=-13.0,
            )
            for index in range(8)
        ]
    )
    failed = summarize_observations(fail_rows, alpha=0.05)
    gates = evaluate_gates(failed)

    assert gates["early_fail"] is True
    assert "early_fail_stress_total_pips" in gates["early_fail_reasons"]


def test_fast134_status_never_approves_strategy() -> None:
    empty = summarize_observations(empty_trade_rows(), alpha=0.05)
    status, reasons = fast134_status(summary=empty, gates=evaluate_gates(empty))

    assert status == "FAST134_OBSERVATION_NO_ELIGIBLE_TRADES_YET"
    assert "not_approved_strategy" in reasons

    small = summarize_observations(
        pd.DataFrame([_row(entry_timestamp="2027-01-01T07:04:41Z")]),
        alpha=0.05,
    )
    status, reasons = fast134_status(summary=small, gates=evaluate_gates(small))

    assert status == "FAST134_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    assert "insufficient_stress_trades" in reasons


def test_build_report_preserves_research_only_safety_flags(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text("{}\n", encoding="utf-8")
    signals = tmp_path / "signals.parquet"
    signals.write_bytes(b"signals")
    materialized = tmp_path / "materialized_trade_rows.parquet"
    materialized.write_bytes(b"materialized")
    observations = tmp_path / "observations.jsonl"
    observations.write_text("", encoding="utf-8")

    report = build_report(
        preregistration={"preregistration_sha256": "sha"},
        m1_dir=tmp_path,
        source_files=[{"path": "source.parquet", "sha256": "source-sha", "bytes": 10}],
        source_validation={"available": False},
        pair_quality={
            "usable_bars": 1,
            "first_bar": "2026-09-01T03:59:00+00:00",
            "last_bar": "2026-09-01T03:59:00+00:00",
            "dropped_crossed_quote_bars": 0,
        },
        signals=empty_trade_rows(),
        signals_path=signals,
        materialized_rows=empty_trade_rows(),
        materialized_path=materialized,
        observations_path=observations,
        observation_records=[],
        summary={"stress": {}, "base": {}, "eligible_observation_count": 0},
        gates={"shadow_candidate": False},
        status="FAST134_OBSERVATION_NO_ELIGIBLE_TRADES_YET",
        status_reasons=["not_approved_strategy"],
        preregistration_path=tmp_path / "prereg.md",
        registry_path=registry,
    )

    for key, value in SAFETY_FLAGS.items():
        assert report[key] is value
    assert report["eligible_observation_count"] == 0
    assert report["materialization_source"] == "raw_m1"
    assert report["frozen_policy_id"] == FROZEN_POLICY_ID

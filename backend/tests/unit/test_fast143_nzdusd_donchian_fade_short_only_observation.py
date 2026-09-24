from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.research.run_fast131_nzdusd_donchian_fade_observation import (
    SAFETY_FLAGS,
    summarize_observations,
)
from scripts.research.run_fast143_nzdusd_donchian_fade_short_only_observation import (
    BASE_FROZEN_POLICY_ID,
    FROZEN_ENTRY_POLICY,
    FROZEN_POLICY_ID,
    FROZEN_RISK_POLICY,
    PREREGISTRATION_SHA256,
    PREREGISTRATION_TIMESTAMP,
    SIDE_FILTER,
    evaluate_fast143_gates,
    fast143_status,
    select_eligible_observations,
    validate_preregistration,
)


def _row(
    *,
    entry_timestamp: str,
    side: str = SIDE_FILTER,
    net_base_pips: float = 40.0,
    net_stress_pips: float = 40.0,
    candidate_id: str = BASE_FROZEN_POLICY_ID,
    risk_policy_id: str = FROZEN_RISK_POLICY.identifier,
    management: str = FROZEN_RISK_POLICY.management.name,
) -> dict:
    entry = pd.Timestamp(entry_timestamp)
    return {
        "candidate_id": candidate_id,
        "pair": "NZDUSD",
        "entry_policy_id": FROZEN_ENTRY_POLICY.identifier,
        "risk_policy_id": risk_policy_id,
        "channel_days": FROZEN_ENTRY_POLICY.channel_days,
        "hold_days": FROZEN_ENTRY_POLICY.hold_days,
        "stop_atr_multiple": FROZEN_RISK_POLICY.stop_atr_multiple,
        "management": management,
        "signal_timestamp": entry - pd.Timedelta(minutes=1),
        "entry_timestamp": entry,
        "side": side,
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


def test_validate_preregistration_uses_real_fast143_registry() -> None:
    result = validate_preregistration(
        preregistration_path=Path(
            "docs/forex/fast143-nzdusd-donchian-fade-short-only-prospective-preregistration.md"
        ),
        registry_path=Path("docs/forex/fast143-registry.jsonl"),
    )

    assert result["preregistration_sha256"] == PREREGISTRATION_SHA256


def test_select_eligible_observations_requires_post_preregistration_short_only() -> None:
    rows = pd.DataFrame(
        [
            _row(entry_timestamp="2026-09-24T08:25:15Z"),
            _row(entry_timestamp="2026-09-24T08:25:14Z"),
            _row(entry_timestamp="2026-09-24T08:25:15Z", side="LONG"),
            _row(entry_timestamp="2026-09-24T08:25:15Z", candidate_id="other"),
            _row(entry_timestamp="2026-09-24T08:25:15Z", risk_policy_id="stop2_fixed_2r"),
            _row(entry_timestamp="2026-09-24T08:25:15Z", management="fixed_2r"),
        ]
    )

    selected = select_eligible_observations(rows, prospective_start=PREREGISTRATION_TIMESTAMP)

    assert len(selected) == 1
    assert selected.iloc[0]["side"] == SIDE_FILTER
    assert selected.iloc[0]["fast143_policy_id"] == FROZEN_POLICY_ID
    assert selected.iloc[0]["side_filter"] == SIDE_FILTER


def test_fast143_gates_use_short_only_shadow_thresholds() -> None:
    rows = []
    for index in range(12):
        year = 2027 + index // 6
        rows.append(
            _row(
                entry_timestamp=f"{year}-01-{(index % 6) + 1:02d}T08:25:15Z",
                net_base_pips=45.0,
                net_stress_pips=40.0,
            )
        )
    observations = pd.DataFrame(rows)
    summary = summarize_observations(observations, alpha=0.05)
    gates = evaluate_fast143_gates(summary=summary, observations=observations)

    assert gates["shadow_candidate"] is True
    assert gates["non_short_observation_count"] == 0


def test_fast143_status_never_approves_strategy() -> None:
    empty = summarize_observations(pd.DataFrame(columns=list(_row(entry_timestamp="2027-01-01T08:25:15Z"))), alpha=0.05)
    status, reasons = fast143_status(
        summary=empty,
        gates={
            "early_fail": False,
            "early_fail_reasons": [],
            "shadow_candidate": False,
            "shadow_candidate_reasons": ["insufficient_stress_trades"],
        },
    )

    assert status == "FAST143_OBSERVATION_NO_ELIGIBLE_SHORT_TRADES_YET"
    assert "not_approved_strategy" in reasons

    summary = summarize_observations(
        pd.DataFrame([_row(entry_timestamp="2027-01-01T08:25:15Z")]),
        alpha=0.05,
    )
    gates = evaluate_fast143_gates(summary=summary, observations=pd.DataFrame([_row(entry_timestamp="2027-01-01T08:25:15Z")]))
    status, reasons = fast143_status(summary=summary, gates=gates)

    assert status == "FAST143_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    assert "no_order_execution" in reasons


def test_safety_flags_remain_research_only() -> None:
    assert SAFETY_FLAGS["research_only"] is True
    assert SAFETY_FLAGS["approved_strategy"] is False
    assert SAFETY_FLAGS["order_submission_allowed"] is False
    assert SAFETY_FLAGS["orders_sent"] is False
